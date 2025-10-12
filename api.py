
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import requests
import os
import traceback
import os
import requests
import base64
import json
app = Flask(__name__)
CORS(app)

def khoanh_mảnh(img, x_offset=0, min_area=1000):
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 80, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None, None, None, img
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return False, None, None, None, img
    x, y, w, h = cv2.boundingRect(cnt)
    bbox_global = (x + x_offset, y, w, h)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    cropped_bgr = img[y:y+h, x:x+w]
    mask_crop = mask[y:y+h, x:x+w]
    cropped = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2BGRA)
    cropped[:, :, 3] = mask_crop
    output = img.copy()
    overlay = np.zeros_like(img, dtype=np.uint8)
    overlay[:, :] = (0, 0, 255)
    mask_rgb = cv2.merge([mask, mask, mask])
    output = np.where(mask_rgb == 255, cv2.addWeighted(img, 0.5, overlay, 0.5, 0), img)
    return True, bbox_global, mask, cropped, output

def khoanh_nét_theo_hinh_dang(img, template_img, similarity_threshold=0.2, x_offset=0):
    imgg = img.copy()
    found = False
    bbox_global = None
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) if img.shape[2] == 4 else img
    if template_img.shape[2] == 4:
        template_bgr = cv2.cvtColor(template_img, cv2.COLOR_BGRA2BGR)
        mask = template_img[:, :, 3]
    else:
        template_bgr = template_img
        mask = None
    if mask is not None:
        res = cv2.matchTemplate(img_bgr, template_bgr, cv2.TM_CCORR_NORMED, mask=mask)
    else:
        res = cv2.matchTemplate(img_bgr, template_bgr, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val >= similarity_threshold:
        found = True
        x0, y0 = max_loc
        h, w = template_bgr.shape[:2]
        cv2.rectangle(imgg, (x0, y0), (x0 + w, y0 + h), (255, 0, 0), 3)
        bbox_global = (x0 + x_offset, y0, w, h)
    return found, bbox_global, imgg

def diem_manh_ghep_chuan(cnt):
    area = cv2.contourArea(cnt)
    if area < 50:
        return 0.0
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return 0.0
    compactness = 4 * np.pi * area / (peri * peri)
    compact_score = max(0, min(1, 1 - abs(compactness - 0.55)/0.55))
    x, y, w, h = cv2.boundingRect(cnt)
    ratio = w / h if h > 0 else 0
    aspect_score = max(0, min(1, 1 - abs(ratio - 1)/1.0))
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area > 0 else 0
    convex_score = max(0, min(1, 1 - abs(convexity - 0.9)/0.9))
    return (compact_score + aspect_score + convex_score)/3.0

def tim_manh_chuan_nhat(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_cnt, best_score = None, 0.0
    for cnt in contours:
        score = diem_manh_ghep_chuan(cnt)
        if score > best_score:
            best_score = score
            best_cnt = cnt
    return best_cnt, best_score

def sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()
def scale_x(x_old, old_width, new_width):
    return x_old * new_width / old_width





GITHUB_OWNER = "quycodethue"
GITHUB_REPO = "CaptchaTikTok"
FILE_PATH = "key.json"
BRANCH = "main"
TOKEN = os.environ.get("GITHUB_TOKEN")

if not TOKEN:
    raise SystemExit("Thiếu GITHUB_TOKEN trong environment variables")

headers = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github+json"
}

def get_file():
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{FILE_PATH}?ref={BRANCH}"
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()

def update_file(new_content_bytes, sha, message="Update key.json"):
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{FILE_PATH}"
    payload = {
        "message": message,
        "content": base64.b64encode(new_content_bytes).decode('utf-8'),
        "sha": sha,
        "branch": BRANCH
    }
    r = requests.put(url, headers=headers, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()

def handle_request(key):
    try:
        data = get_file()
    except requests.HTTPError as e:
        return {"ok": False, "msg": f"Lỗi khi lấy file: {e}"}, 500

    content_b64 = data.get("content", "")
    sha = data.get("sha")
    if not content_b64 or not sha:
        return {"ok": False, "msg": "File không hợp lệ trên repo"}, 500

    raw = base64.b64decode(content_b64)
    try:
        key_dict = json.loads(raw)
    except Exception:
        return {"ok": False, "msg": "Nội dung key.json không phải JSON"}, 500
    k = key_dict.get(key)
    if k is None:
        return {"ok": False, "msg": "Thiếu field key"}, 400
    if k <= 0:
        return {"ok": False, "msg": "Key hết hạn"}, 400
    key_dict[key] = k - 1
    new_raw = json.dumps(key_dict, ensure_ascii=False, indent=2).encode("utf-8")
    return new_raw,sha
@app.route("/process_captcha", methods=["POST"])
def process_captcha():
    """
    POST /api/process_captcha
    - form-data: "image" = file OR json { "b64": "..." }
    Trả về JSON:
      { ok: bool, msg: "", found: bool, distance: int, bbox_left: [...], bbox_right: [...], merged_image: "data:image/png;base64,..." }
    """
    

    try:
        data = request.get_json()
        if not data or "img" not in data or "key" not in data:
            return jsonify({"ok": False, "msg": "Không có trường 'img'"}), 400
        new_raw,sha = handle_request(data["key"])

        b64 = data["img"]
        if "base64," in b64: b64 = b64.split("base64,")[1]
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return jsonify({"ok": False, "msg": "Không đọc được ảnh"}), 400

        h, w = img.shape[:2]
        left, right = img[:, :w//3], img[:, w//3:]

        stt_left, bbox_left, mask_left, piece_img_left, original_img_left = khoanh_mảnh(left, x_offset=0)
        stt_right, bbox_right, mask_right, piece_img_right, original_img_right = khoanh_mảnh(right, x_offset=0)

        if stt_left is False or stt_right is False:
            if stt_left is False and stt_right is True:
                stt_left, bbox_left, original_img_left = khoanh_nét_theo_hinh_dang(left, piece_img_right)
            if stt_right is False and stt_left is True:
                stt_right, bbox_right, original_img_right = khoanh_nét_theo_hinh_dang(right, piece_img_left, x_offset=w//3)
        else:
            h1, w1 = piece_img_left.shape[:2]
            h2, w2 = piece_img_right.shape[:2]
            if h1 <= h2 and w1 <= w2:
                res = cv2.matchTemplate(piece_img_right, piece_img_left, cv2.TM_CCOEFF_NORMED)
            elif h2 <= h1 and w2 <= w1:
                res = cv2.matchTemplate(piece_img_left, piece_img_right, cv2.TM_CCOEFF_NORMED)
            else:
                if h1 * w1 < h2 * w2:
                    scale = min(h2 / h1, w2 / w1)
                    piece_img_left = cv2.resize(piece_img_left, (int(w1 * scale), int(h1 * scale)))
                    res = cv2.matchTemplate(piece_img_right, piece_img_left, cv2.TM_CCOEFF_NORMED)
                else:
                    scale = min(h1 / h2, w1 / w2)
                    piece_img_right = cv2.resize(piece_img_right, (int(w2 * scale), int(h2 * scale)))
                    res = cv2.matchTemplate(piece_img_left, piece_img_right, cv2.TM_CCOEFF_NORMED)


        if not (stt_left or stt_right):
            return jsonify({"ok": False, "msg": "Không tìm thấy mảnh ghép"}), 400
        if not bbox_left or not bbox_right:
            return jsonify({"ok": False, "msg": "Không tìm thấy mảnh ghép"}), 400

        merged = cv2.hconcat([original_img_left, original_img_right])
        x1, y1, w1, h1 = bbox_left
        x2, y2, w2, h2 = bbox_right
        cx1, cy1 = x1 + w1//2, y1 + h1//2
        cx2, cy2 = x2 + w2//2, y2 + h2//2
        height_l, width_l = left.shape[:2]
        height_r, width_r = right.shape[:2]

        center_x_l, center_y_l = x1 + w1 / 2, y1 + h1 / 2
        center_x_r, center_y_r = x2 + w2 / 2, y2 + h2 / 2
        distance = int(width_l-center_x_l) + int(center_x_r)
        pt1 = (int(center_x_l), int(center_y_l))
        pt2 = (int(center_x_r), int(center_y_r))
        mid_pt = (int((center_x_l + center_x_r) / 2), int((center_y_l + center_y_r) / 2))
        cv2.line(merged, (cx1, cy1), (cx2+w//3, cy2), (0, 255, 0), 2)
        cv2.putText(merged, f"{distance}px", ((cx1+cx2)//2, (cy1+cy2)//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        _, buf = cv2.imencode('.png', merged)
        merged_b64 = base64.b64encode(buf.tobytes()).decode('ascii')
        merged_data_uri = f"data:image/png;base64,{merged_b64}"

        resp = {
            "khoảng cách": distance,
            'số lần dùng key còn': int(data["key"]) - 1,
            "img": merged_data_uri
        }
        update_file(new_raw, sha)
        return jsonify(resp), 200

    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)