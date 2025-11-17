import base64
import json
import os

import cv2
import mediapipe as mp
import numpy as np

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from io import BytesIO

# Local debug flag (avoid importing GUI to prevent Tkinter side effects)
ROI_DEBUG_DRAW = False

# Import custom modules (assuming they are in the same directory)
from affine_transformation import warp_with_triangulation
from anti_nasolabial import nasolabial_folds_filter
from anti_black import black_filter
from bg import bg_blur
from dayan import dayan
from dazui import dazui
from forehead import forehead
from long_nose import longnose
from main import apply_big_eye_effect
from meibai import apply_whitening_and_blend, get_eye_mask
from renzhong import renzhong
from shoulian import shoulian
from quangu import quangu
from zhailian import zhailian
from biyi import biyi
from res import enhance_face_detail
from white_eye import white_eyes
from white_teeth import white_teeth
from lipstick import lipstick as apply_lipstick
from blush import apply_blush
from zuijiao import zuijiao
from skin_segmenter import get_exposed_skin_mask

mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_face_detection = mp.solutions.face_detection

# Constants (aligned with GUI.py)
EYE_INDICES = [374, 380, 390, 373, 249, 385, 384, 263, 466, 387, 386, 381, 382, 398, 388, 362, 154, 155, 33, 7, 246,
               161, 159, 158, 144, 145, 173, 133, 157, 163, 153, 160]
BOUNDARY = [270, 409, 317, 402, 81, 82, 91, 181, 37, 0, 84, 17, 269, 321, 375, 318, 324, 312, 311, 415, 308, 314, 61,
            146, 78, 95, 267, 13, 405, 178, 87, 185, 14, 88, 40, 291, 191, 310, 39, 80,
            4,
            334, 296, 276, 283, 293, 295, 285, 336, 282, 300, 46, 53, 66, 107, 52, 65, 63, 105, 70, 55,
            374, 380, 390, 373, 249, 385, 384, 263, 466, 387, 386, 381, 382, 398, 388, 362, 154, 155, 33, 7, 246, 161,
            159, 158, 144, 145, 173, 133, 157, 163, 153, 160,
            132, 58, 172, 136, 150, 149, 176, 148, 361, 288, 397, 365, 379, 378, 400, 377, 152,
            473, 468,
            116, 123, 345, 352, 103, 67,
            109, 10, 338, 297, 332, 126, 129, 98, 75, 166, 79, 239, 238, 20, 60, 355, 358, 327, 289, 392, 309, 459, 458, 250, 290, 97, 2, 327, 326,
            48, 115, 220, 45, 275, 440, 344, 278, 280, 50, 389, 9, 162]
SHOULIAN_INDICES = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 323, 454, 361, 288, 397, 365, 379, 378, 400, 377, 152, 389, 9, 162,
                    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 263, 249, 390, 373,
                    374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

# Defaults
defaults = {
    'DAYAN': 1, 'SHOULIAN': 1, 'QUANGU': 1, 'ZHAILIAN': 1, 'RENZHONG': 0,
    'BIYI': 1, 'LONGNOSE': 0.0, 'FOREHEAD': 0, 'ZUIJIAO': 0.0, 'DAZUI': 1,
    'S': 0, 'V': 0, 'MOPI': 0, 'DETAIL': 0.0, 'FALING': 0, 'BLACK': 0,
    'WHITE_EYE': 1, 'WHITE_TEETH': 1, 'LIPSTICK': 0.0, 'BLUSH': 0.0,
}

all_params = list(defaults.keys())

# Global state (per-request reset in endpoint for new images)
cached_face_landmarks = []
cached_face_landmarks_crops = []
cached_segmentation_mask = None
cached_image_shape = None
cached_face_hulls = []
cached_face_bboxes = []
selected_faces = set()
face_params = {}
shared_params = {'S': 0, 'V': 0, 'MOPI': 0}
bg_blur_enabled = False
global_current_image = None


def transform_to_3d(x, y, z):
    """Cached 3D transformation"""
    return (x * z, y * z, z)


def get_points_all(landmarks, width, height):
    """Optimized point extraction with numpy"""
    points = np.array([(lm.x * width, lm.y * height, lm.z + 1) for lm in landmarks])
    return [(transform_to_3d(int(p[0]), int(p[1]), p[2])) for p in points]


def get_points_2D(landmarks, width, height):
    """Optimized 2D point extraction"""
    return [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]


def get_dynamic_upsample_factor(face_width: int, face_height: int) -> int:
    """Pick an upsample factor based on the smaller face dimension."""
    size = min(face_width, face_height)
    if size < 120:
        return 4
    if size < 180:
        return 3
    if size < 300:
        return 2
    return 1


class _SimpleLandmark:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class _SimpleLandmarkList:
    def __init__(self, landmarks):
        self.landmark = landmarks


def _to_full_image_landmarks(face_lm_list, bbox, img_w, img_h):
    """Map crop-normalized landmarks to normalized full-image coordinates."""
    xmin, ymin, bw, bh = bbox
    landmarks = []
    for lm in face_lm_list.landmark:
        abs_x = xmin + (lm.x * bw)
        abs_y = ymin + (lm.y * bh)
        x_norm = abs_x / float(img_w) if img_w else 0.0
        y_norm = abs_y / float(img_h) if img_h else 0.0
        landmarks.append(_SimpleLandmark(x_norm, y_norm, lm.z))
    return _SimpleLandmarkList(landmarks)


def _clone_normalized_landmarks(face_lm_list):
    """Clone a MediaPipe landmark list as a lightweight _SimpleLandmarkList."""
    landmarks = []
    for lm in face_lm_list.landmark:
        landmarks.append(_SimpleLandmark(lm.x, lm.y, lm.z))
    return _SimpleLandmarkList(landmarks)


def compute_face_hulls(landmarks_list, w, h):
    """Pre-compute convex hulls for all faces"""
    hulls = []
    for lm in landmarks_list:
        all_2d = get_points_2D(lm.landmark, w, h)
        pts = np.array(all_2d, dtype=np.int32)
        if len(pts) >= 3:
            hull = cv2.convexHull(pts)
            hulls.append(hull)
        else:
            hulls.append(None)
    return hulls


def run_model_inference(image):
    """Run Mediapipe Face Detection and FaceMesh per detected face crop."""
    global cached_face_landmarks, cached_face_landmarks_crops, cached_segmentation_mask, cached_image_shape, cached_face_hulls, cached_face_bboxes

    if image is None:
        return None, None

    cached_face_landmarks = []
    cached_face_landmarks_crops = []
    cached_face_hulls = []
    cached_face_bboxes = []

    cached_image_shape = image.shape
    h, w = image.shape[:2]

    # Face detection to define higher quality crops
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        faces = face_detection.process(image_rgb)

    face_bboxes = []
    face_crops = []
    if faces and faces.detections:
        for detection in faces.detections:
            bbox_rel = detection.location_data.relative_bounding_box
            xmin_pix = int(bbox_rel.xmin * w)
            ymin_pix = int(bbox_rel.ymin * h)
            width_pix = int(bbox_rel.width * w)
            height_pix = int(bbox_rel.height * h)

            cx = xmin_pix + width_pix / 2.0
            cy = ymin_pix + height_pix / 2.0
            new_w = int(round(width_pix * 1.2))
            new_h = int(round(height_pix * 2.0))
            new_xmin = int(round(cx - new_w / 2.0))
            new_ymin = int(round(cy - new_h / 2.0))

            x0 = max(0, min(new_xmin, w - 1))
            y0 = max(0, min(new_ymin, h - 1))
            x1 = max(x0 + 1, min(new_xmin + new_w, w))
            lower_half = int(round(cy + 1.2 * (height_pix / 2.0)))
            y1 = max(y0 + 1, min(lower_half, h))

            final_w = max(1, x1 - x0)
            final_h = max(1, y1 - y0)
            face_crop = image[y0:y0 + final_h, x0:x0 + final_w]
            if face_crop.size == 0:
                continue
            face_crops.append(face_crop)
            face_bboxes.append((x0, y0, final_w, final_h))

    # Run mesh per crop (or fallback to whole image)
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        if face_crops:
            for crop, bbox in zip(face_crops, face_bboxes):
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_results = face_mesh.process(crop_rgb)
                if not crop_results.multi_face_landmarks:
                    continue
                for lm_list in crop_results.multi_face_landmarks:
                    mapped_full = _to_full_image_landmarks(lm_list, bbox, w, h)
                    crop_norm = _clone_normalized_landmarks(lm_list)
                    cached_face_landmarks.append(mapped_full)
                    cached_face_landmarks_crops.append(crop_norm)
        else:
            full_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(full_rgb)
            if face_results.multi_face_landmarks:
                for lm_list in face_results.multi_face_landmarks:
                    landmarks = [_SimpleLandmark(lm.x, lm.y, lm.z) for lm in lm_list.landmark]
                    lmlist = _SimpleLandmarkList(landmarks)
                    cached_face_landmarks.append(lmlist)
                    cached_face_landmarks_crops.append(lmlist)
                if cached_face_landmarks:
                    face_bboxes = [(0, 0, w, h)] * len(cached_face_landmarks)

    if face_bboxes:
        cached_face_bboxes = face_bboxes
    elif cached_face_landmarks:
        cached_face_bboxes = [(0, 0, w, h)] * len(cached_face_landmarks)

    if cached_face_landmarks:
        cached_face_hulls = compute_face_hulls(cached_face_landmarks, w, h)
    else:
        cached_face_hulls = []

    cached_segmentation_mask = get_exposed_skin_mask(image_rgb)
    return cached_face_landmarks, cached_segmentation_mask

def process_image(image, selected_faces, apply_bg_blur=False):
    """Process image per cropped face ROI and composite back."""
    global cached_face_landmarks, cached_face_landmarks_crops, cached_face_bboxes, face_params, cached_segmentation_mask

    if image is None or not cached_face_landmarks or not selected_faces:
        return bg_blur(image) if apply_bg_blur else image

    h_img, w_img = image.shape[:2]

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    eye_mask = np.zeros((h_img, w_img), np.uint8)
    for face_idx in selected_faces:
        if face_idx >= len(cached_face_landmarks):
            continue
        all_2d_full = get_points_2D(cached_face_landmarks[face_idx].landmark, w_img, h_img)
        eye_mask = cv2.bitwise_or(eye_mask, get_eye_mask(image, all_2d_full))

    processed_image = apply_whitening_and_blend(
        image,
        eye_mask,
        cached_segmentation_mask,
        hsv_image,
        0,
        shared_params['S'],
        shared_params['V'],
        shared_params['MOPI']
    )

    for face_idx in selected_faces:
        if face_idx >= len(cached_face_bboxes) or face_idx >= len(cached_face_landmarks_crops):
            continue

        bbox = cached_face_bboxes[face_idx]
        if not bbox:
            continue
        x, y, w_crop, h_crop = bbox
        if w_crop <= 1 or h_crop <= 1:
            continue

        params = face_params.get(face_idx)
        if not params:
            params = {k: v for k, v in defaults.items() if k not in ['S', 'V', 'MOPI']}
            face_params[face_idx] = params

        scale = get_dynamic_upsample_factor(w_crop, h_crop)
        w_big = w_crop * scale
        h_big = h_crop * scale

        roi = processed_image[y:y + h_crop, x:x + w_crop].copy()
        if roi.size == 0:
            continue
        if scale > 1:
            roi_big = cv2.resize(roi, (w_big, h_big), interpolation=cv2.INTER_NEAREST)
        else:
            roi_big = roi.copy()

        lm_crop = cached_face_landmarks_crops[face_idx]

        def lm_to_big(landmark_list):
            return [(lm.x * w_big, lm.y * h_big) for lm in landmark_list.landmark]

        def lm3d_to_big(landmark_list):
            return [(lm.x * w_big * (lm.z + 1), lm.y * h_big * (lm.z + 1), lm.z + 1)
                    for lm in landmark_list.landmark]

        all_2d_big = lm_to_big(lm_crop)
        face_points3d_big = lm3d_to_big(lm_crop)

        shoulian_src = {i: face_points3d_big[i] for i in SHOULIAN_INDICES if i < len(face_points3d_big)}
        shoulian_dst = shoulian_src.copy()
        local_dst = dict(shoulian_dst)
        shoulian(params['SHOULIAN'], local_dst)
        for k in local_dst:
            shoulian_dst[k] = local_dst[k]
        roi_big = warp_with_triangulation(roi_big, shoulian_src, shoulian_dst)

        src_points = {i: face_points3d_big[i] for i in BOUNDARY if i < len(face_points3d_big)}
        dst_points = src_points.copy()
        local_dst = dict(dst_points)
        quangu(params['QUANGU'] * params['ZHAILIAN'], local_dst)
        biyi(params['BIYI'], local_dst)
        longnose(params['LONGNOSE'], local_dst)
        zhailian(params['ZHAILIAN'], local_dst)
        renzhong(params['RENZHONG'], local_dst)
        forehead(params['FOREHEAD'], local_dst)
        zuijiao(params['ZUIJIAO'], local_dst)
        dazui(params['DAZUI'], local_dst)
        for k in local_dst:
            dst_points[k] = local_dst[k]
        roi_big = warp_with_triangulation(roi_big, src_points, dst_points)

        if params['DETAIL'] > 0:
            roi_big = enhance_face_detail(roi_big, all_2d_big, strength_pct=params['DETAIL'])
        if params['FALING'] > 0:
            roi_big = nasolabial_folds_filter(roi_big, all_2d_big, params['FALING'])
        if params['BLACK'] > 0:
            roi_big = black_filter(roi_big, all_2d_big, params['BLACK'])
        if params['WHITE_EYE'] != 1:
            roi_big = white_eyes(roi_big, all_2d_big, params['WHITE_EYE'])
        if params['WHITE_TEETH'] != 1:
            roi_big = white_teeth(roi_big, all_2d_big, params['WHITE_TEETH'])
        if params['LIPSTICK'] > 0:
            roi_big = apply_lipstick(roi_big, all_2d_big, params['LIPSTICK'])
        if params['BLUSH'] > 0:
            roi_big = apply_blush(roi_big, all_2d_big, color=(157, 107, 255), intensity=params['BLUSH'])
        roi_big = apply_big_eye_effect(roi_big, params['DAYAN'], all_2d_big)

        if scale > 1:
            roi_final = cv2.resize(roi_big, (w_crop, h_crop), interpolation=cv2.INTER_AREA)
        else:
            roi_final = roi_big

        processed_image[y:y + h_crop, x:x + w_crop] = roi_final

    if apply_bg_blur:
        processed_image = bg_blur(processed_image)

    return processed_image

app = Flask(__name__)

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 最大 16MB

# 创建上传目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/beauty', methods=['POST'])
def upload_image():
    global cached_face_landmarks, cached_face_landmarks_crops, cached_face_bboxes, cached_segmentation_mask, cached_image_shape, cached_face_hulls, selected_faces, face_params, shared_params, bg_blur_enabled, global_current_image

    array_param_str = request.form.get('array_param', json.dumps(defaults))  # Default to defaults if not provided
    try:
        provided = json.loads(array_param_str)
        if not isinstance(provided, dict):
            raise ValueError("array_param must be a JSON object mapping parameter names to values")
    except (json.JSONDecodeError, ValueError) as e:
        return jsonify({'error': f'Invalid array_param: {str(e)}'}), 400

    # Keep only recognized keys
    recognized_keys = set(defaults.keys())
    provided = {k: provided[k] for k in provided.keys() if k in recognized_keys}

    # Shared vs per-face updates (support partial updates)
    shared_updates = {k: provided[k] for k in ['S', 'V', 'MOPI'] if k in provided}
    per_face_updates = {k: v for k, v in provided.items() if k not in ['S', 'V', 'MOPI']}

    # Apply shared updates immediately
    if shared_updates:
        shared_params.update(shared_updates)

    # Get bg_blur from form or default False
    bg_blur_enabled = request.form.get('bg_blur', 'false').lower() == 'true'

    is_new_image = 'file' in request.files and request.files['file'].filename != ''

    if is_new_image:
        file = request.files['file']

        # 检查文件名是否为空
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # 验证文件类型（可选：只允许图片）
        if file and not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            return jsonify({'error': 'Invalid file type'}), 400

        # 安全文件名
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        file.save(filepath)

        # Load and resize image if needed
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        h, w = image.shape[:2]
        if w >= 1920 or h >= 1080:
            image = cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

        global_current_image = image.copy()

        # Reset processing globals for new image
        cached_face_landmarks = []
        cached_face_landmarks_crops = []
        cached_face_bboxes = []
        cached_segmentation_mask = None
        cached_image_shape = None
        cached_face_hulls = []
        selected_faces = set()
        face_params = {}

        # Run inference
        run_model_inference(image)

        # Auto-select all faces if detected (align with GUI)
        if cached_face_landmarks:
            selected_faces = set(range(len(cached_face_landmarks)))
            # Initialize per-face params from defaults, then apply provided updates
            face_params = {i: {k: v for k, v in defaults.items() if k not in ['S','V','MOPI']} for i in selected_faces}
            if per_face_updates:
                for i in selected_faces:
                    face_params[i].update(per_face_updates)
        else:
            os.remove(filepath)
            return jsonify({'error': 'No faces detected'}), 400

        # Clean up
        os.remove(filepath)

        input_image = image
    else:
        # Parameter update only
        if global_current_image is None:
            return jsonify({'error': 'No image loaded. Please upload an image first.'}), 400

        input_image = global_current_image
        # No reset; use existing inference results
        if cached_face_landmarks and not selected_faces:
            selected_faces = set(range(len(cached_face_landmarks)))
            face_params = {i: {k: v for k, v in defaults.items() if k not in ['S','V','MOPI']} for i in selected_faces}
        # Apply only provided per-face updates to selected faces
        if per_face_updates:
            for i in list(selected_faces):
                if i not in face_params:
                    face_params[i] = {k: v for k, v in defaults.items() if k not in ['S','V','MOPI']}
                face_params[i].update(per_face_updates)

    # Process image
    processed_image = process_image(input_image, selected_faces, apply_bg_blur=bg_blur_enabled)
    if processed_image is None:
        return jsonify({'error': 'Image processing failed'}), 500

    # Encode to base64
    _, buffer = cv2.imencode('.jpg', processed_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'message': 'Image processed successfully',
        'image_base64': f'data:image/jpeg;base64,{img_base64}',
        'received_params': provided
    }), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
