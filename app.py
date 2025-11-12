import base64
import json
import os

import cv2
import mediapipe as mp
import numpy as np

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from io import BytesIO

# Import custom modules (assuming they are in the same directory)
from affine_transformation import warp_with_triangulation
from anti_nasolabial import nasolabial_folds_filter
from anti_black import black_filter
from bg import bg_blur
from dayan import dayan
from dazui import dazui
from forehead import forehead
from long_nose import longnose
from meibai import apply_whitening_and_blend
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
from test_speed import get_segmentation_mask

mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Constants
EYE_INDICES = [374, 380, 390, 373, 249, 385, 384, 263, 466, 387, 386, 381, 382, 398, 388, 362, 154, 155, 33, 7, 246,
               161, 159, 158, 144, 145, 173, 133, 157, 163, 153, 160]
BOUNDARY = [270, 409, 317, 402, 81, 82, 91, 181, 37, 0, 84, 17, 269, 321, 375, 318, 324, 312, 311, 415, 308, 314, 61,
            146, 78, 95, 267, 13, 405, 178, 87, 185, 14, 88, 40, 291, 191, 310, 39, 80, 4, 334, 296, 276, 283, 293,
            295, 285, 336, 282, 300, 46, 53, 66, 107, 52, 65, 63, 105, 70, 55, 374, 380, 390, 373, 249, 385, 384,
            263, 466, 387, 386, 381, 382, 398, 388, 362, 154, 155, 33, 7, 246, 161, 159, 158, 144, 145, 173, 133,
            157, 163, 153, 160, 132, 58, 172, 136, 150, 149, 176, 148, 361, 288, 397, 365, 379, 378, 400, 377, 152,
            473, 468, 116, 123, 345, 352, 103, 67, 109, 10, 338, 297, 332, 126, 129, 98, 75, 166, 79, 239, 238, 20,
            60, 355, 358, 327, 289, 392, 309, 459, 458, 250, 290, 97, 2, 327, 326, 48, 115, 220, 45, 275, 440, 344,
            278, 280, 50, 389, 9, 162]
SHOULIAN_INDICES = [132, 58, 172, 136, 150, 149, 176, 148, 361, 288, 397, 365, 379, 378, 400, 377, 152, 389, 9, 162,
                    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 263, 249, 390, 373,
                    374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

POINT_OFFSET = 500

# Defaults
defaults = {
    'DAYAN': 1, 'SHOULIAN': 1, 'QUANGU': 1, 'ZHAILIAN': 1, 'RENZHONG': 0,
    'BIYI': 1, 'LONGNOSE': 0.0, 'FOREHEAD': 0, 'ZUIJIAO': 0.0, 'DAZUI': 1,
    'S': 0, 'V': 0, 'MOPI': 0, 'DETAIL': 0.0, 'FALING': 0, 'BLACK': 0,
    'WHITE_EYE': 1, 'WHITE_TEETH': 1, 'LIPSTICK': 0.0, 'BLUSH': 0.0,
}

all_params = list(defaults.keys())

# Global state (per-request reset in endpoint for new images)
cached_face_landmarks = None
cached_segmentation_mask = None
cached_image_shape = None
cached_face_hulls = None
selected_faces = set()
face_params = {}
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
    """Run face mesh inference once"""
    global cached_face_landmarks, cached_segmentation_mask, cached_image_shape, cached_face_hulls

    if image is None:
        return None, None

    cached_image_shape = image.shape
    h, w = image.shape[:2]

    # Face mesh inference
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(image_rgb)
        cached_face_landmarks = face_results.multi_face_landmarks if face_results.multi_face_landmarks else []

    # Pre-compute convex hulls
    if cached_face_landmarks:
        cached_face_hulls = compute_face_hulls(cached_face_landmarks, w, h)
    else:
        cached_face_hulls = []

    # Segmentation masks (optimized - only when needed)
    all_masks = np.zeros((len(cached_face_landmarks), h, w), dtype=np.uint8)
    if cached_face_landmarks:
        i = 0
        for lm in cached_face_landmarks:
            try:
                x = int(lm.landmark[4].x * w)
                y = int(lm.landmark[4].y * h)
                negatives = get_points_2D((lm.landmark[468],lm.landmark[473],lm.landmark[17],lm.landmark[15]), w, h)
                mask = get_segmentation_mask(image, (x, y), negatives)
                if mask is not None:
                    all_masks[i] = mask
                    i += 1

            except Exception as e:
                pass
    cached_segmentation_mask = all_masks
    return cached_face_landmarks, cached_segmentation_mask


def process_image(image, selected_faces, apply_bg_blur=False):
    """Optimized image processing pipeline"""
    global cached_face_landmarks, face_params

    if image is None or not cached_face_landmarks or not selected_faces:
        return bg_blur(image) if apply_bg_blur else image

    h_img, w_img = image.shape[:2]
    num_faces = len(cached_face_landmarks)

    # Corner points setup
    corner_indices = [-1, -2, -3, -4]
    corner_points = [
        transform_to_3d(0, 0, 1),
        transform_to_3d(w_img - 1, 0, 1),
        transform_to_3d(w_img - 1, h_img - 1, 1),
        transform_to_3d(0, h_img - 1, 1)
    ]
    corners_dict = dict(zip(corner_indices, corner_points))

    # Pre-compute all face points once
    all_face_points = [get_points_all(lm.landmark, w_img, h_img) for lm in cached_face_landmarks]
    # Apply filters (batch HSV conversion)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    processed_image = image.copy()  # Initialize processed_image outside the loop

    for face_idx in selected_faces:
        p = face_params[face_idx]
        all_2d = get_points_2D(cached_face_landmarks[face_idx].landmark, w_img, h_img)
        mask = cached_segmentation_mask[face_idx]

        processed_image = apply_whitening_and_blend(processed_image, mask, hsv_image, 0, p['S'], p['V'], p['MOPI'])

        if p['DETAIL'] > 0:
            processed_image = enhance_face_detail(processed_image, all_2d, strength_pct=p['DETAIL'])
        if p['FALING'] > 0:
            processed_image = nasolabial_folds_filter(processed_image, all_2d, p['FALING'])
        if p['BLACK'] > 0:
            processed_image = black_filter(processed_image, all_2d, p['BLACK'])
        if p['WHITE_EYE'] != 1:
            processed_image = white_eyes(processed_image, all_2d, p['WHITE_EYE'])
        if p['WHITE_TEETH'] != 1:
            processed_image = white_teeth(processed_image, all_2d, p['WHITE_TEETH'])
        if p['LIPSTICK'] > 0:
            processed_image = apply_lipstick(processed_image, all_2d, p['LIPSTICK'])
        if p['BLUSH'] > 0:
            processed_image = apply_blush(processed_image, all_2d, color=(157, 107, 255), intensity=p['BLUSH'])

    # First warp: shoulian
    shoulian_src = corners_dict.copy()
    shoulian_dst = corners_dict.copy()

    for face_idx in range(num_faces):
        base = face_idx * POINT_OFFSET
        face_all_point = all_face_points[face_idx]
        for local in SHOULIAN_INDICES:
            gidx = base + local
            pt = face_all_point[local]
            shoulian_src[gidx] = pt
            shoulian_dst[gidx] = pt

    for face_idx in selected_faces:
        p = face_params[face_idx]
        base = face_idx * POINT_OFFSET
        local_dst = {local: shoulian_dst[base + local] for local in SHOULIAN_INDICES}
        shoulian(p['SHOULIAN'], local_dst)
        for local in SHOULIAN_INDICES:
            shoulian_dst[base + local] = local_dst[local]

    # Restrict shoulian warp to face area via soft mask blending
    pre_warp = processed_image.copy()
    warped = warp_with_triangulation(processed_image, shoulian_src, shoulian_dst)
    if warped is None:
        return bg_blur(image) if apply_bg_blur else image

    # Build union mask of selected face hulls
    mask = np.zeros(pre_warp.shape[:2], dtype=np.uint8)
    if cached_face_hulls is not None:
        for face_idx in selected_faces:
            if face_idx < len(cached_face_hulls):
                hull = cached_face_hulls[face_idx]
                if hull is not None and len(hull) >= 3:
                    cv2.fillConvexPoly(mask, hull, 255)
    # Feather mask to avoid seams
    if np.any(mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        alpha = (mask.astype(np.float32) / 255.0)[..., np.newaxis]
        processed_image = (pre_warp.astype(np.float32) * (1 - alpha) + warped.astype(np.float32) * alpha).astype(np.uint8)
    else:
        processed_image = warped

    # Second warp: facial adjustments
    src_points = corners_dict.copy()
    dst_points = corners_dict.copy()

    for face_idx in range(num_faces):
        base = face_idx * POINT_OFFSET
        face_all_point = all_face_points[face_idx]
        for local in BOUNDARY:
            gidx = base + local
            pt = face_all_point[local]
            src_points[gidx] = pt
            dst_points[gidx] = pt

    for face_idx in selected_faces:
        p = face_params[face_idx]
        base = face_idx * POINT_OFFSET
        local_dst = {local: dst_points[base + local] for local in BOUNDARY}

        # Apply all transformations
        dayan(p['DAYAN'], local_dst)
        quangu(p['QUANGU'] * p['ZHAILIAN'], local_dst)
        biyi(p['BIYI'], local_dst)
        longnose(p['LONGNOSE'], local_dst)
        zhailian(p['ZHAILIAN'], local_dst)
        renzhong(p['RENZHONG'], local_dst)
        forehead(p['FOREHEAD'], local_dst)
        zuijiao(p['ZUIJIAO'], local_dst)
        dazui(p['DAZUI'], local_dst)

        for local in BOUNDARY:
            dst_points[base + local] = local_dst[local]

    processed_image = warp_with_triangulation(processed_image, src_points, dst_points)
    if processed_image is None:
        return processed_image

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


@app.route('/upload', methods=['POST'])
def upload_image():
    global cached_face_landmarks, cached_segmentation_mask, cached_image_shape, cached_face_hulls, selected_faces, face_params, bg_blur_enabled, global_current_image

    array_param_str = request.form.get('array_param', json.dumps(list(defaults.values())))  # Default to defaults if not provided
    try:
        param_array = json.loads(array_param_str)
        if not isinstance(param_array, list) or len(param_array) != len(all_params):
            raise ValueError(f"array_param must be a JSON array of length {len(all_params)}")
    except (json.JSONDecodeError, ValueError) as e:
        return jsonify({'error': f'Invalid array_param: {str(e)}'}), 400

    # Map array to params (assuming order matches all_params)
    param_dict = dict(zip(all_params, param_array))
    face_params[0] = param_dict

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
        cached_face_landmarks = None
        cached_segmentation_mask = None
        cached_image_shape = None
        cached_face_hulls = None
        selected_faces = set()

        # Run inference
        run_model_inference(image)

        # Auto-select first face if detected
        if cached_face_landmarks:
            selected_faces.add(0)
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
        'received_array': param_array
    }), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
