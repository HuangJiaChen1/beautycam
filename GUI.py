import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from functools import lru_cache
import math

from affine_transformation import warp_with_triangulation, warp_with_triangulation_2d
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
from skin_segmenter import get_exposed_skin_mask
from zhailian import zhailian
from biyi import biyi
from res import enhance_face_detail
from white_eye import white_eyes
from white_teeth import white_teeth
from lipstick import lipstick as apply_lipstick
from blush import apply_blush
from zuijiao import zuijiao

mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_face_detection = mp.solutions.face_detection
#TODO: MASK NEEDS TO TRANSFORM AS WELL

# Constants
EYE_INDICES = [374, 380, 390, 373, 249, 385, 384, 263, 466, 387, 386, 381, 382, 398, 388, 362, 154, 155, 33, 7, 246,
               161, 159, 158, 144, 145, 173, 133, 157, 163, 153, 160]
BOUNDARY = [270, 409, 317, 402, 81, 82, 91, 181, 37, 0, 84, 17, 269, 321, 375, 318, 324, 312, 311, 415, 308, 314, 61,
            146, 78, 95, 267, 13, 405, 178, 87, 185, 14, 88, 40, 291, 191, 310, 39, 80, #LIPS AND TEETH

            4, #NOSETIP

            334, 296, 276, 283, 293, 295, 285,336, 282, 300,46, 53, 66, 107, 52, 65, 63, 105, 70, 55, #EYEBROWS

            374, 380,390, 373, 249, 385, 384, 263, 466, 387,386, 381, 382, 398, 388, 362, 154, 155, 33, 7, 246, 161,
            159, 158, 144, 145, 173, 133, 157, 163, 153, 160, #LEFT RIGHT EYE

            132,58,172, 136, 150, 149, 176, 148,361,288,397, 365, 379, 378, 400,377,152, #JAWLINE

            473,468, #IRIS

            116,123,345,352,103,67, #LEFT RIGHT QUANGU

            109,10,338,297,332,126,129,98,75,166,79,239,238,20,60,355,358,327,289,392,309,459,458,250,290,97,2,327,326,
            48,115,220,45,275,440,344,278,280,50,389,9,162]
SHOULIAN_INDICES = [234,93,132, 58, 172, 136, 150, 149, 176, 148,323,454, 361, 288, 397, 365, 379, 378, 400, 377, 152, 389, 9, 162,
                    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 263, 249, 390, 373,
                    374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

POINT_OFFSET = 500
SMALL_FACE_THRESHOLD = 120
UPSAMPLE_FACTOR = 3
# Debug flag: draw per-face ROI rectangles (easy to toggle off)
ROI_DEBUG_DRAW = True

defaults = {
    'DAYAN': 1, 'SHOULIAN': 1, 'QUANGU': 1, 'ZHAILIAN': 1, 'RENZHONG': 0,
    'BIYI': 1, 'LONGNOSE': 0.0, 'FOREHEAD': 0, 'ZUIJIAO': 0.0, 'DAZUI': 1,
    'S': 0, 'V': 0, 'MOPI': 0, 'DETAIL': 0.0, 'FALING': 0, 'BLACK': 0,
    'WHITE_EYE': 1, 'WHITE_TEETH': 1, 'LIPSTICK': 0.0, 'BLUSH': 0.0,
}

all_params = list(defaults.keys())

# Store per-face landmarks as a list
cached_face_landmarks = []
# Per-face landmarks in crop-normalized coordinates (for crop warping)
cached_face_landmarks_crops = []
cached_faces = []
cached_face_bboxes = []  # (x, y, w, h) for each cached face crop
cached_segmentation_mask = None
cached_image_shape = None
cached_face_hulls = None
cached_eye_lips = None
selected_faces = set()
face_params = {}
shared_params = {'S': 0, 'V': 0, 'MOPI': 0}
active_face = None
bg_blur_enabled = False
pending_update = None


def transform_to_3d(x, y, z):
    """Cached 3D transformation"""
    return (x * z, y * z, z)


def get_points_all(landmarks, width, height):
    """Optimized point extraction with numpy"""
    points = np.array([(lm.x * width, lm.y * height, lm.z + 1) for lm in landmarks])
    return [(transform_to_3d(int(p[0]), int(p[1]), p[2])) for p in points]


def get_dynamic_upsample_factor(face_width: int, face_height: int) -> int:
    """
    Dynamically choose the upsample factor based on face size.
    - < 120px  → 4×
    - 120~179px → 3×
    - 180~299px → 2×
    - ≥ 300px   → 1× (no upsample)

    All factors are integer → INTER_NEAREST up + INTER_AREA down = pixel-perfect roundtrip
    """
    size = min(face_width, face_height)  # use the smaller dimension (more conservative)

    if size < 120:
        return 4
    elif size < 180:
        return 3
    elif size < 300:
        return 2
    else:
        return 1

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


def on_click(event):
    """Optimized click handler using cached hulls"""
    global selected_faces, face_params, active_face, cached_face_landmarks, cached_face_hulls

    if not cached_face_landmarks or cached_image_shape is None:
        return

    h, w = cached_image_shape[:2]
    cx, cy = event.x, event.y

    # Check cached hulls first
    for idx, hull in enumerate(cached_face_hulls):
        if hull is None:
            continue
        dist = cv2.pointPolygonTest(hull, (int(cx), int(cy)), False)
        if dist >= 0:
            if idx not in selected_faces:
                selected_faces.add(idx)
                face_params[idx] = {k: v for k, v in defaults.items() if k not in ['S', 'V', 'MOPI']}
            update_active_face(idx)
            schedule_update()
            return

    # Fallback to nearest nose
    min_dist = float('inf')
    closest_idx = -1
    for idx, lm in enumerate(cached_face_landmarks):
        nose_x = int(lm.landmark[1].x * w)
        nose_y = int(lm.landmark[1].y * h)
        dist = (cx - nose_x) ** 2 + (cy - nose_y) ** 2
        if dist < min_dist:
            min_dist = dist
            closest_idx = idx

    if min_dist < 10000:  # 100^2
        if closest_idx not in selected_faces:
            selected_faces.add(closest_idx)
            face_params[closest_idx] = {k: v for k, v in defaults.items() if k not in ['S', 'V', 'MOPI']}
        update_active_face(closest_idx)
        schedule_update()


class _SimpleLandmark:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class _SimpleLandmarkList:
    def __init__(self, landmarks):
        self.landmark = landmarks


def _to_full_image_landmarks(face_lm_list, bbox, img_w, img_h):
    """Map landmarks from a face crop to normalized full-image coordinates.
    bbox: (xmin, ymin, width, height) in absolute pixel units.
    Returns _SimpleLandmarkList with .landmark list containing x,y normalized to full image.
    """
    xmin, ymin, bw, bh = bbox
    landmarks = []
    for lm in face_lm_list.landmark:
        # lm.x, lm.y are normalized to the crop size; map to absolute then normalize by full image size
        abs_x = xmin + (lm.x * bw)
        abs_y = ymin + (lm.y * bh)
        x_norm = abs_x / float(img_w) if img_w else 0.0
        y_norm = abs_y / float(img_h) if img_h else 0.0
        landmarks.append(_SimpleLandmark(x_norm, y_norm, lm.z))
    return _SimpleLandmarkList(landmarks)


def _clone_normalized_landmarks(face_lm_list):
    """Clone a MediaPipe landmark list into _SimpleLandmarkList (keeps normalized coords)."""
    landmarks = []
    for lm in face_lm_list.landmark:
        landmarks.append(_SimpleLandmark(lm.x, lm.y, lm.z))
    return _SimpleLandmarkList(landmarks)


def run_model_inference(image):
    """Run face detection, then FaceMesh per detected face using cached_faces, caching per-face landmarks."""
    global cached_face_landmarks, cached_face_landmarks_crops, cached_segmentation_mask, cached_image_shape, cached_face_hulls, cached_faces, cached_face_bboxes

    if image is None:
        return None, None

    # Reset caches for a fresh inference
    cached_faces.clear()
    cached_face_landmarks = []
    cached_face_landmarks_crops = []
    cached_face_hulls = []

    cached_image_shape = image.shape
    h, w = image.shape[:2]

    # Face detection (to populate cached_faces)
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = face_detection.process(image_rgb)

    face_bboxes = []
    if faces and faces.detections:
        for detection in faces.detections:
            xmin = detection.location_data.relative_bounding_box.xmin
            ymin = detection.location_data.relative_bounding_box.ymin
            height = detection.location_data.relative_bounding_box.height
            width = detection.location_data.relative_bounding_box.width

            xmin_pix = int(xmin * w)
            ymin_pix = int(ymin * h)
            width_pix = int(width * w)
            height_pix = int(height * h)

            # Enlarge bbox around center by FACE_CROP_SCALE and clamp
            cx = xmin_pix + width_pix / 2.0
            cy = ymin_pix + height_pix / 2.0
            new_w = int(round(width_pix * 1.2))
            new_h = int(round(height_pix * 2))
            new_xmin = int(round(cx - new_w / 2.0))
            new_ymin = int(round(cy - new_h / 2.0))

            x0 = max(0, min(new_xmin, w - 1))
            y0 = max(0, min(new_ymin, h - 1))
            x1 = max(x0 + 1, min(new_xmin + new_w, w))
            # The lower part of the face does not need such big crop, 1.2 factor for lower part
            y1 = max(y0 + 1, min(int(round(cy + 1.2*(height_pix/2.0))), h))
            final_w = max(1, x1 - x0)
            final_h = max(1, y1 - y0)
            print(new_w,new_h)
            face_crop = image[y0:y0 + final_h, x0:x0 + final_w]
            # cv2.imshow('face_crop', face_crop)
            # cv2.waitKey(0)
            cached_faces.append(face_crop)
            face_bboxes.append((x0, y0, final_w, final_h))

    # Face mesh inference per detected face; fallback to whole image if no faces found
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        '''
        What about not expand to full image landmark
        '''
        if cached_faces:
            for crop, bbox in zip(cached_faces, face_bboxes):
                if crop is None or crop.size == 0:
                    continue
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_results = face_mesh.process(crop_rgb)
                if crop_results.multi_face_landmarks:
                    # In typical cases, one face per crop; iterate defensively
                    for lm_list in crop_results.multi_face_landmarks:
                        mapped_full = _to_full_image_landmarks(lm_list, bbox, w, h)
                        crop_norm = _clone_normalized_landmarks(lm_list)
                        cached_face_landmarks.append(mapped_full)
                        cached_face_landmarks_crops.append(crop_norm)
        else:
            # Fallback: run mesh on the full image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(image_rgb)
            if face_results.multi_face_landmarks:
                for lm_list in face_results.multi_face_landmarks:
                    # Already normalized to full image dimensions
                    landmarks = [_SimpleLandmark(lm.x, lm.y, lm.z) for lm in lm_list.landmark]
                    lmlist = _SimpleLandmarkList(landmarks)
                    cached_face_landmarks.append(lmlist)
                    cached_face_landmarks_crops.append(lmlist)
                    cached_face_bboxes = [(0, 0, w, h)]

    # Save bboxes for ROI usage in warping
    cached_face_bboxes = face_bboxes

    # Pre-compute convex hulls for selection hit-testing and masks
    if cached_face_landmarks:
        cached_face_hulls = compute_face_hulls(cached_face_landmarks, w, h)
    else:
        cached_face_hulls = []

    # Skin segmentation mask for background blur
    # Use latest RGB computed above if available; otherwise compute from image
    try:
        rgb_for_seg = image_rgb  # from detection or fallback branch
    except NameError:
        rgb_for_seg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    all_masks = get_exposed_skin_mask(rgb_for_seg)
    cached_segmentation_mask = all_masks
    return cached_face_landmarks, cached_segmentation_mask


def process_image(image, selected_faces, apply_bg_blur=False):
    """Process image by applying warping per cropped face and compositing back."""
    global cached_face_landmarks, cached_face_landmarks_crops, cached_face_bboxes, face_params

    if image is None or not cached_face_landmarks or not selected_faces:
        return bg_blur(image) if apply_bg_blur else image

    h_img, w_img = image.shape[:2]

    # Shared adjustments on the whole image (whitening, etc.)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    eye_mask = np.zeros((h_img, w_img), np.uint8)
    for face_idx in selected_faces:
        if face_idx >= len(cached_face_landmarks):
            continue
        all_2d_full = get_points_2D(cached_face_landmarks[face_idx].landmark, w_img, h_img)
        eye_mask = cv2.bitwise_or(eye_mask, get_eye_mask(image, all_2d_full))
    processed_image = apply_whitening_and_blend(image, eye_mask, cached_segmentation_mask, hsv_image, 0,
                                                shared_params['S'], shared_params['V'], shared_params['MOPI'])

    for face_idx in selected_faces:
        if face_idx >= len(cached_face_bboxes) or face_idx >= len(cached_face_landmarks_crops):
            continue
        x, y, w_crop, h_crop = cached_face_bboxes[face_idx]
        if w_crop <= 1 or h_crop <= 1:
            continue

        # Dynamic upsample factor
        scale = get_dynamic_upsample_factor(w_crop, h_crop)
        w_big = w_crop * scale
        h_big = h_crop * scale

        # Extract and upsample ROI
        roi = processed_image[y:y + h_crop, x:x + w_crop].copy()
        if scale > 1:
            roi_big = cv2.resize(roi, (w_big, h_big), interpolation=cv2.INTER_NEAREST)
        else:
            roi_big = roi.copy()

        lm_crop = cached_face_landmarks_crops[face_idx]
        p = face_params[face_idx]

        # Scale landmarks to the big (upsampled) coordinate space
        def lm_to_big(landmark_list):
            return [(lm.x * w_big, lm.y * h_big) for lm in landmark_list.landmark]

        def lm3d_to_big(landmark_list):
            return [(lm.x * w_big * (lm.z + 1), lm.y * h_big * (lm.z + 1), lm.z + 1)
                    for lm in landmark_list.landmark]

        all_2d_big = lm_to_big(lm_crop)
        face_points3d_big = lm3d_to_big(lm_crop)

        # === First warp: SHOULIAN ===
        shoulian_src = {i: face_points3d_big[i] for i in SHOULIAN_INDICES if i < len(face_points3d_big)}
        shoulian_dst = shoulian_src.copy()
        local_dst = {k: v for k, v in shoulian_dst.items()}
        shoulian(p['SHOULIAN'], local_dst)
        for k in local_dst:
            shoulian_dst[k] = local_dst[k]
        roi_big = warp_with_triangulation(roi_big, shoulian_src, shoulian_dst)

        # === Second warp: main facial adjustments ===
        src_points = {i: face_points3d_big[i] for i in BOUNDARY if i < len(face_points3d_big)}
        dst_points = src_points.copy()
        local_dst = {k: v for k, v in dst_points.items()}
        quangu(p['QUANGU'] * p['ZHAILIAN'], local_dst)
        dayan(p['DAYAN'], local_dst)
        biyi(p['BIYI'], local_dst)
        longnose(p['LONGNOSE'], local_dst)
        zhailian(p['ZHAILIAN'], local_dst)
        renzhong(p['RENZHONG'], local_dst)
        forehead(p['FOREHEAD'], local_dst)
        zuijiao(p['ZUIJIAO'], local_dst)
        dazui(p['DAZUI'], local_dst)
        for k in local_dst:
            dst_points[k] = local_dst[k]

        roi_big = warp_with_triangulation(roi_big, src_points, dst_points)

        # === All 2D-based effects (use scaled 2D landmarks) ===
        if p['DETAIL'] > 0:
            roi_big = enhance_face_detail(roi_big, all_2d_big, strength_pct=p['DETAIL'])
        if p['FALING'] > 0:
            roi_big = nasolabial_folds_filter(roi_big, all_2d_big, p['FALING'])
        if p['BLACK'] > 0:
            roi_big = black_filter(roi_big, all_2d_big, p['BLACK'])
        if p['WHITE_EYE'] != 1:
            roi_big = white_eyes(roi_big, all_2d_big, p['WHITE_EYE'])
        if p['WHITE_TEETH'] != 1:
            roi_big = white_teeth(roi_big, all_2d_big, p['WHITE_TEETH'])
        if p['LIPSTICK'] > 0:
            roi_big = apply_lipstick(roi_big, all_2d_big, p['LIPSTICK'])
        if p['BLUSH'] > 0:
            roi_big = apply_blush(roi_big, all_2d_big, color=(157, 107, 255), intensity=p['BLUSH'])
        # roi_big = apply_big_eye_effect(roi_big, p['DAYAN'], all_2d_big)

        if scale > 1:
            roi_final = cv2.resize(roi_big, (w_crop, h_crop), interpolation=cv2.INTER_AREA)
        else:
            roi_final = roi_big

        processed_image[y:y + h_crop, x:x + w_crop] = roi_final

    if apply_bg_blur:
        processed_image = bg_blur(processed_image)

    return processed_image


def schedule_update():
    """Debounce updates to avoid excessive reprocessing"""
    global pending_update
    if pending_update:
        root.after_cancel(pending_update)
    pending_update = root.after(5, update_image)  # 50ms debounce


def set_param(param, val):
    """Set parameter for active face"""
    if param in ['S', 'V', 'MOPI']:
        shared_params[param] = val
    elif active_face is not None:
        face_params[active_face][param] = val


def make_slider_command(param):
    """Create slider callback with debouncing"""

    def cmd(val):
        set_param(param, float(val))
        schedule_update()

    return cmd


def update_active_face(idx):
    """Update UI to reflect active face"""
    global active_face
    active_face = idx
    active_label.config(text=f"Editing face {idx + 1}")
    for p in all_params:
        if p in ['S', 'V', 'MOPI']:
            val = shared_params[p]
        else:
            val = face_params[idx][p]
        param_to_var[p].set(val)


def update_image():
    """Update displayed image"""
    global image_path, bg_blur_enabled, pending_update
    pending_update = None

    image = cv2.imread(image_path)
    if image is None:
        error_label.config(text=f"Error: Could not load image")
        return

    h, w = image.shape[:2]
    if w >= 1920 or h >= 1080:
        image = cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

    processed_image = process_image(image, selected_faces, apply_bg_blur=bg_blur_enabled)
    if processed_image is None:
        error_label.config(text="Error: Image processing failed.")
        return

    # Convert and display
    display_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    display_image_pil = Image.fromarray(display_image_rgb)
    display_image_tk = ImageTk.PhotoImage(display_image_pil)

    image_label.config(image=display_image_tk)
    image_label.image = display_image_tk
    error_label.config(text="")
    blur_button.config(text="取消模糊背景" if bg_blur_enabled else "模糊背景")


def toggle_bg_blur():
    """Toggle background blur"""
    global bg_blur_enabled
    bg_blur_enabled = not bg_blur_enabled
    update_image()


def open_image():
    """Open new image and run inference"""
    global image_path, selected_faces, face_params, active_face

    new_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not new_image_path:
        return

    image_path = new_image_path
    selected_faces = set()
    face_params = {}
    active_face = None
    active_label.config(text="Editing: None")

    image = cv2.imread(image_path)
    if image is None:
        error_label.config(text=f"Error: Could not load image")
        return

    h, w = image.shape[:2]
    if w >= 1920 or h >= 1080:
        image = cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

    run_model_inference(image)

    for i in range(len(cached_face_landmarks)):
        selected_faces.add(i)
        face_params[i] = {k: v for k, v in defaults.items() if k not in ['S', 'V', 'MOPI']}
        update_active_face(i)

    update_image()


def save_image():
    """Save processed image"""
    global image_path, bg_blur_enabled

    image = cv2.imread(image_path)
    if image is None:
        error_label.config(text=f"Error: Could not load image")
        return

    h, w = image.shape[:2]
    if w >= 1920 or h >= 1080:
        image = cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

    processed_image = process_image(image, selected_faces, apply_bg_blur=bg_blur_enabled)
    if processed_image is None:
        error_label.config(text="Error: Image processing failed.")
        return

    base_name = image_path.split('/')[-1].split('.')[0]
    blur_suffix = "_blur" if bg_blur_enabled else ""
    filename = f"{base_name}{blur_suffix}.jpg"

    cv2.imwrite(filename, processed_image)
    error_label.config(text=f"Image saved as {filename}")


# UI Setup
root = tk.Tk()
root.title("美颜算法")

image_path = 'imgs/face4.jpg'

# Image frame
image_frame = tk.Frame(root)
image_frame.pack(side=tk.LEFT, padx=10, pady=10)
image_label = tk.Label(image_frame)
image_label.pack()
image_label.bind("<Button-1>", on_click)
error_label = tk.Label(image_frame, text="", fg="red")
error_label.pack()

# Control frame
control_frame = tk.Frame(root)
control_frame.pack(side=tk.RIGHT, padx=10, pady=10)
for i in range(3):
    control_frame.grid_columnconfigure(i, weight=1)

# Create all slider variables
param_to_var = {param: tk.DoubleVar(value=defaults[param]) for param in all_params}

slider_config = [
    ("大眼", 'DAYAN', 1.0, 1.1, 0.001, 0, 0),
    ("瘦脸", 'SHOULIAN', 0.95, 1.0, 0.005, 0, 1),
    ("颧骨", 'QUANGU', 0.95, 1.0, 0.005, 0, 2),
    ("人中", 'RENZHONG', -3.0, 3.0, 0.005, 1, 2),
    ("额头", 'FOREHEAD', -5.0, 5.0, 0.005, 2, 2),
    ("窄脸", 'ZHAILIAN', 0.95, 1.0, 0.005, 1, 0),
    ("鼻翼", 'BIYI', 0.9, 1.1, 0.005, 1, 1),
    ("长鼻", 'LONGNOSE', -2.0, 2.0, 0.005, 2, 1),
    ("嘴角", 'ZUIJIAO', 0.0, 3.0, 0.005, 2, 0),
    ("大嘴", 'DAZUI', 0.9, 1.1, 0.005, 3, 2),
    ("美白", 'S', 0.0, 0.08, 0.002, 3, 0),
    ("磨皮", 'MOPI', 0, 0.5, 0.01, 5, 0),
    ("细节增强", 'DETAIL', 0, 50, 1, 5, 1),
    ("去法令纹", 'FALING', 0, 1, 0.02, 6, 0),
    ("去黑眼圈", 'BLACK', 0, 1, 0.02, 6, 1),
    ("亮眼", 'WHITE_EYE', 1, 1.2, 0.02, 7, 0),
    ("亮牙", 'WHITE_TEETH', 1, 1.2, 0.01, 7, 1),
    ("口红", 'LIPSTICK', 0.0, 0.25, 0.01, 8, 0),
    ("腮红", 'BLUSH', 0.0, 1, 0.05, 8, 1),
]

for label, param, from_, to_, res, row, col in slider_config:
    tk.Scale(control_frame, from_=from_, to=to_, resolution=res, orient=tk.HORIZONTAL,
             label=label, variable=param_to_var[param],
             command=make_slider_command(param)).grid(row=row, column=col, padx=5, pady=5, sticky="ew")

# Status and buttons
active_label = tk.Label(control_frame, text="Editing: None", font=("Arial", 12, "bold"))
active_label.grid(row=9, column=0, columnspan=3, pady=5)

open_button = tk.Button(control_frame, text="打开图片", command=open_image)
open_button.grid(row=10, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

blur_button = tk.Button(control_frame, text="模糊背景", command=toggle_bg_blur,
                        bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
blur_button.grid(row=11, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

save_button = tk.Button(control_frame, text="保存图片", command=save_image)
save_button.grid(row=12, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

# Initialize
initial_image = cv2.imread(image_path)
if initial_image is not None:
    h, w = initial_image.shape[:2]
    if w >= 1920 or h >= 1080:
        initial_image = cv2.resize(initial_image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    run_model_inference(initial_image)
    for i in range(len(cached_face_landmarks)):
        selected_faces.add(i)
        face_params[i] = {k: v for k, v in defaults.items() if k not in ['S', 'V', 'MOPI']}
        update_active_face(i)

update_image()
root.mainloop()
