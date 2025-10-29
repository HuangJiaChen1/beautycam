import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, List

# ----------------------------------------------------------------------
# MediaPipe Face Mesh setup
# ----------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
FACE_CONNECTIONS = mp_face_mesh.FACEMESH_CONTOURS  # only outer contour + eyes + lips

# ----------------------------------------------------------------------
# Helper: create binary masks from landmark indices
# ----------------------------------------------------------------------
def indices_to_mask(img_h: int, img_w: int, indices: List[int],
                    landmarks) -> np.ndarray:
    """Return a single-channel mask (0/1) for the given landmark indices."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    points = np.array([(landmarks[i].x * img_w, landmarks[i].y * img_h)
                       for i in indices], dtype=np.int32)
    cv2.fillConvexPoly(mask, points, 1)
    return mask.astype(np.float32)

# ----------------------------------------------------------------------
# Region definitions (MediaPipe 468-landmark indices)
# ----------------------------------------------------------------------
# Full-face contour (jaw + forehead)


CONTOUR_IDX = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10]

# Eyes (left + right)
LEFT_EYE_IDX  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
                 159, 160, 161, 246]          # outer + iris
RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388,
                 387, 386, 385, 384, 398]

# Lips (outer + inner)
LIP_OUTER_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375,
                 321, 308, 415, 310, 311, 312, 13, 82, 81, 80, 78, 191,
                 178, 179, 180, 183, 184, 185, 40, 39, 37, 0, 267, 269,
                 270, 409, 291, 375, 321, 314, 17, 84, 181, 91, 146, 61]
def enhance_face_detail(
    img_bgr: np.ndarray,
    strength_pct: float = 50.0,
) -> np.ndarray:

    if not (0 <= strength_pct <= 100):
        raise ValueError("strength_pct must be 0-100")

    h, w = img_bgr.shape[:2]
    img = img_bgr.copy().astype(np.float32) / 255.0
    s = strength_pct / 100.0

    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return img_bgr  # no face → return original

        landmarks = results.multi_face_landmarks[0].landmark

    def indices_to_mask(indices):
        points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, points, 1)
        # Soft edges
        kernel = cv2.getGaussianKernel(15, 5)
        kernel2D = kernel @ kernel.T
        return cv2.filter2D(mask.astype(np.float32), -1, kernel2D)

    mask_contour = indices_to_mask(CONTOUR_IDX)
    mask_left    = indices_to_mask(LEFT_EYE_IDX)
    mask_right   = indices_to_mask(RIGHT_EYE_IDX)
    mask_lips    = indices_to_mask(LIP_OUTER_IDX)

    face_mask = np.clip(mask_contour + mask_left + mask_right + mask_lips, 0, 1)


    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L_float = L.astype(np.float32)

    blur_L = cv2.GaussianBlur(L_float, (0, 0), sigmaX=1.5)
    L_sharp = cv2.addWeighted(L_float, 1 + s, blur_L, -s, 0)

    L_final = L_float * (1 - face_mask) + L_sharp * face_mask
    L_final = np.clip(L_final, 0, 255).astype(np.uint8)

    lab_final = cv2.merge([L_final, A, B])
    result = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0

    return (result * 255).astype(np.uint8)


# import cv2
# img = cv2.imread("mopi.jpg")
# enhanced = enhance_face_detail(img, strength_pct=75)   # heavy mode
# cv2.imshow("enhanced", enhanced)
# cv2.waitKey(0)