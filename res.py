import cv2
import numpy as np
from typing import List



CONTOUR_IDX = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10]

LEFT_EYE_IDX  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
                 159, 160, 161, 246]          # outer + iris
RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388,
                 387, 386, 385, 384, 398]

LIP_OUTER_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375,
                 321, 308, 415, 310, 311, 312, 13, 82, 81, 80, 78, 191,
                 178, 179, 180, 183, 184, 185, 40, 39, 37, 0, 267, 269,
                 270, 409, 291, 375, 321, 314, 17, 84, 181, 91, 146, 61]
def _mask_from_indices(h: int, w: int, indices: List[int], landmarks_2d: List[tuple]) -> np.ndarray:
    pts = np.array([landmarks_2d[i] for i in indices], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(pts) >= 3:
        cv2.fillConvexPoly(mask, pts, 1)
    return mask.astype(np.float32)


def enhance_face_detail(
    img_bgr: np.ndarray,
    landmarks_2d: List[tuple],
    strength_pct: float = 50.0,
) -> np.ndarray:
    if not (0 <= strength_pct <= 100):
        raise ValueError("strength_pct must be 0-100")

    h, w = img_bgr.shape[:2]
    s = strength_pct / 100.0

    # Build soft face mask using provided landmarks (no extra FaceMesh call)
    mask_contour = _mask_from_indices(h, w, CONTOUR_IDX, landmarks_2d)
    mask_left    = _mask_from_indices(h, w, LEFT_EYE_IDX, landmarks_2d)
    mask_right   = _mask_from_indices(h, w, RIGHT_EYE_IDX, landmarks_2d)
    mask_lips    = _mask_from_indices(h, w, LIP_OUTER_IDX, landmarks_2d)

    face_mask = np.clip(mask_contour + mask_left + mask_right + mask_lips, 0, 1)
    # Soften edges once with Gaussian blur
    face_mask = cv2.GaussianBlur(face_mask, (15, 15), 5)

    # Unsharp masking on L channel for natural detail enhancement
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L_float = L.astype(np.float32)

    blur_L = cv2.GaussianBlur(L_float, (0, 0), sigmaX=1.5)
    L_sharp = cv2.addWeighted(L_float, 1 + s, blur_L, -s, 0)

    L_final = L_float * (1 - face_mask) + L_sharp * face_mask
    L_final = np.clip(L_final, 0, 255).astype(np.uint8)

    lab_final = cv2.merge([L_final, A, B])
    result = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)
    return result


# import cv2
# img = cv2.imread("mopi.jpg")
# enhanced = enhance_face_detail(img, strength_pct=75)   # heavy mode
# cv2.imshow("enhanced", enhanced)
# cv2.waitKey(0)
