import cv2
import numpy as np


def black_filter(image, landmarks, strength=1.0):
    if strength <= 0:
        return image.copy()

    h, w, _ = image.shape

    left_idx = [341, 256, 252, 253, 261, 350, 330, 346]
    right_idx = [112, 26, 22, 23, 31, 121, 101, 117]

    left_pts = np.array([landmarks[i] for i in left_idx], np.int32)
    right_pts = np.array([landmarks[i] for i in right_idx], np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [left_pts, right_pts], 255)

    blend_mask = mask.astype(np.float32) / 255.0
    blend_mask = cv2.GaussianBlur(blend_mask, (31, 31), 0)  # same as original
    alpha = blend_mask[..., np.newaxis]  # (H,W,1)

    border = int(15 + 16 * strength)  # safe margin
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border + 1, 2 * border + 1))
    dilated = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        face_mesh.close()
        return image.copy()
    x, y, rw, rh = cv2.boundingRect(np.vstack(contours))

    x1 = max(x - border, 0)
    y1 = max(y - border, 0)
    x2 = min(x + rw + border, w)
    y2 = min(y + rh + border, h)

    crop_orig = image[y1:y2, x1:x2]
    crop_h, crop_w = crop_orig.shape[:2]
    d = int(32 * strength)
    sigma_color = 128 * strength
    sigma_space = 16 * strength

    crop_blurred = cv2.bilateralFilter(
        crop_orig, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space
    )

    crop_alpha = alpha[y1:y2, x1:x2]  # (crop_h, crop_w, 1)

    orig_f = crop_orig.astype(np.float32)
    blur_f = crop_blurred.astype(np.float32)

    crop_blended = orig_f * (1 - crop_alpha) + blur_f * crop_alpha
    crop_blended = crop_blended.astype(np.uint8)

    result = image.copy()
    result[y1:y2, x1:x2] = crop_blended

    return result