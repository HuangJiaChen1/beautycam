import cv2
import mediapipe as mp
import numpy as np

TEETH_INDICES = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]

def white_teeth(image, landmarks, strength):
    h, w, _ = image.shape
    # print(strength)
    teeth_pts = np.array([landmarks[i] for i in TEETH_INDICES], np.int32)
    mask_u8 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_u8, [teeth_pts], 255)
    mask_u8 = cv2.erode(mask_u8, np.ones((3, 3), np.uint8), iterations=2)
    mask_u8 = cv2.GaussianBlur(mask_u8, (5, 5), 0)

    # Restrict processing to ROI
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    x, y, rw, rh = cv2.boundingRect(np.vstack(contours))
    border = 6
    x1 = max(x - border, 0)
    y1 = max(y - border, 0)
    x2 = min(x + rw + border, w)
    y2 = min(y + rh + border, h)

    roi = image[y1:y2, x1:x2]
    mask_roi = (mask_u8[y1:y2, x1:x2].astype(np.float32) / 255.0)[..., None]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = cv2.split(hsv)

    increase_v_factor = strength
    decrease_s_factor = 1 / (strength ** 3)

    V = np.minimum(255.0, V * increase_v_factor)
    S = np.maximum(0.0, S * decrease_s_factor)

    hsv_adj = cv2.merge([H, S, V]).astype(np.uint8)
    result_roi = cv2.cvtColor(hsv_adj, cv2.COLOR_HSV2BGR).astype(np.float32)

    blended = roi.astype(np.float32) * (1 - mask_roi) + result_roi * mask_roi
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    out = image.copy()
    out[y1:y2, x1:x2] = blended
    return out
