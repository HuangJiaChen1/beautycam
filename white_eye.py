import cv2
import mediapipe as mp
import numpy as np

RIGHT_EYE_RING = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_RING = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

def white_eyes(image, landmarks, strength):
    h, w, _ = image.shape

    right_pts = cv2.convexHull(np.array([landmarks[i] for i in RIGHT_EYE_RING], np.int32))
    left_pts = cv2.convexHull(np.array([landmarks[i] for i in LEFT_EYE_RING], np.int32))

    mask_u8 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask_u8, left_pts, 255)
    cv2.fillConvexPoly(mask_u8, right_pts, 255)
    mask_u8 = cv2.erode(mask_u8, np.ones((3, 3), np.uint8), iterations=1)
    mask_u8 = cv2.GaussianBlur(mask_u8, (5, 5), 0)

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

    blurred_roi = cv2.GaussianBlur(roi, (5, 5), 0).astype(np.float32) * 0.04
    result_roi = blurred_roi + (roi.astype(np.float32) - blurred_roi) * strength
    result_roi = np.clip(result_roi, 0, 255).astype(np.uint8)

    blended_roi = (1 - mask_roi) * roi + mask_roi * result_roi
    out = image.copy()
    out[y1:y2, x1:x2] = np.clip(blended_roi, 0, 255).astype(np.uint8)
    return out


# image = cv2.imread('img.png')
# result = white_eyes(image, 5)
# cv2.imshow('image', image)
# cv2.imshow('result', result)
# cv2.waitKey(0)
