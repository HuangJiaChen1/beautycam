import cv2
import mediapipe as mp
import numpy as np


def nasolabial_folds_filter(image, landmarks, strength):
    if strength <= 0:
        return image

    h, w, _ = image.shape

    left_indices = [36, 165, 92, 186, 57, 207]
    right_indices = [391, 432, 411, 266]

    left_coords = np.array([landmarks[i] for i in left_indices], np.int32)
    right_coords = np.array([landmarks[i] for i in right_indices], np.int32)

    mask_u8 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_u8, [left_coords], 255)
    cv2.fillPoly(mask_u8, [right_coords], 255)

    # Determine a tight ROI around the mask for faster filtering
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    x, y, rw, rh = cv2.boundingRect(np.vstack(contours))
    border = max(8, int(12 * strength))
    x1 = max(x - border, 0)
    y1 = max(y - border, 0)
    x2 = min(x + rw + border, w)
    y2 = min(y + rh + border, h)

    mask_roi = mask_u8[y1:y2, x1:x2].astype(np.float32) / 255.0
    mask_roi = cv2.GaussianBlur(mask_roi, (31, 31), 0)
    alpha = mask_roi[..., np.newaxis]

    crop = image[y1:y2, x1:x2]
    d = max(5, int(32 * strength))
    crop_blur = cv2.bilateralFilter(crop, d=d, sigmaColor=128 * strength, sigmaSpace=16 * strength)

    orig_f = crop.astype(np.float32)
    blur_f = crop_blur.astype(np.float32)
    blended = orig_f * (1 - alpha) + blur_f * alpha

    result = image.copy()
    result[y1:y2, x1:x2] = blended.astype(np.uint8)
    return result
    # result = image.copy()
    # result[mask > 0] = blended[mask > 0]
    #
    # face_mesh.close()
    # return result


# image = cv2.imread('imgs/faling.jpg')
# result = nasolabial_folds_filter(image, 1)
# cv2.imshow('image', image)
# cv2.imshow('result', result)
# cv2.waitKey(0)
