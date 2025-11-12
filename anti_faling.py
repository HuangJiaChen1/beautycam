import cv2
import mediapipe as mp
import numpy as np
nose_indices = [193, 168, 417, 122, 351, 196, 419, 3, 248, 236, 456, 198, 420, 131, 360, 49, 279, 48, 278, 219, 439,
                59, 289, 218, 438, 237, 457, 44, 19, 274]
mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185,
                 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

def create_region_mask(landmarks, indices, h, w):
    points = []
    for idx in indices:
        x,y = landmarks[idx]
        points.append([x, y])

    if len(points) < 3:
        return np.zeros((h, w), dtype=np.float32)

    points = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(points)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask.astype(np.float32) / 255.0

def nasolabial_folds_filter(image, landmarks, strength):
    if strength <= 0:
        return image

    h, w, _ = image.shape

    left_indices = [142,36,205,207,216,206,98]
    right_indices = [371, 266,425,427,436,426,327]

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

    mask_roi = mask_u8.astype(np.float32) / 255.0
    mask_roi = cv2.dilate(mask_roi, (5,5),iterations=1)
    mask_roi = cv2.GaussianBlur(mask_roi, (21, 21), 0)
    nose_mask = create_region_mask(landmarks, nose_indices, h, w)
    mouth_mask = create_region_mask(landmarks, mouth_indices, h, w)
    mask_roi = 1-np.maximum(1 - mask_roi, np.maximum(nose_mask, mouth_mask))[y1:y2, x1:x2].astype(np.float32)
    # cv2.imshow("mask_roi", mask_roi)
    # cv2.waitKey(0)
    alpha = mask_roi[..., np.newaxis]

    crop = image[y1:y2, x1:x2]
    d = max(5, int(32 * strength))
    crop_blur = cv2.bilateralFilter(crop, d=d, sigmaColor=256 * strength, sigmaSpace=32 * strength)
    # crop_blur = cv2.bilateralFilter(crop_blur, d=d, sigmaColor=256 * strength, sigmaSpace=32 * strength)
    # crop_blur = retouch(crop,strength)
    # cv2.imshow('crop', crop_blur)
    # cv2.waitKey(0)
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
