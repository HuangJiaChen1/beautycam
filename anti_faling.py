import cv2
import mediapipe as mp
import numpy as np


def nasolabial_folds_filter(image, landmarks, strength):

    if strength <= 0:
        return image


    h, w, _ = image.shape

    left_indices = [36, 165, 92, 186,57,207]
    right_indices = [391,432,411,266]

    left_coords = [landmarks[i] for i in left_indices]
    right_coords = [landmarks[i] for i in right_indices]

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(left_coords)], 255)
    cv2.fillPoly(mask, [np.array(right_coords)], 255)

    # mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    mask = mask / 255.0
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)

    blurred = cv2.bilateralFilter(image, d=int(32*strength), sigmaColor=128*strength, sigmaSpace=16*strength)
    alpha = mask[..., np.newaxis]          # (H, W, 1)

    original = image.astype(np.float32)
    blurred = blurred.astype(np.float32)

    blended = original * (1 - alpha) + blurred * alpha
    return blended.astype(np.uint8)
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