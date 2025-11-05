import cv2
import mediapipe as mp
import numpy as np

RIGHT_EYE_RING = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_RING = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

def white_eyes(image, landmarks, strength):


    h, w, _ = image.shape

    right_pts = np.array([landmarks[i] for i in RIGHT_EYE_RING], np.int32)
    left_pts = np.array([landmarks[i] for i in LEFT_EYE_RING], np.int32)
    right_pts = cv2.convexHull(right_pts)
    left_pts = cv2.convexHull(left_pts)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.array(left_pts), 255)
    cv2.fillConvexPoly(mask, np.array(right_pts), 255)
    # mask = mask / 255.0
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    mask_normalized = mask.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    blurred = blurred * 0.04
    # print('blurred',blurred)
    # print('image',image)
    # cv2.imshow('blurred', blurred)
    result = blurred + (image - blurred) * strength
    # print('result',result)
    # result = np.maximum(blurred, result)
    # print(result == image)
    result = np.clip(result, 0, 255).astype(np.uint8)
    # cv2.imshow('image', image)
    # cv2.imshow('result', result)

    blended_image = (1 - mask_normalized[..., None]) * image + mask_normalized[..., None] * result
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
    # cv2.imshow('blended', blended_image)
    # cv2.waitKey(0)
    return blended_image


# image = cv2.imread('img.png')
# result = white_eyes(image, 5)
# cv2.imshow('image', image)
# cv2.imshow('result', result)
# cv2.waitKey(0)