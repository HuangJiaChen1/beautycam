TEETH_INDICES = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95] #SORTED
LIPS_INDICES = [61,185,40, 39, 37, 0, 267, 269,270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146] # SORTED
import cv2
import mediapipe as mp
import numpy as np
'''
现在问题是会出嘴唇范围，看能不能用像素均值再segment一下
好像好了
'''
def segment_lips(hsv_image, mask, within_sd = 1.8):
    h_channel = hsv_image[:, :, 0]
    s_channel = hsv_image[:, :, 1]
    v_channel = hsv_image[:, :, 2]

    h_mean, h_stddev = cv2.meanStdDev(h_channel, mask=mask)
    s_mean, s_stddev = cv2.meanStdDev(s_channel, mask=mask)
    v_mean, v_stddev = cv2.meanStdDev(v_channel, mask=mask)

    h_mean = h_mean[0][0]
    h_stddev = h_stddev[0][0]
    s_mean = s_mean[0][0]
    s_stddev = s_stddev[0][0]
    v_mean = v_mean[0][0]
    v_stddev = v_stddev[0][0]

    h_range = (int(h_mean - h_stddev * within_sd), int(h_mean + h_stddev * within_sd))
    s_range = (int(s_mean - s_stddev * within_sd), int(s_mean + s_stddev * within_sd))
    v_range = (int(v_mean - v_stddev * within_sd), int(v_mean + v_stddev * within_sd))

    lower_hsv = np.array([h_range[0] if h_range[0] >= 0 else 0,
                          s_range[0] if s_range[0] >= 0 else 0,
                          v_range[0] if v_range[0] >= 0 else 0])
    upper_hsv = np.array([h_range[1] if h_range[1] <= 255 else 255,
                          s_range[1] if s_range[1] <= 255 else 255,
                          v_range[1] if v_range[1] <= 255 else 255])

    return lower_hsv, upper_hsv


def lipstick(image,landmarks, strength=1.0, color=(255, 20, 20)):
    if strength <= 0:
        return image.copy()


    h, w, _ = image.shape

    lips_pts = np.array([landmarks[i] for i in LIPS_INDICES], np.int32)
    teeth_pts = np.array([landmarks[i] for i in TEETH_INDICES], np.int32)

    lip_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(lip_mask, [lips_pts], 255)

    teeth_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(teeth_mask, [teeth_pts], 255)

    mask = cv2.bitwise_and(lip_mask, cv2.bitwise_not(teeth_mask))
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    mask_normalized = mask.astype(np.float32) / 255.0

    im_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0

    bgr_mat = np.array([[[color[2], color[1], color[0]]]], dtype=np.uint8)
    ycrcb_mat = cv2.cvtColor(bgr_mat, cv2.COLOR_BGR2YCrCb)
    lipstick_ycrcb = (ycrcb_mat[0, 0].astype(np.float32) / 255.0)[::-1]
    indices = np.where(mask_normalized > 0)
    if len(indices[0]) == 0:
        return image.copy()

    weights = mask_normalized[indices]
    masked_pixels = im_ycrcb[indices]
    m = np.average(masked_pixels, axis=0, weights=weights)

    mpxl = np.clip(mask_normalized[indices] * strength, 0, 1)
    src_pxl = im_ycrcb[indices]

    for idx in range(3):
        im_ycrcb[indices[0], indices[1], idx] = mpxl * (lipstick_ycrcb[idx] + (src_pxl[:, idx] - m[idx])) + (1 - mpxl) * \
                                                src_pxl[:, idx]

    im_out = cv2.cvtColor(im_ycrcb, cv2.COLOR_YCrCb2BGR) * 255.0
    result = np.clip(im_out, 0, 255).astype(np.uint8)

    return result

#
# image = cv2.imread('imgs/smile.jpg')
# result = lipstick(image, strength=0.3, color=(255, 20, 20))
# cv2.imshow('Original', image)
# cv2.imshow('Lipstick', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
