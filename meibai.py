import cv2
import numpy as np

from test import smooth_and_sharpen_skin


def segment_face(hsv_image, face_landmarks, within_sd=2):
    landmarks = np.array(face_landmarks, dtype=np.int32)
    hull = cv2.convexHull(landmarks)

    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    h_channel = hsv_image[:, :, 0]  # Hue
    s_channel = hsv_image[:, :, 1]  # Saturation
    v_channel = hsv_image[:, :, 2]  # Value

    # bright_mask = (v_channel > 100) & (s_channel < 80)
    #
    # final_mask = cv2.bitwise_and(mask, bright_mask.astype(np.uint8))

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


def meibai(hsv_image, mask,H,S,V, factor_max=3.0, decrease_s_factor_min=0):

    h_mask = mask / 255.0 * H
    v_mask = mask / 255.0 * V
    s_mask = mask / 255.0 * S

    h_factors = 1 + factor_max * h_mask
    v_factors = 1 + factor_max * v_mask
    s_factors = 1 + factor_max * s_mask

    h_channel = hsv_image[..., 0]
    s_channel = hsv_image[..., 1]
    v_channel = hsv_image[..., 2]

    h_channel = np.clip(h_channel * h_factors, 0, 255)
    v_channel = np.clip(v_channel * v_factors, 0, 255)
    s_channel = np.clip(s_channel / s_factors, 0, 255)

    hsv_image[..., 0] = h_channel
    hsv_image[..., 1] = s_channel
    hsv_image[..., 2] = v_channel

    modified_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return modified_image

def apply_bilateral_filter(image,
                           face_mask_uint8,
                           MOPI,
                           d=32,
                           sigma_color_base=128,
                           sigma_space_base=16):
    sigma_color = int(sigma_color_base * MOPI)
    sigma_space = int(sigma_space_base * MOPI)

    filtered_full = cv2.bilateralFilter(
        src=image,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
        borderType=cv2.BORDER_DEFAULT
    )

    mask_f = face_mask_uint8.astype(np.float32) / 255.0
    mask_f = mask_f[..., np.newaxis]

    result = (image.astype(np.float32) * (1.0 - mask_f) +
              filtered_full.astype(np.float32) * mask_f)
    return np.clip(result, 0, 255).astype(np.uint8)



def create_robust_skin_mask(hsv_image, lower_hsv, upper_hsv,
                            close_kernel=(25, 25), fill_holes=True):
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if fill_holes:
        num_labels, labels = cv2.connectedComponents(mask)
        if num_labels > 1:
            sizes = [(np.sum(labels == i), i) for i in range(1, num_labels)]
            if sizes:
                largest_label = max(sizes, key=lambda x: x[0])[1]
                mask = np.uint8(labels == largest_label) * 255

    mask_f = mask.astype(np.float32) / 255.0
    feathered = cv2.GaussianBlur(mask_f, (51, 51), 15)
    feathered = np.clip(feathered, 0, 1)
    # cv2.imshow("mask", mask)
    # cv2.imshow("feathered", feathered)
    # cv2.waitKey(0)
    return feathered ,mask
def blend_whitened(original_bgr, whitened_bgr, soft_mask):
    alpha = soft_mask[..., np.newaxis]          # (H, W, 1)

    orig_f = original_bgr.astype(np.float32)
    whit_f = whitened_bgr.astype(np.float32)

    blended = orig_f * (1 - alpha) + whit_f * alpha
    return blended.astype(np.uint8)
def apply_whitening_and_blend(original_image,
                              hsv_image,
                              lower_hsv, upper_hsv,
                              H,
                              S,V, MOPI,
                              blur_sigma=12):

    hard_mask, soft_mask = create_robust_skin_mask(
        hsv_image, lower_hsv, upper_hsv,
        close_kernel=(31,31), fill_holes=True
    )

    whitened = meibai(hsv_image.copy(), hard_mask*255.0, H,S,V)

    whitened = apply_bilateral_filter(
        whitened, hard_mask*255.0, MOPI,
        d=32, sigma_color_base=128, sigma_space_base=16)
    # cv2.imshow('whitened', whitened)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    final_image = blend_whitened(original_image, whitened, hard_mask)
    # cv2.imshow('final_image', final_image)
    # cv2.waitKey(0)
    # skin_processed = cv2.bitwise_and(whitened, whitened, mask=hard_mask)
    # holes = hard_mask & (~cv2.inRange(cv2.cvtColor(skin_processed, cv2.COLOR_BGR2GRAY), 1, 255))
    # if np.any(holes):
    #     whitened = cv2.inpaint(whitened, holes, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    #
    # alpha = soft_mask[..., np.newaxis]
    # blended = (original_image.astype(np.float32) * (1.0 - alpha) +
    #            whitened.astype(np.float32) * alpha)
    return np.clip(final_image, 0, 255).astype(np.uint8)