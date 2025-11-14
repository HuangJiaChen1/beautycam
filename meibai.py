import cv2
import numpy as np

RIGHT_EYE_RING = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_RING = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
LIPS_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]


def linear_light_blend_numba(base, blend, result):
    h, w, c = base.shape
    for i in range(h):
        for j in range(w):
            for k in range(c):
                base_norm = base[i, j, k] / 255.0
                blend_norm = blend[i, j, k] / 255.0

                if blend_norm > 0.5:
                    res = base_norm + 2 * (blend_norm - 0.5)
                else:
                    res = base_norm + 2 * blend_norm - 1

                result[i, j, k] = min(max(res * 255.0, 0.0), 255.0)
    return result


def linear_light_blend(base, blend):
    base_norm = base.astype(np.float32) / 255.0
    blend_norm = blend.astype(np.float32) / 255.0

    result_norm = np.where(
        blend_norm > 0.5,
        base_norm + 2 * (blend_norm - 0.5),
        base_norm + 2 * blend_norm - 1
    )

    result_norm = np.clip(result_norm, 0.0, 1.0)
    return (result_norm * 255.0).astype(np.uint8)


def retouch(image, strength):
    lpf = cv2.GaussianBlur(image, (3, 3), 0)
    hpf = image.astype(np.int16) - lpf.astype(np.int16)

    d = 16 if strength < 0.5 else 24
    lpf = cv2.bilateralFilter(lpf, d, int(128 * strength), int(16 * strength))

    hpf_remapped = hpf * 0.5 + 128
    hpf_remapped = np.clip(hpf_remapped, 0, 255).astype(np.uint8)

    result = linear_light_blend(lpf, hpf_remapped)
    return result


def segment_face(hsv_image, face_landmarks, within_sd=2):
    h, w, _ = hsv_image.shape

    eye_mask = np.zeros((h, w), dtype=np.uint8)
    lip_mask = np.zeros((h, w), dtype=np.uint8)
    right_pts = cv2.convexHull(np.array([face_landmarks[i] for i in RIGHT_EYE_RING], np.int32))
    left_pts = cv2.convexHull(np.array([face_landmarks[i] for i in LEFT_EYE_RING], np.int32))
    lips_pts = np.array([face_landmarks[i] for i in LIPS_INDICES], np.int32)

    cv2.fillConvexPoly(eye_mask, left_pts, 255)
    cv2.fillConvexPoly(eye_mask, right_pts, 255)
    cv2.fillConvexPoly(lip_mask, lips_pts, 255)
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    #
    # lip_mask = cv2.erode(lip_mask,kernel,iterations=2)
    eye_mask = cv2.bitwise_or(eye_mask, lip_mask)
    # cv2.imshow("eye_mask", eye_mask)
    # cv2.waitKey(0)

    hull = cv2.convexHull(np.array(face_landmarks, dtype=np.int32))
    face_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, hull, 255)

    mask = cv2.bitwise_and(face_mask, cv2.bitwise_not(eye_mask))

    h_channel, s_channel, v_channel = cv2.split(hsv_image)

    h_mean, h_stddev = cv2.meanStdDev(h_channel, mask=mask)
    s_mean, s_stddev = cv2.meanStdDev(s_channel, mask=mask)
    v_mean, v_stddev = cv2.meanStdDev(v_channel, mask=mask)

    h_mean, h_stddev = h_mean[0][0], h_stddev[0][0]
    s_mean, s_stddev = s_mean[0][0], s_stddev[0][0]
    v_mean, v_stddev = v_mean[0][0], v_stddev[0][0]

    lower_hsv = np.array([
        max(0, int(h_mean - h_stddev * within_sd)),
        max(0, int(s_mean - s_stddev * within_sd)),
        max(0, int(v_mean - v_stddev * within_sd))
    ], dtype=np.uint8)

    upper_hsv = np.array([
        min(255, int(h_mean + h_stddev * within_sd)),
        min(255, int(s_mean + s_stddev * within_sd)),
        min(255, int(v_mean + v_stddev * within_sd))
    ], dtype=np.uint8)

    return lower_hsv, upper_hsv, eye_mask


def meibai(hsv_image, mask, H, S, V, factor_max=3.0):
    mask_norm = mask / 255.0

    h_mask = mask_norm * H
    v_mask = mask_norm * (S/0.8*0.2)
    s_mask = mask_norm * S

    h_factors = 1 + factor_max * h_mask
    v_factors = 1 + factor_max * v_mask
    s_factors = 1 + factor_max * s_mask

    # In-place channel modification
    hsv_image[..., 0] = np.clip(hsv_image[..., 0] * h_factors, 0, 255)
    hsv_image[..., 2] = np.clip(hsv_image[..., 2] * v_factors, 0, 255)
    hsv_image[..., 1] = np.clip(hsv_image[..., 1] / s_factors, 0, 255)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def apply_bilateral_filter(image, face_mask_uint8, MOPI):
    filtered_full = retouch(image, MOPI)

    mask_f = (face_mask_uint8.astype(np.float32) / 255.0)[..., np.newaxis]
    result = image.astype(np.float32) * (1.0 - mask_f) + filtered_full.astype(np.float32) * mask_f

    return np.clip(result, 0, 255).astype(np.uint8)


def create_robust_skin_mask(hsv_image, lower_hsv, upper_hsv, eye_mask, close_kernel=(15, 15)):
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(eye_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # mask = cv2.dilate(mask, kernel, iterations=1)
    # mask = cv2.erode(mask, kernel, iterations=1)

    mask_f = mask.astype(np.float32) / 255.0
    feathered = cv2.GaussianBlur(mask_f, close_kernel, 0)

    return np.clip(feathered, 0, 1)


def blend_whitened(original_bgr, whitened_bgr, soft_mask):
    alpha = soft_mask[..., np.newaxis]

    blended = original_bgr.astype(np.float32) * (1 - alpha) + whitened_bgr.astype(np.float32) * alpha
    return blended.astype(np.uint8)


def apply_whitening_and_blend(original_image,eye_mask, mask, hsv_image, H, S, V, MOPI):
    # lower_hsv, upper_hsv, eye_mask = segment_face(hsv_image, face_landmark)
    #
    # hard_mask = create_robust_skin_mask(
    #     hsv_image, lower_hsv, upper_hsv, eye_mask,
    #     close_kernel=(21, 21)
    # )
    close_kernel = (15, 15)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(eye_mask))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_f = mask.astype(np.float32) / 255.0
    mask = cv2.GaussianBlur(mask_f, close_kernel, 0)

    whitened = meibai(hsv_image.copy(), mask * 255.0, H, S, V)

    whitened = apply_bilateral_filter(whitened, (mask * 255.0).astype(np.uint8), MOPI)

    final_image = blend_whitened(original_image, whitened, mask)

    return np.clip(final_image, 0, 255).astype(np.uint8)

def get_eye_mask(image, face_landmarks):
    h, w, _ = image.shape


    eye_mask = np.zeros((h, w), dtype=np.uint8)
    lip_mask = np.zeros((h, w), dtype=np.uint8)
    right_pts = cv2.convexHull(np.array([face_landmarks[i] for i in RIGHT_EYE_RING], np.int32))
    left_pts = cv2.convexHull(np.array([face_landmarks[i] for i in LEFT_EYE_RING], np.int32))
    lips_pts = np.array([face_landmarks[i] for i in LIPS_INDICES], np.int32)

    cv2.fillConvexPoly(eye_mask, left_pts, 255)
    cv2.fillConvexPoly(eye_mask, right_pts, 255)
    cv2.fillConvexPoly(lip_mask, lips_pts, 255)
    eye_mask = cv2.bitwise_or(eye_mask, lip_mask)
    return eye_mask