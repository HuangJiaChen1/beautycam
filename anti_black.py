import cv2
import numpy as np

# Your exact eye safety zones (never touch these pixels)
eye_left_indices  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
eye_right_indices = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

# Cheek sampling + dark-circle regions (unchanged from the pro version)
right_cheek_indices = [205, 207, 187, 123, 116, 117, 118, 50, 101, 36, 203, 206]
left_cheek_indices  = [425, 427, 411, 352, 345, 346, 347, 280, 330, 266, 423, 426]

dark_left_idx  = [341, 256, 252, 253,254,339,255,340,346,347,348,349,350,357]
dark_right_idx = [112, 26, 22, 23,24,110,25,111,117,118,119,120,121,128]

def create_forbidden_eye_mask(landmarks, h, w):
    """Absolutely forbidden zone – eyes will never be touched"""
    mask = np.zeros((h, w), dtype=np.uint8)
    left_pts  = np.array([landmarks[i] for i in eye_left_indices],  np.int32)
    right_pts = np.array([landmarks[i] for i in eye_right_indices], np.int32)
    cv2.fillPoly(mask, [left_pts, right_pts], 255)
    # Huge safety dilation so even eyelashes & slight landmark drift are safe
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)), iterations=1)
    return mask  # uint8 0/255

def get_cheek_color(image, landmarks):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array([landmarks[i] for i in left_cheek_indices], np.int32),
                        np.array([landmarks[i] for i in right_cheek_indices], np.int32)], 255)
    return cv2.mean(image, mask=mask)[:3]

def create_dark_circle_mask(landmarks, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array([landmarks[i] for i in dark_left_idx],  np.int32),
                        np.array([landmarks[i] for i in dark_right_idx], np.int32)], 255)
    return mask

def black_filter(image, landmarks, strength=1.0):
    if strength <= 0:
        return image.copy()

    strength = np.clip(strength, 0.0, 3.0)
    h, w = image.shape[:2]

    # 1. Sample real cheek skin tone
    cheek_bgr = get_cheek_color(image, landmarks)

    # 2. Create dark-circle correction mask
    dc_mask = create_dark_circle_mask(landmarks, h, w)
    dc_mask = cv2.dilate(dc_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 35)), iterations=1)

    # 3. === ABSOLUTE EYE PROTECTION ===
    eye_forbidden = create_forbidden_eye_mask(landmarks, h, w)   # uses your exact indices
    dc_mask = cv2.subtract(dc_mask, cv2.bitwise_and(dc_mask, eye_forbidden))  # hard removal

    # 4. Soft feather
    blend = dc_mask.astype(np.float32) / 255.0
    blend = cv2.GaussianBlur(blend, (21, 21), 0)

    # 5. Create correction layer filled with cheek color
    correction = np.full_like(image, cheek_bgr)

    # Optional extra brightness for very strong correction
    if strength > 1.2:
        lab = cv2.cvtColor(correction, cv2.COLOR_BGR2Lab)
        lab[..., 0] = np.clip(lab[..., 0] + int(10 * (strength - 1.0)), 0, 255)
        correction = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    # 6. Blend
    alpha = (blend[..., np.newaxis]) * strength
    alpha = np.clip(alpha, 0.0, 1.0)

    result = image.astype(np.float32) * (1 - alpha) + correction.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)