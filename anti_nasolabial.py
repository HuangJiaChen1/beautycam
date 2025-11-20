import cv2
import numpy as np

nose_indices = [193, 168, 417, 122, 351, 196, 419, 3, 248, 236, 456, 198, 420, 131, 360, 49, 279, 48, 278, 219, 439,
                59, 289, 218, 438, 237, 457, 44, 19, 274]
mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185,
                 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78,186,410]


def create_region_mask(landmarks, indices, h, w):
    points = []
    for idx in indices:
        x, y = landmarks[idx]
        points.append([x, y])
    if len(points) < 3:
        return np.zeros((h, w), dtype=np.float32)
    points = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(points)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask.astype(np.float32) / 255.0


def detect_dark_ridges(image, landmarks, strength=1.0):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Nasolabial region (allowed area) ---
    left_indices = [142, 36, 205, 207, 212, 202, 204, 194, 182, 61, 39, 167]
    right_indices = [371, 266, 425, 427, 432, 422, 424, 418, 406, 291, 269, 393]
    left_coords = np.array([landmarks[i] for i in left_indices], np.int32)
    right_coords = np.array([landmarks[i] for i in right_indices], np.int32)

    region_mask_u8 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(region_mask_u8, [left_coords, right_coords], 255)
    region_mask_u8 = cv2.dilate(region_mask_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))

    # --- Detection pipeline ---
    clahe = cv2.createCLAHE(clipLimit=2.0 + 3 * strength, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=3)
    highpass = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    dark_response = 255 - highpass

    mean_dark = cv2.mean(dark_response, mask=region_mask_u8)[0]
    thresh_val = mean_dark + 30 * strength
    _, binary = cv2.threshold(dark_response, int(thresh_val), 255, cv2.THRESH_BINARY)

    ridge_mask = binary.astype(np.float32) / 255.0
    ridge_mask = cv2.bitwise_and(ridge_mask, ridge_mask, mask=region_mask_u8)

    # =============================================
    # CRITICAL: ABSOLUTELY EXCLUDE NOSE & MOUTH
    # =============================================
    nose_mask = create_region_mask(landmarks, nose_indices, h, w)
    mouth_mask = create_region_mask(landmarks, mouth_indices, h, w)

    # Extra safety: dilate exclusion zones a lot
    forbidden = (nose_mask + mouth_mask) > 0
    forbidden = cv2.dilate(forbidden.astype(np.uint8), None, iterations=1).astype(bool)

    # Force zero in forbidden areas — this runs BEFORE any morphological cleanup or blur
    ridge_mask[forbidden] = 0.0
    # =============================================

    # Now safe to clean up and feather
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ridge_mask = cv2.morphologyEx(ridge_mask, cv2.MORPH_CLOSE, kernel)
    ridge_mask = cv2.morphologyEx(ridge_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    ridge_mask = cv2.GaussianBlur(ridge_mask, (21, 21), 0)

    return ridge_mask


def nasolabial_folds_filter(image, landmarks, strength=1.0):
    if strength <= 0:
        return image.copy()

    strength = np.clip(strength, 0.0, 2.0)

    ridge_mask = detect_dark_ridges(image, landmarks, strength)

    if ridge_mask.max() < 0.05:
        return image.copy()

    d = max(5, int(35 * strength))
    blurred = cv2.bilateralFilter(image, d=d, sigmaColor=int(220 * strength), sigmaSpace=int(35 * strength))

    mask3d = ridge_mask[..., np.newaxis]
    result = image.astype(np.float32) * (1 - mask3d) + blurred.astype(np.float32) * mask3d
    return np.clip(result, 0, 255).astype(np.uint8)