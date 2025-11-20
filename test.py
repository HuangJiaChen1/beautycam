# ========================
# PERIORBITAL DARK CIRCLES REMOVER
# ========================
import cv2
import numpy as np

# Under-eye region polygons (tight around the lower eyelid / tear trough)
under_eye_left_indices  = [33, 133, 130, 159, 145, 153, 154, 155, 133]   # left eye (subject's right)
under_eye_right_indices = [263, 362, 359, 386, 374, 380, 381, 382, 362]  # right eye (subject's left)

# Extra safety: full eye regions to exclude completely (eyelids + iris)
eye_left_indices  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
eye_right_indices = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

eyebrow_left_indices  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
eyebrow_right_indices = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

def create_under_eye_mask(landmarks, h, w):
    left_pts  = np.array([landmarks[i] for i in under_eye_left_indices],  np.int32)
    right_pts = np.array([landmarks[i] for i in under_eye_right_indices], np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [left_pts, right_pts], 255)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,25)), iterations=2)
    return mask

def create_forbidden_eye_mask(landmarks, h, w):
    # Full eyes + eyebrows + a big dilation so nothing ever leaks
    pts1 = np.array([landmarks[i] for i in eye_left_indices + eyebrow_left_indices],  np.int32)
    pts2 = np.array([landmarks[i] for i in eye_right_indices + eyebrow_right_indices], np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts1, pts2], 255)
    mask = cv2.dilate(mask, None, iterations=25)   # huge safety buffer
    return mask.astype(bool)

def detect_under_eye_darkness(image, landmarks, strength=1.0):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    region_mask_u8 = create_under_eye_mask(landmarks, h, w)

    # CLAHE + high-pass exactly like your nasolabial version
    clahe = cv2.createCLAHE(clipLimit=3.0 + 4*strength, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (0,0), sigmaX=2.5)
    highpass = cv2.addWeighted(enhanced, 1.8, blurred, -0.8, 0)
    dark_response = 255 - highpass

    mean_dark = cv2.mean(dark_response, mask=region_mask_u8)[0]
    thresh = mean_dark + 20 * strength
    _, binary = cv2.threshold(dark_response, int(thresh), 255, cv2.THRESH_BINARY)

    dark_mask = binary.astype(np.float32) / 255.0
    dark_mask = cv2.bitwise_and(dark_mask, dark_mask, mask=region_mask_u8)

    # ========= ABSOLUTE EXCLUSION OF EYES & EYEBROWS =========
    forbidden = create_forbidden_eye_mask(landmarks, h, w)
    dark_mask[forbidden] = 0.0
    # ===============================================================

    # Gentle cleanup & feather
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
    dark_mask = cv2.GaussianBlur(dark_mask, (25,25), 0)

    return dark_mask

def remove_dark_circles(image, landmarks, strength=1.0):
    """
    strength: 0.5 ~ 2.0  → 1.0 is already very strong and natural
    """
    if strength <= 0:
        return image.copy()

    strength = np.clip(strength, 0.0, 3.0)
    h, w = image.shape[:2]

    dark_mask = detect_under_eye_darkness(image, landmarks, strength)

    if dark_mask.max() < 0.05:
        return image.copy()

    # ==== 1. Very strong bilateral (texture & bags removal) ====
    d = max(8, int(40 * strength))
    smoothed = cv2.bilateralFilter(image, d=d, sigmaColor=int(300*strength), sigmaSpace=int(50*strength))
    smoothed = cv2.bilateralFilter(smoothed, d=d, sigmaColor=int(300*strength), sigmaSpace=int(50*strength))  # 2nd pass

    # ==== 2. Brighten + match surrounding skin color ====
    # Work in Lab color space for proper brightness/color transfer
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    smoothed_lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2Lab)

    # Extract L channel (lightness) only under eyes
    L_under_eye = lab[..., 0]
    mask_u8 = (dark_mask * 255).astype(np.uint8)

    # Compute average skin lightness from cheeks (safe zones next to under-eye)
    cheek_left  = np.array([landmarks[i] for i in [123, 117, 118, 119, 206, 203, 205]], np.int32)
    cheek_right = np.array([landmarks[i] for i in [352, 346, 347, 348, 426, 423, 425]], np.int32)
    cheek_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(cheek_mask, [cheek_left, cheek_right], 255)
    mean_skin_L = cv2.mean(lab[..., 0], mask=cheek_mask)[0]

    # Current average lightness under eyes
    mean_dark_L = cv2.mean(L_under_eye, mask=mask_u8)[0]

    # How much we need to lift
    delta_L = mean_skin_L - mean_dark_L
    boost = delta_L * 1.2 * strength   # 1.2× overshoot slightly for punch

    # Apply brightness boost only where mask exists
    smoothed_lab[..., 0] = np.clip(smoothed_lab[..., 0].astype(np.float32) + boost * dark_mask, 0, 255).astype(np.uint8)

    # Slight color warming (dark circles are often bluish)
    smoothed_lab[..., 1] = np.clip(smoothed_lab[..., 1].astype(np.int16)  + int(8 * strength), -128, 127).astype(np.uint8)   # less red → warmer
    smoothed_lab[..., 2] = np.clip(smoothed_lab[..., 2].astype(np.int16)  + int(10 * strength), -128, 127).astype(np.uint8)   # more yellow

    smoothed = cv2.cvtColor(smoothed_lab, cv2.COLOR_Lab2BGR)

    # ==== 3. Poisson blending instead of alpha blend → zero halo, perfect color match ====
    # Find center point inside mask for Poisson seamless cloning
    moments = cv2.moments(mask_u8)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = w//2, h//2

    try:
        result = cv2.seamlessClone(
            src=smoothed,
            dst=image,
            mask=mask_u8,
            p=(cx, cy),
            flags=cv2.NORMAL_CLONE   # or MIXED_CLONE for even softer
        )
    except:
        # Fallback to alpha blend if seamlessClone fails
        mask3d = dark_mask[..., np.newaxis]
        result = (image.astype(np.float32) * (1 - mask3d) + smoothed.astype(np.float32) * mask3d)
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result

import cv2
import mediapipe as mp
import numpy as np

# ========================
# Your two functions from before (copy-paste them here)
# ========================
# nasolabial_folds_filter_precise(...)
# remove_dark_circles(...)
# (plus the helper functions: create_region_mask, detect_dark_ridges, etc.)
# Just paste everything we built earlier into this file

# ========================
# MediaPipe setup
# ========================
mp_face_mesh = mp.solutions.face_mesh

# Use static_image_mode=True for photos (faster + more accurate)
# refine_landmarks=True → gives you all 468 points (not just 468 sparse)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ========================
# Main processing function
# ========================
def process_image(image_path, nasolabial_strength=1.0, darkcircle_strength=1.0):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print("No face detected")
        return image

    # Get the first (and only) face
    face_landmarks = results.multi_face_landmarks[0]

    # Convert to your exact format: list/array of [x_pixel, y_pixel] (468 × 2)
    landmarks = []
    for lm in face_landmarks.landmark:
        landmarks.append([lm.x * w, lm.y * h])  # ← this is what your code expects

    landmarks = np.array(landmarks)  # shape (468, 2)

    image = remove_dark_circles(image, landmarks, strength=darkcircle_strength)

    return image


# ========================
# Run it
# ========================
if __name__ == "__main__":
    input_path = "imgs/Face 12-16 (14).jpeg"

    result = process_image(
        input_path,
        nasolabial_strength=1.0,  # 0.7–1.5 usually perfect
        darkcircle_strength=1.0  # 0.8–1.4 usually perfect
    )

    # cv2.imwrite(output_path, result)
    cv2.imshow("Beauty Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
