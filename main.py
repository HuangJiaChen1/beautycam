import cv2
import numpy as np

def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def get_source_map_elliptical(pos, center, minor, major, angle, scale_ratio, radius_factor=2):
    # pos: (h, w, 2) [x, y]
    # center: [cx, cy]
    # minor, major: full axis lengths
    # angle: rotation angle to minor axis
    delta = pos - center[None, None, :]
    semi_a = (major / 2) * radius_factor  # semi-major
    semi_b = (minor / 2) * radius_factor  # semi-minor
    theta_major = angle - 90.0
    alpha = -theta_major
    angle_rad = np.deg2rad(alpha)
    cos_alpha = np.cos(angle_rad)
    sin_alpha = np.sin(angle_rad)
    delta_rot_x = delta[:, :, 0] * cos_alpha + delta[:, :, 1] * sin_alpha
    delta_rot_y = -delta[:, :, 0] * sin_alpha + delta[:, :, 1] * cos_alpha
    d_ell = np.sqrt((delta_rot_x / semi_a) ** 2 + (delta_rot_y / semi_b) ** 2)
    ss = smoothstep(0.0, 1.0, d_ell)
    gamma = 1.0 - scale_ratio * (ss - 1.0)**2
    mask = d_ell < 1.0
    gamma = np.where(mask, gamma, 1.0)
    source = center[None, None, :] + gamma[..., None] * delta
    return source[..., 0], source[..., 1]

def apply_big_eye_effect(image, scale_ratio, face_landmarks, radius_factor=3):
    """
    Apply big eye effect to the image using provided face landmarks, with elliptical warping.

    :param image: Input image as numpy array (BGR format)
    :param scale_ratio: Amplification factor (e.g., 1.2 for 20% enlargement)
    :param face_landmarks: List or array of face landmark points in pixel coordinates (index: [x, y])
    :param radius_factor: Factor to scale the ellipse axes (default: 1.5)
    :return: Processed image with big eye effect
    """
    # Get image dimensions
    h, w = image.shape[:2]
    scale_ratio -= 1  # As per your modification, assuming input >1

    # Create meshgrid for positions
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    pos = np.stack((xx, yy), axis=-1).astype(np.float32)  # (h, w, 2)

    # Initialize map as identity
    map_x = xx.astype(np.float32)
    map_y = yy.astype(np.float32)

    if face_landmarks:
        # Centers
        left_center = np.array(face_landmarks[468], dtype=np.float32)  # Image left eye (subject's right)
        right_center = np.array(face_landmarks[473], dtype=np.float32)  # Image right eye (subject's left)

        # Landmark indices for eye contours
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]  # Image left eye
        right_eye_indices = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]  # Image right eye

        # For left eye
        left_eye_points = np.array([face_landmarks[i] for i in left_eye_indices], dtype=np.float32)
        if len(left_eye_points) >= 5:
            try:
                ellipse = cv2.fitEllipse(left_eye_points)
                center_fit, axes, angle = ellipse
                minor, major = min(axes), max(axes)
                # Apply warp for left eye
                temp_pos = np.stack((map_x, map_y), axis=-1)
                map_x, map_y = get_source_map_elliptical(temp_pos, left_center, minor, major, angle, scale_ratio, radius_factor)
            except:
                pass  # Fallback to no warp or circular if needed

        # For right eye
        right_eye_points = np.array([face_landmarks[i] for i in right_eye_indices], dtype=np.float32)
        if len(right_eye_points) >= 5:
            try:
                ellipse = cv2.fitEllipse(right_eye_points)
                center_fit, axes, angle = ellipse
                minor, major = min(axes), max(axes)
                # Apply warp for right eye
                temp_pos = np.stack((map_x, map_y), axis=-1)
                map_x, map_y = get_source_map_elliptical(temp_pos, right_center, minor, major, angle, scale_ratio, radius_factor)
            except:
                pass  # Fallback
    # Remap the image
    output_image = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)

    return output_image