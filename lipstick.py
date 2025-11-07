TEETH_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]  # SORTED
LIPS_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]  # SORTED

# Define upper and lower lip separately for better control
UPPER_LIP_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409]
LOWER_LIP_INDICES = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# Mouth corner indices for measuring width
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291

import cv2
import mediapipe as mp
import numpy as np


def segment_lips(hsv_image, mask, within_sd=1.8):
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


def is_mouth_open(landmarks, threshold_ratio=0.08):
    """
    Detect if mouth is open by analyzing the teeth region.
    When mouth is closed, teeth landmarks form a very narrow ellipse in the y-direction.

    Args:
        landmarks: facial landmarks
        threshold_ratio: ratio threshold (teeth_height / teeth_width)
    """
    teeth_pts = np.array([landmarks[i] for i in TEETH_INDICES])

    # Calculate bounding box of teeth region
    y_coords = teeth_pts[:, 1]
    x_coords = teeth_pts[:, 0]

    teeth_height = np.max(y_coords) - np.min(y_coords)
    teeth_width = np.max(x_coords) - np.min(x_coords)

    # Avoid division by zero
    if teeth_width == 0:
        return False

    # Calculate aspect ratio (height/width)
    # When mouth is closed, this ratio will be very small (narrow in y-direction)
    aspect_ratio = teeth_height / teeth_width

    # If aspect ratio is above threshold, mouth is open
    return aspect_ratio > threshold_ratio


def create_closed_lip_contour(landmarks):
    """Create a closed contour that fills the gap between upper and lower lips"""
    upper_lip_pts = np.array([landmarks[i] for i in UPPER_LIP_INDICES], np.int32)
    lower_lip_pts = np.array([landmarks[i] for i in LOWER_LIP_INDICES], np.int32)

    # Reverse lower lip to create a continuous contour
    closed_contour = np.vstack([upper_lip_pts, lower_lip_pts[::-1]])

    return closed_contour


def lipstick(image, landmarks, strength=1.0, color=(255, 20, 20)):
    if strength <= 0:
        return image.copy()

    h, w, _ = image.shape
    mouth_open = is_mouth_open(landmarks, threshold_ratio=0.08)

    if mouth_open:
        # For open mouth: use original approach with teeth exclusion
        lips_pts = np.array([landmarks[i] for i in LIPS_INDICES], np.int32)
        teeth_pts = np.array([landmarks[i] for i in TEETH_INDICES], np.int32)

        lip_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(lip_mask, [lips_pts], 255)
        lip_mask = cv2.GaussianBlur(lip_mask, (51, 51), 0)
        teeth_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(teeth_mask, [teeth_pts], 255)
        cv2.dilate(teeth_mask, np.ones((3, 3), np.uint8), iterations=1)

        mask = cv2.bitwise_and(lip_mask, cv2.bitwise_not(teeth_mask))
    else:
        # For closed mouth: use closed contour that fills the gap
        closed_contour = create_closed_lip_contour(landmarks)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [closed_contour], 255)

        # Apply Gaussian blur for smooth edges
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
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