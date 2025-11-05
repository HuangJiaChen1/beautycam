import cv2
import numpy as np
import mediapipe as mp

LEFT_CHEEK = [277, 426, 288, 356]
RIGHT_CHEEK = [47, 206, 58, 127]
C_LEFT = 280
C_RIGHT = 50

def compute_cheek_masks(
    image,
    landmarks,
    box_ratio=[1, 1],
    kernel_ratio=[[0.7, 0.7], [0.5, 0.5]],
    mask_thresh=0.1,
    indices_right=None,
    indices_left=None,
):
    h, w, _ = image.shape
    points = np.array(landmarks)

    def sort_quad(pts):
        # Sort to top-left, top-right, bottom-right, bottom-left
        sorted_y = sorted(pts, key=lambda p: p[1])
        top_two = sorted(sorted_y[:2], key=lambda p: p[0])
        bottom_two = sorted(sorted_y[2:], key=lambda p: p[0])
        top_left, top_right = top_two[0], top_two[1]
        bottom_left, bottom_right = bottom_two[0], bottom_two[1]
        return np.array([top_left, top_right, bottom_right, bottom_left])

    def height_width(pts):
        if len(pts) == 0:
            return 0, 0
        min_pt = np.min(pts, axis=0)
        max_pt = np.max(pts, axis=0)
        return max_pt[1] - min_pt[1], max_pt[0] - min_pt[0]

    indices = [indices_right or RIGHT_CHEEK, indices_left or LEFT_CHEEK]
    c_indices = [C_RIGHT, C_LEFT]
    masks = []

    for i, ids in enumerate(indices):
        # 1) Get quad points
        pts = points[ids]

        # Sort quad for consistent order
        pts_sorted = sort_quad(pts)

        # 2) Get given center (no computation needed)
        center = points[c_indices[i]]

        # Full target points: sorted quad + center
        pts_full = np.vstack([pts_sorted, center])

        # 3) Height and width
        height, width = height_width(pts_full)

        # 4) Square side
        side = int(box_ratio[i] * min(width, height))
        if side % 2 == 0:
            side += 1  # Make odd

        # 5) Create delta and Gaussian kernel
        mask_delta = np.zeros((side, side), dtype=np.float32)
        midpoint = (side - 1) // 2
        mask_delta[midpoint, midpoint] = 1.0

        sigma_x = kernel_ratio[i][0] * (0.5 * (side - 1) - 1) + 0.8
        sigma_y = kernel_ratio[i][1] * (0.5 * (side - 1) - 1) + 0.8
        kernel = cv2.GaussianBlur(mask_delta, (side, side), sigmaX=sigma_x, sigmaY=sigma_y,
                                  borderType=cv2.BORDER_ISOLATED)

        # 6) Normalize and threshold
        min_val, max_val = np.min(kernel), np.max(kernel)
        norm_kernel = (kernel - min_val) / (max_val - min_val) if max_val > min_val else kernel
        norm_kernel -= mask_thresh
        norm_kernel = np.clip(norm_kernel, 0, 1)

        # 7) Source points for homography: top-left, top-right, bottom-right, bottom-left, center
        src_points = np.array([
            [0, 0],                # top-left
            [side - 1, 0],         # top-right
            [side - 1, side - 1],  # bottom-right
            [0, side - 1],         # bottom-left
            [midpoint, midpoint]   # center
        ], dtype=np.float32)

        # Target points: sorted quad + center
        tgt_points = np.array(pts_full, dtype=np.float32)

        # 8) Homography
        homography, _ = cv2.findHomography(src_points, tgt_points)

        # 9) Warp
        warped_mask = cv2.warpPerspective(norm_kernel, homography, (w, h))

        # 10) Append
        masks.append(warped_mask)

    return masks  # [right, left]


def apply_blush(image,landmarks, strength=0.0, color=(255, 20, 20), indices_right=RIGHT_CHEEK, indices_left=LEFT_CHEEK):

    if image is None:
        return None

    if strength <= 0:
        return image.copy()


    # Compute cheek masks and combine
    right_mask, left_mask = compute_cheek_masks(
        image,
        landmarks,
        indices_right=indices_right,
        indices_left=indices_left,
    )
    mask = np.maximum(right_mask, left_mask).astype(np.float32)

    # Prepare for YCrCb blending
    mask_normalized = np.clip(mask, 0.0, 1.0)
    im_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0

    bgr_mat = np.array([[[color[2], color[1], color[0]]]], dtype=np.uint8)
    ycrcb_mat = cv2.cvtColor(bgr_mat, cv2.COLOR_BGR2YCrCb)
    target_ycrcb = (ycrcb_mat[0, 0].astype(np.float32) / 255.0)[::-1]

    indices = np.where(mask_normalized > 0)
    if len(indices[0]) == 0:
        return image.copy()

    weights = mask_normalized[indices]
    masked_pixels = im_ycrcb[indices]
    m = np.average(masked_pixels, axis=0, weights=weights)

    mpxl = np.clip(mask_normalized[indices] * float(strength), 0.0, 1.0)
    src_pxl = im_ycrcb[indices]

    for idx in range(3):
        im_ycrcb[indices[0], indices[1], idx] = (
            mpxl * (target_ycrcb[idx] + (src_pxl[:, idx] - m[idx])) + (1 - mpxl) * src_pxl[:, idx]
        )

    im_out = cv2.cvtColor(im_ycrcb, cv2.COLOR_YCrCb2BGR) * 255.0
    result = np.clip(im_out, 0, 255).astype(np.uint8)
    return result