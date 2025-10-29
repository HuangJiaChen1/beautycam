
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import selfie_segmentation

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
#
# # For static images:
# IMAGE_FILES = ['res-high.jpg']
# BG_COLOR = (0, 0, 0)
# BLUR_KERNEL = (55, 55)
# BLUR_SIGMA = 30
#
# with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
#     for idx, file in enumerate(IMAGE_FILES):
#         image = cv2.imread(file)
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         # Process the image and get segmentation
#         results = selfie_segmentation.process(image_rgb)
#         mask = results.segmentation_mask  # Values in [0,1]
#
#         print(mask)
#         mask = cv2.GaussianBlur(mask, (1, 1), 0)
#         mask_3d = mask[:, :, np.newaxis]
#
#         bg_blurred = cv2.GaussianBlur(image, BLUR_KERNEL, BLUR_SIGMA)
#
#         fg = image.astype(np.float32)
#         bg = bg_blurred.astype(np.float32)
#
#         alpha = mask_3d
#         output = (fg * alpha) + (bg * (1 - alpha))
#         output = output.astype(np.uint8)
#
#         # bg_solid = np.full(image.shape, BG_COLOR, dtype=np.uint8)
#         # output = (image.astype(np.float32) * alpha) + (bg_solid.astype(np.float32) * (1 - alpha))
#         # output = output.astype(np.uint8)
#
#         # Save result
#         cv2.imwrite(f'selfie_segmentation_smooth_output_{idx}.png', output)
#


def bg_blur(image_bgr):
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_seg:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = selfie_seg.process(image_rgb)
        mask = results.segmentation_mask

        # Soften mask
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask_3d = mask[:, :, np.newaxis]

        # Blur background strongly
        bg_blurred = cv2.GaussianBlur(image_bgr, (55, 55), 30)

        # Alpha blend
        fg = image_bgr.astype(np.float32)
        bg = bg_blurred.astype(np.float32)
        alpha = mask_3d

        output = (fg * alpha) + (bg * (1 - alpha))
        return output.astype(np.uint8)