
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import selfie_segmentation

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation



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