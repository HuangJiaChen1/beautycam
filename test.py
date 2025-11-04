import cv2
import numpy as np


def linear_light_blend(base, blend):
    """
    Apply Linear Light blending (Photoshop-style) on [0,255] images.
    base: Bottom layer (e.g., LPF or original).
    blend: Top layer (e.g., remapped HPF).
    """
    # Normalize to [0,1]
    base_norm = base.astype(np.float32) / 255.0
    blend_norm = blend.astype(np.float32) / 255.0

    # Linear Light formula per channel
    result_norm = np.where(
        blend_norm > 0.5,
        base_norm + 2 * (blend_norm - 0.5),
        base_norm + 2 * blend_norm - 1
    )

    # Clamp and denormalize
    result_norm = np.clip(result_norm, 0.0, 1.0)
    return (result_norm * 255.0).astype(np.uint8)


def retouch(image,strength):
    lpf = cv2.GaussianBlur(image, (3, 3), 0)
    hpf = image.astype(np.int16) - lpf.astype(np.int16)  # Signed to handle negatives

    lpf = cv2.bilateralFilter(lpf, 32, 128*strength, 16*strength)
    # cv2.imshow('bilateral', cv2.bilateralFilter(image, 32, 128, 16))
    # lpf = cv2.GaussianBlur(lpf, (51, 51), 0)
    # cv2.imshow('lpf+hpf', (lpf+hpf-255).astype(np.uint8))
    hpf_remapped = hpf*0.5 + 128
    # cv2.imshow('hpf_remapped', hpf_remapped.astype(np.uint8))
    hpf_remapped = np.clip(hpf_remapped, 0, 255).astype(np.uint8)

    # Blend: Use LPF as base, remapped HPF as blend (for sharpening effect)
    result = linear_light_blend(lpf, hpf_remapped)
    return result
#
# # Load and compute filters
# image = cv2.imread('imgs/face4.jpg')
# # Display
# cv2.imshow('Original', image)
# result = retouch(image)
# cv2.imshow('Result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()