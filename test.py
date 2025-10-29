import cv2
import numpy as np

def detect_pimples_from_hpf(img_bgr,
                            skin_mask_uint8,          # <-- NEW
                            blur_radius=25,
                            thresh_block=151,
                            C=5):

    img = img_bgr.astype(np.float32)

    ksize = int(blur_radius * 2 + 1) | 1
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
    hpf = img - blurred
    hpf = np.clip(hpf + 128, 0, 255).astype(np.uint8)

    hpf_gray = cv2.cvtColor(hpf, cv2.COLOR_BGR2GRAY)

    skin_f = skin_mask_uint8.astype(np.float32) / 255.0
    cv2.imshow('skin', skin_f)
    hpf_gray_masked = (hpf_gray * skin_f).astype(np.uint8)
    cv2.imshow("hpf_gray", hpf_gray_masked)
    thresh = cv2.adaptiveThreshold(
        hpf_gray_masked, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, thresh_block, C
    )
    face_artefacts = cv2.bitwise_not(cv2.bitwise_xor(thresh,skin_mask_uint8))
    print(face_artefacts)
    cv2.imshow("thresh", thresh)
    cv2.imshow('face artefacts',cv2.bitwise_and(face_artefacts, skin_mask_uint8))
    cv2.waitKey(0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(face_artefacts.astype(np.float32), (1, 1), 10) / 255.0
    mask = mask * skin_f
    cv2.imshow('mask',mask)
    cv2.waitKey(0)
    # <-- force zero outside skin
    return mask, hpf_gray, thresh

def smooth_and_sharpen_skin(
        img_bgr,
        skin_mask_uint8,          # binary 0/255
        MOPI,
        blur_radius=25,
        bilateral_d=32,
        sharpen_strength=1.5,
        eye_protect=True,
        landmarks=None,
):
    img = img_bgr.astype(np.float32)
    h, w = img.shape[:2]

    skin_f = skin_mask_uint8.astype(np.float32) / 255.0

    pimple_mask, hpf_gray, _ = detect_pimples_from_hpf(
        img_bgr, skin_mask_uint8, blur_radius, C=MOPI)

    eye_mask = np.zeros((h, w), np.float32)
    if eye_protect and landmarks is not None:
        left  = landmarks[36:42].astype(np.int32)
        right = landmarks[42:48].astype(np.int32)
        cv2.fillPoly(eye_mask, [left, right], 1.0)
        eye_mask = cv2.dilate(eye_mask, None, iterations=15)
        eye_mask = cv2.GaussianBlur(eye_mask, (51, 51), 15)
    sharpen_mask = 1.0 - 0.7 * eye_mask

    ksize = int(blur_radius * 2 + 1) | 1
    lpf = cv2.GaussianBlur(img, (ksize, ksize), 0)
    hpf_full = img - lpf

    hpf_clean = hpf_full * (1 - pimple_mask[..., np.newaxis])
    cv2.imshow('hpfclean', hpf_clean)
    cv2.waitKey(0)
    lpf_uint8 = np.clip(lpf, 0, 255).astype(np.uint8)
    lpf_smooth = cv2.bilateralFilter(lpf_uint8, bilateral_d, 100, 15)
    lpf_smooth = lpf_smooth.astype(np.float32)

    hpf_sharpened = hpf_clean * sharpen_strength
    hpf_sharpened = hpf_sharpened * sharpen_mask[..., np.newaxis]

    skin_processed = lpf_smooth + hpf_sharpened
    skin_processed = np.clip(skin_processed, 0, 255)
    cv2.imshow('skinprocessed', skin_processed)
    cv2.waitKey(0)
    alpha = skin_f[..., np.newaxis]
    result = (img * (1 - alpha) + skin_processed * alpha).astype(np.uint8)

    return result, pimple_mask, hpf_clean, skin_f