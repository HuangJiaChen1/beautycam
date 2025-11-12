import cv2
import numpy as np
import mediapipe as mp

mp_selfie = mp.solutions.selfie_segmentation
mp_face = mp.solutions.face_mesh

# --- load image ---
img = cv2.imread("imgs/2face.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w, _ = img.shape

# --- selfie segmentation ---
with mp_selfie.SelfieSegmentation(model_selection=0) as selfie:
    seg = selfie.process(img_rgb)
segmask = (seg.segmentation_mask > 0.5).astype(np.uint8) * 255
cv2.imshow("mask", segmask)
cv2.waitKey(0)

# --- facemesh for facial masking ---
with mp_face.FaceMesh(static_image_mode=True, refine_landmarks=True) as face:
    res = face.process(img_rgb)
    if res.multi_face_landmarks:
        landmarks = res.multi_face_landmarks[0]
        face_mask = np.zeros((h, w), dtype=np.uint8)
        points = np.array([(int(p.x * w), int(p.y * h)) for p in landmarks.landmark])
        cv2.fillConvexPoly(face_mask, cv2.convexHull(points), 255)
        mask = face_mask

# --- get skin region ---
skin_region = cv2.bitwise_and(img, img, mask=mask)
hsv = cv2.cvtColor(skin_region, cv2.COLOR_BGR2HSV)

# --- compute histogram from skin region ---
skin_pixels = hsv[mask > 0]
# reshape to a 2D image with 3 channels
skin_pixels = skin_pixels.reshape(-1, 1, 3)
hist = cv2.calcHist([skin_pixels], [0, 1], None, [180, 256], [0, 180, 0, 256])
cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

# --- backprojection ---
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
backproj = cv2.calcBackProject([hsv_img], [0, 1], hist, [0, 180, 0, 256], 1)
cv2.imshow('BackProj', backproj)
# --- refine mask ---
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
backproj = cv2.filter2D(backproj, -1, disc)
_, thresh = cv2.threshold(backproj, 50, 255, cv2.THRESH_BINARY)
skin_mask_final = cv2.merge([thresh, thresh, thresh])
skin_result = cv2.bitwise_and(img, skin_mask_final)
skin_result[segmask == 0] = 0

cv2.imshow("Skin Segmentation", skin_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
