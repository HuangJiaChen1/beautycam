import cv2
import mediapipe as mp
import numpy as np

TEETH_INDICES = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]

def white_teeth(image, strength):
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=False
    )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        face_mesh.close()

    lm = results.multi_face_landmarks[0].landmark
    h, w, _ = image.shape

    teeth_pts = np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in TEETH_INDICES], np.int32)
    # right_pts = cv2.convexHull(right_pts)
    # left_pts = cv2.convexHull(left_pts)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(teeth_pts)], 255)
    # mask = mask / 255.0
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    # cv2.imshow("", cv2.bitwise_and(image, image, mask=mask))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5),0)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    mask_normalized = mask.astype(np.float32) / 255.0

    # Whitening in HSV space (keep mask and blending unchanged)
    # Convert to HSV and adjust V (brightness) upward and S (saturation) downward
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = cv2.split(hsv)

    # Parameters: tune as needed
    increase_v_factor = strength
    decrease_s_factor = 1/(strength ** 3)

    # Apply adjustments with clamping
    V = np.minimum(255.0, V * increase_v_factor)
    S = np.maximum(0.0,   S * decrease_s_factor)

    hsv_adj = cv2.merge([H, S, V]).astype(np.uint8)
    result = cv2.cvtColor(hsv_adj, cv2.COLOR_HSV2BGR)

    # cv2.imshow('image', image)
    # cv2.imshow('result', result)
    blended_image = (1 - mask_normalized[..., None]) * image + mask_normalized[..., None] * result
    blended_image = np.clip(blended_image,0,255).astype(np.uint8)
    # cv2.imshow('blended', blended_image)
    # cv2.waitKey(0)
    return blended_image
