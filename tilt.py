# --------------------------------------------------------------
#  head_pose_image.py   –  static-image version
# --------------------------------------------------------------
import cv2
import numpy as np
import mediapipe as mp
import time
import glob
import os

# ------------------------------------------------------------------
# MediaPipe setup
# ------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,          # <-- important for images
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=2, circle_radius=1)

# ------------------------------------------------------------------
# Helper: process ONE image and return the annotated frame
# ------------------------------------------------------------------
def process_image(img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read {img_path}")

    start = time.time()

    image = cv2.flip(img_bgr, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    img_h, img_w, _ = image.shape
    face_2d = []
    face_3d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                # Keep only the 6 points you used for PnP
                if idx in (33, 263, 1, 61, 291, 199):
                    if idx == 1:                                   # nose tip
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z+1)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z+1])                # z is relative depth

            # ------------------------------------------------------------------
            # 1. PnP
            # ------------------------------------------------------------------
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                   [0, focal_length, img_h / 2],
                                   [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_coeffs)
            print(rot_vec)
            print(trans_vec)
            # ------------------------------------------------------------------
            # 2. Rotation matrix → Euler angles
            # ------------------------------------------------------------------
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            x_deg = angles[0] * 360
            y_deg = angles[1] * 360
            z_deg = angles[2] * 360

            if y_deg < -10:
                text = "Looking Left"
            elif y_deg > 10:
                text = "Looking Right"
            elif x_deg < -10:
                text = "Looking Down"
            elif x_deg > 10:
                text = "Looking Up"
            else:
                text = "Forward"

            # ------------------------------------------------------------------
            # 4. Nose direction line
            # ------------------------------------------------------------------
            nose_3d_proj, _ = cv2.projectPoints(
                np.array([nose_3d]), rot_vec, trans_vec, cam_matrix, dist_coeffs)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y_deg * 10), int(nose_2d[1] - x_deg * 10))
            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # ------------------------------------------------------------------
            # 5. On-screen info
            # ------------------------------------------------------------------
            cv2.putText(image, text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, f"x: {x_deg:.1f}", (500, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, f"y: {y_deg:.1f}", (500, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, f"z: {z_deg:.1f}", (500, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # ------------------------------------------------------------------
            # 6. Draw landmarks (contours only)
            # ------------------------------------------------------------------
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    # ------------------------------------------------------------------
    # FPS (per image)
    # ------------------------------------------------------------------
    end = time.time()
    fps = 1 / (end - start) if (end - start) > 0 else 0
    cv2.putText(image, f'FPS: {int(fps)}', (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    return image, fps

if __name__ == "__main__":
    # ---- OPTION 1: single image ------------------------------------------------
    IMG_PATH = "imgs/face_celian.jpg"          # <-- change to your picture
    annotated, fps = process_image(IMG_PATH)
    print(f"FPS: {fps:.2f}")
    cv2.imshow("Head Pose – static image", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
