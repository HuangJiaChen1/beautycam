import cv2
import mediapipe as mp
import time

import numpy as np

from affine_transformation import warp_with_triangulation
from dayan import dayan
from meibai import segment_face, apply_whitening_and_blend
from shoulian import shoulian
from test import smooth_and_sharpen_skin

# 大眼强度
DAYAN = 1.2 # 1 以上变大
SHOULIAN = 0.8 # 0.85-1 小了会很奇怪,越小越瘦
# 1.1, 0.9 效果最好
MEIBAI = 0 #0-1, 1为白
MOPI = 10 #越大越磨

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

IMAGE_FILES = ['mopi.jpg']
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def get_points_all(image, landmarks):
    points = [(landmarks[i].x, landmarks[i].y) for i in range(len(landmarks))]
    points_pixels = [(int(p[0] * image.shape[1]), int(p[1] * image.shape[0])) for p in points]
    return points_pixels


def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            if not results.multi_face_landmarks:
                continue

            annotated_image = image.copy()
            src_points = []
            dst_points = []

            for face_landmarks in results.multi_face_landmarks:
                all_point = get_points_all(image, face_landmarks.landmark)
                lower_hsv, upper_hsv = segment_face(hsv_image, all_point)
                dayan(hsv_image, face_landmarks, DAYAN, src_points, dst_points)
                shoulian(hsv_image, face_landmarks, SHOULIAN, src_points, dst_points)

                # for landmark in dst_points:
                #     # Convert normalized coordinates to pixel values
                #     x = int(landmark[0])
                #     y = int(landmark[1])
                #     print(x,y)
                #     cv2.circle(annotated_image,(x, y), 1, (0, 255, 0), 2)
                # for landmark in src_points:
                #     # Convert normalized coordinates to pixel values
                #     x = int(landmark[0])
                #     y = int(landmark[1])
                #     print(x,y)
                #     cv2.circle(annotated_image,(x, y), 1, (0, 0, 0), 2)
                # cv2.imshow('image', annotated_image)
                # cv2.waitKey(0)
                processed_image = warp_with_triangulation(hsv_image, src_points, dst_points)
                skinMask = cv2.inRange(processed_image, lower_hsv, upper_hsv)
                meibai_image = apply_whitening_and_blend(cv2.cvtColor(processed_image, cv2.COLOR_HSV2BGR),
                                                         processed_image, skinMask, MEIBAI)

            out.write(meibai_image)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"Video processed and saved to {output_video_path}")

# input_video_path = 'input_video.mp4'
# output_video_path = 'output_video.mp4'
#
# start_time = time.time()
# process_video(input_video_path, output_video_path)
# end_time = time.time()
#
# print(f"Processing time: {end_time - start_time:.4f} seconds")

# image
# Time measurement and processing
with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
    start_time = time.time()
    for idx, file in enumerate(IMAGE_FILES):

        image = cv2.imread(file)
        rows, cols, _channels = map(int, image.shape)
        # image = cv2.resize(image, dsize=(cols//2 , rows//2 ),interpolation=cv2.INTER_LANCZOS4)
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            continue

        src_points = []
        dst_points = []

        for face_landmarks in results.multi_face_landmarks:
            all_point = get_points_all(image, face_landmarks.landmark)
            lower_hsv, upper_hsv = segment_face(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), all_point)
            dayan(image, face_landmarks, DAYAN,src_points, dst_points)
            shoulian(image, face_landmarks, SHOULIAN,src_points, dst_points)

            processed_image = warp_with_triangulation(image, src_points, dst_points)
            skinMask = cv2.inRange(cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV), lower_hsv, upper_hsv).astype(np.uint8)
            processed_image = apply_whitening_and_blend(processed_image, cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV), skinMask,MEIBAI, MOPI)

            # cv2.imshow('image', processed_image)
            # cv2.waitKey(0)

    # final_image = cv2.resize(processed_image, dsize=(cols, rows), interpolation=cv2.INTER_LANCZOS4)
    final_image = processed_image
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processing time for image {idx}: {elapsed_time:.4f} seconds")

    cv2.imshow('image', final_image)
    cv2.waitKey(0)
    cv2.imwrite(f'result_{idx}.jpg', final_image)
#
#
#
# # video stream
# cap = cv2.VideoCapture(0)
# with mp_face_mesh.FaceMesh(
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as face_mesh:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue
#
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(image)
#
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#     if not results.multi_face_landmarks:
#         continue
#
#     annotated_image = image.copy()
#
#     for face_landmarks in results.multi_face_landmarks:
#         processed_image = dayan(annotated_image, face_landmarks, DAYAN)
#         processed_image = shoulian(processed_image,face_landmarks, SHOULIAN)
#
#     processed_image = cv2.resize(processed_image, dsize=(cols, rows), interpolation=cv2.INTER_LANCZOS4)
#
#
#     cv2.imshow('MediaPipe Face Mesh', cv2.flip(processed_image, 1))
#     if cv2.waitKey(5) & 0xFF == 27:
#         break
#     cap.release()
