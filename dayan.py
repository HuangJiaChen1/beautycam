import cv2
import mediapipe as mp
import numpy as np

from affine_transformation import warp_image_piecewise_affine

mp_face_mesh = mp.solutions.face_mesh

def calc_enlarge_points(eye_points,scale_factor, centre):
    # # based on mean
    # center = np.mean(eye_points, axis=0)

    #based on 3d coord
    center = np.array(centre).squeeze()
    eye_points = np.array(eye_points)
    scaled_points = eye_points * scale_factor + center * (1 - scale_factor)
    return scaled_points.tolist()


def get_eye_points(image, landmarks, eye_indices):
    eye_points = [(landmarks[i].x, landmarks[i].y) for i in eye_indices]
    eye_points_pixels = [(int(p[0] * image.shape[1]), int(p[1] * image.shape[0])) for p in eye_points]
    return eye_points_pixels

def get_eye_centre(image, landmarks, i):
    eye_points = [(landmarks[i].x, landmarks[i].y)]
    eye_points_pixels = [(int(p[0] * image.shape[1]), int(p[1] * image.shape[0])) for p in eye_points]
    return eye_points_pixels

def dayan(annotated_image, face_landmarks, DAYAN,src_points,dst_points):
    left_eye_index = []
    right_eye_index = []
    for (x, y) in mp_face_mesh.FACEMESH_LEFT_EYE:
        left_eye_index.append(x) if x not in left_eye_index else None
        left_eye_index.append(y) if y not in left_eye_index else None
    for (x1, y1) in mp_face_mesh.FACEMESH_RIGHT_EYE:
        right_eye_index.append(x1) if x1 not in right_eye_index else None
        right_eye_index.append(y1) if y1 not in right_eye_index else None
    left_eye_points = get_eye_points(annotated_image, face_landmarks.landmark, left_eye_index)
    left_centre = get_eye_centre(annotated_image, face_landmarks.landmark, 473)
    right_eye_points = get_eye_points(annotated_image, face_landmarks.landmark, right_eye_index)
    right_centre = get_eye_centre(annotated_image, face_landmarks.landmark, 468)
    src_points.extend(left_eye_points)
    src_points.extend(right_eye_points)
    left_points = calc_enlarge_points(left_eye_points, DAYAN, left_centre)
    right_points = calc_enlarge_points(right_eye_points, DAYAN, right_centre)
    dst_points.extend(left_points)
    dst_points.extend(right_points)
    # print(dst_points)
    # enlarged_image = enlarge_eye_using_triangulation(annotated_image, src_points, dst_points, DAYAN)
    # return enlarged_image