import cv2
import mediapipe as mp
import numpy as np
from affine_transformation import warp_image_piecewise_affine

left_cheek = [172, 136, 150, 149, 176, 148]
right_cheek = [397, 365, 379, 378, 400, 377]

# left_cheek = [147,213,192,214]
# right_cheek = [376,433,415,434]
lrn = [58, 288, 4]

mp_face_mesh = mp.solutions.face_mesh


def calculate_midpoint(lrn):
    jaw_left ,jaw_right, nose_tip = lrn
    # print(jaw_left, jaw_right, nose_tip)
    midpoint = (jaw_left + jaw_right + nose_tip) / 3.0
    # print(project_to_2d(midpoint))
    return midpoint


def transform_to_3d(x, y, z):
    return (x * z, y * z, z)


def shrink_points_towards_midpoint(cheek_points, midpoint, shrink_factor):
    diff = cheek_points - midpoint
    new_cheek_points = diff * shrink_factor + midpoint
    return new_cheek_points


def project_to_2d(point_3d):
    x, y, z = point_3d
    if z != 0:
        return (x / z, y / z)
    return (x, y)  # Return original if z == 0 (edge case)


def calc_slim_points(lcp, rcp, scale_factor, lrn_points):
    cheek_points = np.array(lcp + rcp)
    transformed_points = np.array([transform_to_3d(x, y, z) for x, y, z in cheek_points])
    transformed_lrn = np.array([transform_to_3d(x, y, z) for x, y, z in lrn_points])
    center = calculate_midpoint(transformed_lrn)
    new_points_3d = shrink_points_towards_midpoint(transformed_points, center, scale_factor)

    new_points_2d = np.array([project_to_2d(point_3d) for point_3d in new_points_3d])

    return new_points_2d.tolist()


def get_points_3D(image, landmarks, indices):
    points = [(landmarks[i].x, landmarks[i].y, landmarks[i].z+1) for i in indices]
    points_pixels = [(int(p[0] * image.shape[1]), int(p[1] * image.shape[0]), p[2]) for p in points]
    return points_pixels

def get_points_2D(image, landmarks, indidces):
    points = [(landmarks[i].x, landmarks[i].y) for i in indidces]
    points_pixels = [(int(p[0] * image.shape[1]), int(p[1] * image.shape[0])) for p in points]
    return points_pixels

def shoulian(annotated_image, face_landmarks, SHOULIAN, src_points, dst_points):
    left_cheek_points = get_points_3D(annotated_image, face_landmarks.landmark, left_cheek)
    right_cheek_points = get_points_3D(annotated_image, face_landmarks.landmark, right_cheek)
    left_cheek_points2d = get_points_2D(annotated_image, face_landmarks.landmark, left_cheek)
    right_cheek_points2d = get_points_2D(annotated_image, face_landmarks.landmark, right_cheek)
    src_points.extend(left_cheek_points2d)
    src_points.extend(right_cheek_points2d)

    lrn_points = get_points_3D(annotated_image, face_landmarks.landmark, lrn)

    new_cheek_points = calc_slim_points(left_cheek_points, right_cheek_points, SHOULIAN, lrn_points)
    dst_points.extend(new_cheek_points)
    # print(src_points)
    # print(dst_points)
