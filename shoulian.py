import cv2
import mediapipe as mp
import numpy as np
from affine_transformation import warp_image_piecewise_affine

left_cheek = [132,58,172, 136, 150, 149, 176, 148]
right_cheek = [361,288,397, 365, 379, 378, 400, 377]
mid_jaw = 152


# left_cheek = [147,213,192,214]
# right_cheek = [376,433,415,434]
lrn = [58, 288, 4]
mp_face_mesh = mp.solutions.face_mesh


def calculate_midpoint(lrn):
    jaw_left ,jaw_right, nose_tip = lrn
    jaw_left = np.array(jaw_left)
    jaw_right = np.array(jaw_right)
    nose_tip = np.array(nose_tip)
    # print(jaw_left, jaw_right, nose_tip)
    midpoint = (jaw_left + jaw_right + nose_tip) / 3.0
    # print(project_to_2d(midpoint))
    return midpoint


def shrink_points_towards_midpoint(cheek_points, midpoint, shrink_factor):
    diff = cheek_points - midpoint
    new_cheek_points = diff * shrink_factor + midpoint
    return new_cheek_points

def calc_slim_points(lcp, rcp,mjp, scale_factor, lrn_points):
    cheek_points = np.array(lcp + rcp+mjp)
    center = calculate_midpoint(lrn_points)
    new_points_3d = shrink_points_towards_midpoint(cheek_points, center, scale_factor)
    # print(new_points_3d)
    return new_points_3d.tolist()


def get_points_3D(dst, indices):
    return [dst[i] for i in indices]

def shoulian(SHOULIAN,dst_points):
    left_cheek_points = get_points_3D(dst_points, left_cheek)
    right_cheek_points = get_points_3D(dst_points, right_cheek)
    mid_jaw_point = get_points_3D(dst_points,[mid_jaw])

    lrn_points = get_points_3D(dst_points, lrn)

    new_cheek_points = calc_slim_points(left_cheek_points, right_cheek_points, mid_jaw_point,SHOULIAN, lrn_points)
    i = 0
    for idx in left_cheek+right_cheek + [mid_jaw]:
        dst_points[idx] = (new_cheek_points[i][0], new_cheek_points[i][1], new_cheek_points[i][2])
        i += 1