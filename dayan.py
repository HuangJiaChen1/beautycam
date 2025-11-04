import cv2
import mediapipe as mp
import numpy as np

from affine_transformation import warp_image_piecewise_affine

mp_face_mesh = mp.solutions.face_mesh
L_EYE_INDICES = [374, 380, 390, 373, 249, 385, 384, 263, 466, 387, 386, 381, 382, 398, 388, 362]
R_EYE_INDICES = [154, 155, 33, 7, 246, 161, 159, 158, 144, 145, 173, 133, 157, 163, 153, 160]


def enlarge(cheek_points, midpoint, shrink_factor):
    diff = cheek_points - midpoint
    new_cheek_points = diff * shrink_factor + midpoint
    return new_cheek_points



def calc_eyes_points(lcp, scale_factor,center):
    transformed_points = np.array(lcp)
    transformed_center = np.array(center)
    new_points_3d = enlarge(transformed_points, transformed_center, scale_factor)

    return new_points_3d.tolist()


def get_points_3D(dst, indices):
    return [dst[i] for i in indices]

def dayan(DAYAN, dst_points):
    left_eye_points = get_points_3D(dst_points, L_EYE_INDICES)
    right_eye_points = get_points_3D(dst_points,  R_EYE_INDICES)

    l_point = get_points_3D(dst_points, [473])
    r_point = get_points_3D(dst_points, [468])

    l_new = calc_eyes_points(left_eye_points, DAYAN, l_point)

    r_new = calc_eyes_points(right_eye_points, DAYAN, r_point)
    all_new = l_new + r_new
    i = 0
    for idx in L_EYE_INDICES+R_EYE_INDICES:
        dst_points[idx] = (all_new[i][0], all_new[i][1], all_new[i][2])
        i += 1