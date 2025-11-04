import cv2
import mediapipe as mp
import numpy as np

from affine_transformation import warp_image_piecewise_affine

mp_face_mesh = mp.solutions.face_mesh

LIPS_INDICES = [61,185,40, 39, 37, 0, 267, 269,270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,78,191,80,81,
                82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95] # SORTED



def enlarge(cheek_points, midpoint, shrink_factor):
    diff = cheek_points - midpoint
    new_cheek_points = diff * shrink_factor + midpoint
    return new_cheek_points



def calc_points(lcp, scale_factor,center):
    transformed_points = np.array(lcp)
    transformed_center = np.array(center)
    new_points_3d = enlarge(transformed_points, transformed_center, scale_factor)

    return new_points_3d.tolist()

def calc_center(points):
    return np.mean(points, axis=0)

def get_points_3D(dst, indices):
    return [dst[i] for i in indices]

def dazui(DAZUI, dst_points):
    lip_points = get_points_3D(dst_points, LIPS_INDICES)
    center = calc_center(lip_points)
    # print(center)
    all_new = calc_points(lip_points,DAZUI,center)
    i = 0
    for idx in LIPS_INDICES:
        dst_points[idx] = (all_new[i][0], all_new[i][1], all_new[i][2])
        i += 1