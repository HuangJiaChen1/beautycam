import cv2
import mediapipe as mp
import numpy as np
from affine_transformation import warp_image_piecewise_affine

left_cheek = [116,123,50]
right_cheek = [345,352,280]


mp_face_mesh = mp.solutions.face_mesh


def calculate_midpoint(lcp,rcp):
    midpoint = (lcp+rcp) / 2.0
    return midpoint



def shrink_points_towards_midpoint(cheek_points, midpoint, shrink_factor):
    diff = cheek_points - midpoint
    new_cheek_points = diff * shrink_factor + midpoint
    return new_cheek_points


def calc_slim_points(lcp, rcp, scale_factor):
    left_cheek_points = np.array(lcp)
    right_cheek_points = np.array(rcp)

    center = calculate_midpoint(left_cheek_points,right_cheek_points)
    # print(transformed_points)
    new_points_3d_l = shrink_points_towards_midpoint(left_cheek_points, center, scale_factor)
    new_points_3d_r = shrink_points_towards_midpoint(right_cheek_points, center, scale_factor)
    new_points_3d = np.concatenate((new_points_3d_l,new_points_3d_r),axis=0)
    print(new_points_3d)
    return new_points_3d.tolist()

def get_points_3D(dst, indices):
    return [dst[i] for i in indices]

def quangu(QUANGU, dst_points):
    left_cheek_points = get_points_3D(dst_points, left_cheek)
    right_cheek_points = get_points_3D(dst_points, right_cheek)


    new_cheek_points = calc_slim_points(left_cheek_points, right_cheek_points, QUANGU)
    i = 0
    for idx in left_cheek + right_cheek:
        dst_points[idx] = (new_cheek_points[i][0], new_cheek_points[i][1], new_cheek_points[i][2])
        i += 1