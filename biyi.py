import cv2
import mediapipe as mp
import numpy as np
from affine_transformation import warp_image_piecewise_affine

left_nose = [126,129,98,75,166,79,239,238,20,60,48,115,220,45]
right_nose = [275,440,344,278,355,358,327,289,392,309,459,458,250,290]


mp_face_mesh = mp.solutions.face_mesh


def calculate_midpoint(lcp,rcp):
    midpoint = (lcp+rcp) / 2.0
    return midpoint



def shrink_points_towards_midpoint(nose_points, midpoint, shrink_factor):
    diff = nose_points - midpoint
    new_nose_points = diff * shrink_factor + midpoint
    return new_nose_points


def calc_slim_points(lnp, rnp, scale_factor):
    left_nose_points = np.array(lnp)
    right_nose_points = np.array(rnp)

    center = calculate_midpoint(left_nose_points,right_nose_points)
    new_points_3d_l = shrink_points_towards_midpoint(left_nose_points, center, scale_factor)
    new_points_3d_r = shrink_points_towards_midpoint(right_nose_points, center, scale_factor)
    new_points_3d = np.concatenate((new_points_3d_l,new_points_3d_r),axis=0)
    return new_points_3d.tolist()

def get_points_3D(dst, indices):
    return [dst[i] for i in indices]

def biyi(BIYI, dst_points):
    left_nose_points = get_points_3D(dst_points, left_nose)
    right_nose_points = get_points_3D(dst_points, right_nose)


    new_nose_points = calc_slim_points(left_nose_points, right_nose_points, BIYI)
    i = 0
    for idx in left_nose + right_nose:
        dst_points[idx] = (new_nose_points[i][0], new_nose_points[i][1], new_nose_points[i][2])
        i += 1
