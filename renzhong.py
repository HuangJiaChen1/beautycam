import cv2
import mediapipe as mp
import numpy as np
from affine_transformation import warp_image_piecewise_affine

LIPS_INDICES = [61,185,40, 39, 37, 0, 267, 269,270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95] # SORTED
mid_jaw = 152
nose_tip = 4

mp_face_mesh = mp.solutions.face_mesh


def move_points_along_line(p1, p2, points, s=1.0, normalize=True):
    p1 = np.array(p1)
    p2 = np.array(p2)
    points = np.array(points)
    d = p2 - p1

    if np.allclose(d, 0):
        raise ValueError("p1 and p2 are the same point; no direction defined.")

    if normalize:
        u = d / np.linalg.norm(d)
        movement = s * u
    else:
        movement = s * d

    moved_points = points + movement

    return moved_points


def calc_lips_points(lp,mjp,ntp, scale_factor):
    new_points_3d = move_points_along_line(ntp, mjp,lp, scale_factor)

    return new_points_3d.tolist()


def get_points_3D(dst, indices):
    return [dst[i] for i in indices]

def renzhong(RENZHONG,dst_points):
    lips_points = get_points_3D(dst_points, LIPS_INDICES)
    mid_jaw_point = get_points_3D(dst_points,[mid_jaw])
    nose_tip_point = get_points_3D(dst_points,[nose_tip])


    new_lips_points = calc_lips_points(lips_points, mid_jaw_point,nose_tip_point, RENZHONG)
    i = 0
    for idx in LIPS_INDICES:
        dst_points[idx] = (new_lips_points[i][0], new_lips_points[i][1], new_lips_points[i][2])
        i += 1