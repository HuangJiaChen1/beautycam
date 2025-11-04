import cv2
import mediapipe as mp
import numpy as np
from affine_transformation import warp_image_piecewise_affine

FOREHEAD_INDICES = [103,67,109,10,338,297,332]
mid_jaw = 152
head_tip = 10

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
        movement = -s * u
    else:
        movement = -s * d

    moved_points = points + movement

    return moved_points


def calc_forehead_points(lp,mjp,htp, scale_factor):
    new_points_3d = move_points_along_line(htp, mjp,lp, scale_factor)

    return new_points_3d.tolist()


def get_points_3D(dst, indices):
    return [dst[i] for i in indices]

def forehead(FOREHEAD,dst_points):
    forehead_points = get_points_3D(dst_points, FOREHEAD_INDICES)
    mid_jaw_point = get_points_3D(dst_points,[mid_jaw])
    head_tip_point = get_points_3D(dst_points,[head_tip])


    new_forehead_points = calc_forehead_points(forehead_points, mid_jaw_point,head_tip_point, FOREHEAD)
    i = 0
    for idx in FOREHEAD_INDICES:
        dst_points[idx] = (new_forehead_points[i][0], new_forehead_points[i][1], new_forehead_points[i][2])
        i += 1