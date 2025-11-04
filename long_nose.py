import cv2
import mediapipe as mp
import numpy as np
from affine_transformation import warp_image_piecewise_affine

nose_bottom = [126,129,98,75,166,79,239,238,20,60,48,115,220,45,4,275,440,344,278,355,358,327,289,392,309,459,458,250,290,2]
nose_bot = 10
nose_top = 2

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

def calc_new_points(np, p1,p2, scale_factor):
    new_points_3d = move_points_along_line(p1, p2,np, scale_factor)

    return new_points_3d.tolist()

def get_points_3D(dst, indices):
    return [dst[i] for i in indices]

def longnose(LONGNOSE, dst_points):
    nose_points = get_points_3D(dst_points, nose_bottom)
    p1 = get_points_3D(dst_points, [nose_bot])
    p2 = get_points_3D(dst_points, [nose_top])
    new_nose_points = calc_new_points(nose_points,p1,p2, LONGNOSE)
    i = 0
    for idx in nose_bottom:
        dst_points[idx] = (new_nose_points[i][0], new_nose_points[i][1], new_nose_points[i][2])
        i += 1
