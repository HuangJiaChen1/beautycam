import cv2
import numpy as np
import mediapipe as mp

ZUIJIAO_INDICES = [61,291]
C_LEFT = 50
C_RIGHT = 280

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

def zuijiao(ZUIJIAO, dst_points):
    nose_points = get_points_3D(dst_points, ZUIJIAO_INDICES)
    p1 = get_points_3D(dst_points, [C_LEFT])
    p2 = get_points_3D(dst_points, [C_RIGHT])
    new_mouth_point_left = calc_new_points(nose_points[0],nose_points[0],p1, ZUIJIAO)
    new_mouth_point_right = calc_new_points(nose_points[1], nose_points[1], p2, ZUIJIAO)
    new_mouth_point = np.concatenate((new_mouth_point_left,new_mouth_point_right), axis=0)
    i = 0
    for idx in ZUIJIAO_INDICES:
        dst_points[idx] = (new_mouth_point[i][0], new_mouth_point[i][1], new_mouth_point[i][2])
        i += 1
