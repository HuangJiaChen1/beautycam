import cv2
import numpy as np

from scipy.spatial import Delaunay




def delaunay_triangulation(points, w, h):
    points = np.array(points)
    points = np.vstack([points, [[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]])
    tri = Delaunay(points)
    return tri.simplices

def clamp_rect(rect, max_w, max_h):
    x, y, w, h = rect
    x1 = min(max(x, 0), max_w)
    y1 = min(max(y, 0), max_h)
    x2 = min(x + w, max_w)
    y2 = min(y + h, max_h)
    return [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]


def warp_triangle(src, dst, t_src, t_dst):
    src_h, src_w = src.shape[:2]
    dst_h, dst_w = dst.shape[:2]

    r1 = list(cv2.boundingRect(t_src))
    r2 = list(cv2.boundingRect(t_dst))

    clamped_r1 = clamp_rect(r1, src_w, src_h)
    clamped_r2 = clamp_rect(r2, dst_w, dst_h)

    if clamped_r1[2] == 0 or clamped_r1[3] == 0 or clamped_r2[2] == 0 or clamped_r2[3] == 0:
        return

    t1Rect = np.array([[t_src[0][0]-clamped_r1[0], t_src[0][1]-clamped_r1[1]],
                       [t_src[1][0]-clamped_r1[0], t_src[1][1]-clamped_r1[1]],
                       [t_src[2][0]-clamped_r1[0], t_src[2][1]-clamped_r1[1]]], dtype=np.float32)

    t2Rect = np.array([[t_dst[0][0]-clamped_r2[0], t_dst[0][1]-clamped_r2[1]],
                       [t_dst[1][0]-clamped_r2[0], t_dst[1][1]-clamped_r2[1]],
                       [t_dst[2][0]-clamped_r2[0], t_dst[2][1]-clamped_r2[1]]], dtype=np.float32)

    M = cv2.getAffineTransform(t1Rect, t2Rect)

    srcROI = src[clamped_r1[1]:clamped_r1[1]+clamped_r1[3], clamped_r1[0]:clamped_r1[0]+clamped_r1[2]]
    warped = cv2.warpAffine(srcROI, M, (clamped_r2[2], clamped_r2[3]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    mask = np.zeros((clamped_r2[3], clamped_r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2Rect), (1.0, 1.0, 1.0), cv2.LINE_AA)
    dstROI = dst[clamped_r2[1]:clamped_r2[1]+clamped_r2[3], clamped_r2[0]:clamped_r2[0]+clamped_r2[2]]
    dstROI[:] = (dstROI * (1 - mask) + warped * mask).astype(dstROI.dtype)

def warp_image_piecewise_affine(src_img, src_pts, dst_pts, triangles=None, strength=1.0):
    h, w = src_img.shape[:2]

    src_pts = np.asarray(src_pts, dtype=np.float32)
    dst_pts = np.asarray(dst_pts, dtype=np.float32)
    blended_dst = (1.0 - strength)*src_pts + strength*dst_pts

    if triangles is None:
        mean_shape = 0.5*(src_pts + blended_dst)
        triangles = delaunay_triangulation(mean_shape, w, h)
        if triangles.size == 0:
            raise RuntimeError("Delaunay triangulation failed; check landmark coverage and image bounds.")

    out = src_img.copy()

    for tri in triangles:
        t_src = src_pts[tri].astype(np.float32)
        t_dst = blended_dst[tri].astype(np.float32)

        def area(tri3):
            return 0.5*np.linalg.det(np.array([
                [tri3[1,0]-tri3[0,0], tri3[1,1]-tri3[0,1]],
                [tri3[2,0]-tri3[0,0], tri3[2,1]-tri3[0,1]],
            ]))
        # if abs(area(t_src)) < 1e-3 or abs(area(t_dst)) < 1e-3:
        #     continue

        warp_triangle(src_img, out, t_src, t_dst)

    return out

def project_to_2d(point_3d):
    x, y, z = point_3d
    if z != 0:
        return (x / z, y / z)
    return (x, y)

def warp_with_triangulation(image, source_points, scaled_points):
    H, W = image.shape[:2]
    border_pts = np.array(
        [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1], [W // 2, 0], [W - 1, H // 2], [W // 2, H - 1], [0, H // 2]],
        np.float32)
    s_list = []
    d_list = []
    for k,v in source_points.items():
        s_list.append(project_to_2d(v))
    for k,v in scaled_points.items():
        d_list.append(project_to_2d(v))
    src_all = np.vstack([s_list, border_pts])
    dst_all = np.vstack([d_list, border_pts])
    # print(src_all.shape)
    # print(dst_all.shape)
    warped_eye = warp_image_piecewise_affine(image, src_all, dst_all)

    return warped_eye

def warp_with_triangulation_2d(image, source_points, scaled_points):
    H, W = image.shape[:2]
    border_pts = np.array(
        [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1], [W // 2, 0], [W - 1, H // 2], [W // 2, H - 1], [0, H // 2]],
        np.float32)
    s_list = []
    d_list = []
    for k,v in source_points.items():
        s_list.append(v)
    for k,v in scaled_points.items():
        d_list.append(v)
    src_all = np.vstack([s_list, border_pts])
    dst_all = np.vstack([d_list, border_pts])
    # print(src_all.shape)
    # print(dst_all.shape)
    warped_eye = warp_image_piecewise_affine(image, src_all, dst_all)

    return warped_eye
