import cv2
import numpy as np

from scipy.spatial import Delaunay




def delaunay_triangulation(points, w, h):
    points = np.array(points)
    points = np.vstack([points, [[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]])
    tri = Delaunay(points)
    return tri.simplices

def warp_triangle(src, dst, t_src, t_dst):
    r1 = cv2.boundingRect(t_src)
    r2 = cv2.boundingRect(t_dst)

    t1Rect = np.array([[t_src[0][0]-r1[0], t_src[0][1]-r1[1]],
                       [t_src[1][0]-r1[0], t_src[1][1]-r1[1]],
                       [t_src[2][0]-r1[0], t_src[2][1]-r1[1]]], dtype=np.float32)

    t2Rect = np.array([[t_dst[0][0]-r2[0], t_dst[0][1]-r2[1]],
                       [t_dst[1][0]-r2[0], t_dst[1][1]-r2[1]],
                       [t_dst[2][0]-r2[0], t_dst[2][1]-r2[1]]], dtype=np.float32)

    M = cv2.getAffineTransform(t1Rect, t2Rect)

    srcROI = src[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    warped = cv2.warpAffine(srcROI, M, (r2[2], r2[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2Rect), (1.0, 1.0, 1.0), cv2.LINE_AA)
    # print(r1)
    # print(r2)
    dstROI = dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    # print(f"mask shape: {mask.shape}")
    # print(f"dstROI shape: {dstROI.shape}")
    # print(f"warped shape: {warped.shape}")
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
        if abs(area(t_src)) < 1e-3 or abs(area(t_dst)) < 1e-3:
            continue

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