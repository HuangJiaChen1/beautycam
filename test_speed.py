import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict


def get_triangles_and_coords(landmarks, image_width, image_height):
    coords = {i: (lm.x * image_width, lm.y * image_height) for i, lm in enumerate(landmarks)}

    mp_face_mesh = mp.solutions.face_mesh
    adj = defaultdict(set)
    for conn in mp_face_mesh.FACEMESH_TESSELATION:
        a, b = conn
        adj[a].add(b)
        adj[b].add(a)

    triangles = []
    for u in range(468):
        neighbors = [n for n in adj[u] if n > u]
        for i in range(len(neighbors)):
            v = neighbors[i]
            for j in range(i + 1, len(neighbors)):
                w = neighbors[j]
                if w in adj[v]:
                    triangles.append((u, v, w))

    return triangles, coords


def transform_image(image, src_pts, dst_pts):
    height, width = image.shape[:2]
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return image  # No face detected, return original

    landmarks = results.multi_face_landmarks[0].landmark
    triangles, coords = get_triangles_and_coords(landmarks, width, height)

    new_coords = coords.copy()
    for idx, pt in zip(src_pts, dst_pts):
        new_coords[idx] = tuple(pt)

    moved_set = set(src_pts)
    affected_tri = [tri for tri in triangles if moved_set & set(tri)]

    output = image.copy()

    # Erase old affected areas
    mask = np.zeros((height, width), np.uint8)
    for tri in affected_tri:
        pts = np.array([coords[p] for p in tri], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
    output[mask == 255] = 0  # Set to black; could be improved with inpainting

    # Helper functions
    def barycentric(p, tri_pts):
        a, b, c = tri_pts
        ax, ay = a
        bx, by = b
        cx, cy = c
        px, py = p
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-6:
            return -1, -1, -1
        l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
        l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
        l3 = 1 - l1 - l2
        return l1, l2, l3

    def interpolate(img, x, y):
        if x < 0 or y < 0 or x >= width - 0.001 or y >= height - 0.001:
            return np.array([0, 0, 0], dtype=np.uint8)
        x1, y1 = int(x), int(y)
        xa, ya = x - x1, y - y1
        x2 = min(x1 + 1, width - 1)
        y2 = min(y1 + 1, height - 1)
        val = (1 - xa) * (1 - ya) * img[y1, x1] + \
              xa * (1 - ya) * img[y1, x2] + \
              (1 - xa) * ya * img[y2, x1] + \
              xa * ya * img[y2, x2]
        return val.astype(np.uint8)

    # Warp affected triangles to new positions
    for tri in affected_tri:
        src_tri_pts = [coords[p] for p in tri]
        dst_tri_pts = [new_coords[p] for p in tri]
        dst_tri = np.array(dst_tri_pts, dtype=np.float32)

        # Get bounding box for dst triangle
        bbox_pts = dst_tri.reshape((-1, 1, 2)).astype(np.int32)
        x, y, w, h = cv2.boundingRect(bbox_pts)

        for py_i in range(max(0, y), min(height, y + h)):
            for px_i in range(max(0, x), min(width, x + w)):
                bar = barycentric((px_i, py_i), dst_tri_pts)
                if min(bar) >= -1e-6:
                    src_x = bar[0] * src_tri_pts[0][0] + bar[1] * src_tri_pts[1][0] + bar[2] * src_tri_pts[2][0]
                    src_y = bar[0] * src_tri_pts[0][1] + bar[1] * src_tri_pts[1][1] + bar[2] * src_tri_pts[2][1]
                    color = interpolate(image, src_x, src_y)
                    output[py_i, px_i] = color

    return output

# Example usage (commented out, as no specific image provided)
image = cv2.imread('imgs/faling.jpg')
src_pts = [374, 380, 390, 373, 249, 385, 384, 263, 466, 387, 386, 381, 382, 398, 388, 362,154, 155, 33, 7, 246, 161, 159, 158, 144, 145, 173, 133, 157, 163, 153, 160]
dst_pts = [[305.6, 212.60000000000002], [292.40000000000003, 210.2], [333.2, 212.60000000000002], [321.2, 212.60000000000002], [341.6, 211.39999999999998], [297.2, 183.8], [284.0, 191.0], [350.0, 207.8], [345.2, 201.8], [329.6, 188.60000000000002], [312.8, 183.8], [281.6, 209.0], [274.40000000000003, 207.8], [274.40000000000003, 200.60000000000002], [340.40000000000003, 195.8], [269.6, 206.60000000000002], [186.8, 217.6], [192.8, 216.4], [144.8, 214.0], [148.4, 217.6], [147.20000000000002, 209.2], [149.60000000000002, 203.2], [166.4, 193.6], [177.20000000000002, 196.0], [160.4, 221.20000000000002], [170.0, 221.20000000000002], [192.8, 209.2], [196.4, 214.0], [186.8, 202.0], [153.20000000000002, 219.99999999999997], [178.4, 219.99999999999997], [156.8, 197.2]]
print(len(src_pts))
print(len(dst_pts))
transformed = transform_image(image, src_pts, dst_pts)
cv2.imwrite('transformed.jpg', transformed)