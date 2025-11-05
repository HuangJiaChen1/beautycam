# --------------------------------------------------------------
#  visualize_facemesh_3d_pycharm.py   (PyCharm-ready)
# --------------------------------------------------------------
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------------------------------
# 1. Force interactive backend (must be BEFORE any plt.*)
# ------------------------------------------------------------------
plt.switch_backend('TkAgg')          # works in PyCharm console / SciView

# ------------------------------------------------------------------
# 2. MediaPipe setup
# ------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)

# ------------------------------------------------------------------
# 3. Load image
# ------------------------------------------------------------------
IMG_PATH = "imgs/face_celian.jpg"
img_bgr = cv2.imread(IMG_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Cannot open {IMG_PATH}")

h, w, _ = img_bgr.shape
rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ------------------------------------------------------------------
# 4. Run Face Mesh
# ------------------------------------------------------------------
results = face_mesh.process(rgb)
if not results.multi_face_landmarks:
    raise RuntimeError("No face detected!")

landmarks = results.multi_face_landmarks[0].landmark   # 468 points

# ------------------------------------------------------------------
# 5. 2-D points (for overlay) + 3-D points (YOUR way)
# ------------------------------------------------------------------
pts_2d = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

# ---- YOUR custom 3-D extraction -----------------------------------
def get_points_all(image, landmarks):
    points = [(landmarks[i].x, landmarks[i].y, landmarks[i].z + 1)
              for i in range(len(landmarks))]
    points_pixels = [transform_to_3d(int(p[0] * image.shape[1]),
                                   int(p[1] * image.shape[0]),
                                   p[2]) for p in points]
    return np.array(points_pixels, dtype=np.float32)

def transform_to_3d(x, y, z):
    return np.array([x * z, y * z, z*1000], dtype=np.float32)

pts_3d = get_points_all(img_bgr, landmarks)      # <-- your 3-D array
print(pts_3d)

# ------------------------------------------------------------------
# 6. Visibility filtering based on outer hull max_z
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 6. Visibility filtering based on outer hull max_z (ROBUST)
# ------------------------------------------------------------------
from scipy.spatial import KDTree

tree = KDTree(pts_2d)
hull = cv2.convexHull(pts_2d)
hull_points = hull.reshape(-1, 2)

hull_indices = []
for hp in hull_points:
    dist, idx = tree.query(hp, k=1, distance_upper_bound=2.0)
    if dist != np.inf:
        hull_indices.append(idx)

hull_indices = np.unique(hull_indices)
if len(hull_indices) == 0:
    raise RuntimeError("No hull points matched. Check landmarks.")

hull_z = pts_3d[hull_indices, 2]
max_z = np.max(hull_z)
visible = pts_3d[:, 2] <= max_z

pts_3d_visible = pts_3d[visible]
pts_2d_visible = pts_2d[visible]

CONNECTIONS = mp_face_mesh.FACEMESH_TESSELATION
visible_connections = [(a, b) for a, b in CONNECTIONS if visible[a] and visible[b]]

orig_to_new = {old: new for new, old in enumerate(np.where(visible)[0])}

# ------------------------------------------------------------------
# 7. 2-D overlay (only visible)
# ------------------------------------------------------------------
img_vis = img_bgr.copy()
for (x, y) in pts_2d_visible.astype(int):
    cv2.circle(img_vis, (x, y), 1, (0, 255, 0), -1)

for a, b in visible_connections:
    cv2.line(img_vis,
             pts_2d[a].astype(int),
             pts_2d[b].astype(int),
             (255, 150, 0), 1)

cv2.imshow("2-D overlay (visible only) – press any key", img_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------------------------------------------------
# 8. 3-D interactive plot (only visible)
# ------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pts_3d_visible[:, 0], pts_3d_visible[:, 1], pts_3d_visible[:, 2],
           c='cyan', s=8, depthshade=False)

# Remap connections to visible indices
for a, b in visible_connections:
    new_a = orig_to_new[a]
    new_b = orig_to_new[b]
    ax.plot(pts_3d_visible[[new_a, new_b], 0],
            pts_3d_visible[[new_a, new_b], 1],
            pts_3d_visible[[new_a, new_b], 2],
            color='orange', linewidth=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (scaled by depth)')
ax.set_title('MediaPipe Face Mesh – 3-D (visible only, drag to rotate)')
ax.view_init(elev=20, azim=-70)
ax.set_box_aspect([np.ptp(pts_3d_visible[:,0]), np.ptp(pts_3d_visible[:,1]), np.ptp(pts_3d_visible[:,2])])

plt.tight_layout()
print("\n=== 3-D plot ready (visible landmarks only) ===")
print("  • LEFT-CLICK + DRAG  → rotate")
print("  • SCROLL             → zoom")
print("  • MIDDLE-CLICK + DRAG→ pan")
print("  • Press 'r'          → reset view\n")
plt.show()