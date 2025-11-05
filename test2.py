import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def inverse_transform_points(image_path, landmark_indices):
    """
    Apply inverse head pose transformation to facial landmarks.
    Uses 2D rotation approach based on head pose angles.

    Args:
        image_path: Path to the input image
        landmark_indices: List of landmark indices to transform

    Returns:
        Dictionary with original points, transformed points, and visualization
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        min_detection_confidence=0.5
    )

    # Read and process image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    img_h, img_w, img_c = image.shape

    if not results.multi_face_landmarks:
        raise ValueError("No face detected in image")

    face_landmarks = results.multi_face_landmarks[0]

    # Build arrays for pose estimation (using key points)
    face_2d_pose = []
    face_3d_pose = []

    for idx, lm in enumerate(face_landmarks.landmark):
        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d_pose.append([x, y])
            face_3d_pose.append([x, y, lm.z])

    face_2d_pose = np.array(face_2d_pose, dtype=np.float64)
    face_3d_pose = np.array(face_3d_pose, dtype=np.float64)

    # Camera matrix
    focal_length = 1 * img_w
    cam_matrix = np.array([
        [focal_length, 0, img_h / 2],
        [0, focal_length, img_w / 2],
        [0, 0, 1]
    ])
    distortion_matrix = np.zeros((4, 1), dtype=np.float64)

    # Solve PnP to get head pose
    success, rotation_vec, translation_vec = cv2.solvePnP(
        face_3d_pose, face_2d_pose, cam_matrix, distortion_matrix
    )

    # Get rotation matrix
    rmat, _ = cv2.Rodrigues(rotation_vec)

    # Calculate Euler angles
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    x_angle = angles[0] * 360
    y_angle = angles[1] * 360
    z_angle = angles[2] * 360

    print(f"Head Pose Angles - X: {x_angle:.2f}°, Y: {y_angle:.2f}°, Z: {z_angle:.2f}°")

    # Get face center (use nose tip as reference)
    nose_landmark = face_landmarks.landmark[1]
    face_center = np.array([nose_landmark.x * img_w, nose_landmark.y * img_h])

    print(f"Face center: {face_center}")

    # Extract ALL landmarks for full face transformation
    all_landmarks_2d = []
    for lm in face_landmarks.landmark:
        all_landmarks_2d.append([lm.x * img_w, lm.y * img_h])
    all_landmarks_2d = np.array(all_landmarks_2d)

    # Extract specified landmarks
    original_points_2d = []
    for idx in landmark_indices:
        if idx >= len(face_landmarks.landmark):
            print(f"Warning: Index {idx} out of range, skipping")
            continue
        lm = face_landmarks.landmark[idx]
        original_points_2d.append([lm.x * img_w, lm.y * img_h])

    original_points_2d = np.array(original_points_2d)

    # METHOD 1: Apply inverse 2D rotation based on head pose angles
    # Create 2D rotation matrix for in-plane rotation (z-axis)
    theta_z = -np.radians(z_angle)  # Negative for inverse
    rotation_2d_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z)],
        [np.sin(theta_z), np.cos(theta_z)]
    ])

    # Apply 2D rotation around face center
    centered_points = original_points_2d - face_center
    canonical_points_2d_simple = (rotation_2d_z @ centered_points.T).T + face_center

    # Do the same for all landmarks
    centered_all = all_landmarks_2d - face_center
    canonical_all_2d_simple = (rotation_2d_z @ centered_all.T).T + face_center

    # METHOD 2: Use full 3D rotation and project back
    # Build 3D points with relative z
    original_points_3d = []
    for idx in landmark_indices:
        lm = face_landmarks.landmark[idx]
        original_points_3d.append([lm.x * img_w, lm.y * img_h, lm.z])
    original_points_3d = np.array(original_points_3d)

    all_landmarks_3d = []
    for lm in face_landmarks.landmark:
        all_landmarks_3d.append([lm.x * img_w, lm.y * img_h, lm.z])
    all_landmarks_3d = np.array(all_landmarks_3d)

    # Apply inverse 3D rotation
    rmat_inv = rmat.T

    # For proper transformation, we need to:
    # 1. Transform points to camera coordinate system
    # 2. Apply inverse rotation
    # 3. Project back to image plane

    # Convert image points to camera coordinates (homogeneous)
    def to_camera_coords(pts_2d, depth_values):
        """Convert 2D image points with depth to 3D camera coordinates"""
        pts_homogeneous = np.column_stack([pts_2d, np.ones(len(pts_2d))])
        # Assume depth is proportional to relative z
        cam_coords = []
        for i, pt_h in enumerate(pts_homogeneous):
            # Back-project using camera matrix
            depth = 1000 + depth_values[i] * 100  # Scale depth appropriately
            x_cam = (pt_h[0] - cam_matrix[0, 2]) * depth / cam_matrix[0, 0]
            y_cam = (pt_h[1] - cam_matrix[1, 2]) * depth / cam_matrix[1, 1]
            z_cam = depth
            cam_coords.append([x_cam, y_cam, z_cam])
        return np.array(cam_coords)

    # Get camera coordinates
    cam_coords_selected = to_camera_coords(original_points_2d, original_points_3d[:, 2])
    cam_coords_all = to_camera_coords(all_landmarks_2d, all_landmarks_3d[:, 2])

    # Apply inverse rotation in 3D camera space
    canonical_cam_selected = (rmat_inv @ cam_coords_selected.T).T
    canonical_cam_all = (rmat_inv @ cam_coords_all.T).T

    # Project back to 2D
    def project_to_2d(cam_coords):
        """Project 3D camera coordinates back to 2D image plane"""
        pts_2d = []
        for pt in cam_coords:
            x_img = cam_matrix[0, 0] * pt[0] / pt[2] + cam_matrix[0, 2]
            y_img = cam_matrix[1, 1] * pt[1] / pt[2] + cam_matrix[1, 2]
            pts_2d.append([x_img, y_img])
        return np.array(pts_2d)

    canonical_points_2d_3d = project_to_2d(canonical_cam_selected)
    canonical_all_2d_3d = project_to_2d(canonical_cam_all)

    print(f"\nTransforming {len(landmark_indices)} landmarks to canonical pose")
    print(f"Original 2D points shape: {original_points_2d.shape}")
    print(f"Canonical 2D (simple) shape: {canonical_points_2d_simple.shape}")
    print(f"Canonical 2D (3D method) shape: {canonical_points_2d_3d.shape}")

    # Visualize
    visualize_transformation(
        original_points_2d,
        canonical_points_2d_simple,
        canonical_points_2d_3d,
        all_landmarks_2d,
        canonical_all_2d_simple,
        canonical_all_2d_3d,
        landmark_indices,
        x_angle, y_angle, z_angle,
        image, img_w, img_h,
        face_center
    )

    face_mesh.close()

    return {
        'original_points_2d': original_points_2d,
        'canonical_points_2d_simple': canonical_points_2d_simple,
        'canonical_points_2d_3d': canonical_points_2d_3d,
        'all_landmarks_original_2d': all_landmarks_2d,
        'all_landmarks_canonical_2d_simple': canonical_all_2d_simple,
        'all_landmarks_canonical_2d_3d': canonical_all_2d_3d,
        'rotation_matrix': rmat,
        'rotation_matrix_inv': rmat_inv,
        'angles': {'x': x_angle, 'y': y_angle, 'z': z_angle},
        'landmark_indices': landmark_indices,
        'face_center': face_center
    }


def visualize_transformation(original_pts_2d, canonical_pts_2d_simple, canonical_pts_2d_3d,
                             all_original_2d, all_canonical_2d_simple, all_canonical_2d_3d,
                             indices, x_ang, y_ang, z_ang,
                             image, img_w, img_h, face_center):
    """Create comprehensive visualization of the transformation"""

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 10))

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Row 1: 2D visualizations - Simple 2D rotation method
    # 1. Original image with highlighted points
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(image_rgb)
    ax1.set_title(f'Original Image\nPose: X={x_ang:.1f}°, Y={y_ang:.1f}°, Z={z_ang:.1f}°',
                  fontsize=12, fontweight='bold')

    # Draw all landmarks lightly
    ax1.scatter(all_original_2d[:, 0], all_original_2d[:, 1],
                c='cyan', s=1, alpha=0.3)

    # Highlight selected landmarks
    ax1.scatter(original_pts_2d[:, 0], original_pts_2d[:, 1],
                c='red', s=100, edgecolors='white', linewidths=2,
                label='Selected Points', zorder=5)

    # Mark face center
    ax1.scatter(face_center[0], face_center[1], c='yellow', s=150,
                marker='x', linewidths=3, label='Face Center', zorder=5)

    # Add index labels
    for i, (x, y) in enumerate(original_pts_2d):
        ax1.text(x, y - 15, str(indices[i]), fontsize=8,
                 ha='center', color='yellow', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))

    ax1.set_xlim(0, img_w)
    ax1.set_ylim(img_h, 0)
    ax1.axis('off')
    ax1.legend(loc='upper right')

    # 2. Canonical 2D (Simple 2D rotation method)
    ax2 = fig.add_subplot(2, 3, 2)
    canvas = np.ones((img_h, img_w, 3), dtype=np.uint8) * 240
    ax2.imshow(canvas)
    ax2.set_title('Canonical 2D - Simple Method\n(2D rotation around face center)',
                  fontsize=12, fontweight='bold')

    # Draw ALL canonical landmarks
    ax2.scatter(all_canonical_2d_simple[:, 0], all_canonical_2d_simple[:, 1],
                c='lightblue', s=1, alpha=0.5)

    # Plot selected canonical points
    ax2.scatter(canonical_pts_2d_simple[:, 0], canonical_pts_2d_simple[:, 1],
                c='blue', s=100, edgecolors='white', linewidths=2,
                label='Canonical Points', zorder=5)

    # Mark face center
    ax2.scatter(face_center[0], face_center[1], c='yellow', s=150,
                marker='x', linewidths=3, label='Face Center', zorder=5)

    # Add index labels
    for i, (x, y) in enumerate(canonical_pts_2d_simple):
        ax2.text(x, y - 15, str(indices[i]), fontsize=8,
                 ha='center', color='yellow', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7))

    ax2.set_xlim(0, img_w)
    ax2.set_ylim(img_h, 0)
    ax2.axis('off')
    ax2.legend(loc='upper right')

    # 3. Overlay comparison - Simple method
    ax3 = fig.add_subplot(2, 3, 3)
    comparison = np.ones((img_h, img_w, 3), dtype=np.uint8) * 240
    ax3.imshow(comparison)
    ax3.set_title('Overlay - Simple Method\nRed=Original, Blue=Canonical',
                  fontsize=12, fontweight='bold')

    ax3.scatter(original_pts_2d[:, 0], original_pts_2d[:, 1],
                c='red', s=80, alpha=0.7, edgecolors='darkred', linewidths=2,
                label='Original', zorder=5)
    ax3.scatter(canonical_pts_2d_simple[:, 0], canonical_pts_2d_simple[:, 1],
                c='blue', s=80, alpha=0.7, edgecolors='darkblue', linewidths=2,
                label='Canonical', zorder=5)

    # Draw arrows
    for i in range(len(original_pts_2d)):
        ax3.annotate('', xy=canonical_pts_2d_simple[i], xytext=original_pts_2d[i],
                     arrowprops=dict(arrowstyle='->', color='green', lw=1.5, alpha=0.6))

    ax3.scatter(face_center[0], face_center[1], c='yellow', s=150,
                marker='x', linewidths=3, zorder=6)

    ax3.set_xlim(0, img_w)
    ax3.set_ylim(img_h, 0)
    ax3.axis('off')
    ax3.legend(loc='upper right')

    # Row 2: 3D rotation method
    # 4. Canonical 2D (3D rotation method)
    ax4 = fig.add_subplot(2, 3, 4)
    canvas2 = np.ones((img_h, img_w, 3), dtype=np.uint8) * 240
    ax4.imshow(canvas2)
    ax4.set_title('Canonical 2D - 3D Method\n(Full 3D rotation + reprojection)',
                  fontsize=12, fontweight='bold')

    # Draw ALL canonical landmarks
    valid = (all_canonical_2d_3d[:, 0] >= -img_w) & (all_canonical_2d_3d[:, 0] < 2 * img_w) & \
            (all_canonical_2d_3d[:, 1] >= -img_h) & (all_canonical_2d_3d[:, 1] < 2 * img_h)
    ax4.scatter(all_canonical_2d_3d[valid, 0], all_canonical_2d_3d[valid, 1],
                c='lightgreen', s=1, alpha=0.5)

    # Plot selected canonical points
    ax4.scatter(canonical_pts_2d_3d[:, 0], canonical_pts_2d_3d[:, 1],
                c='green', s=100, edgecolors='white', linewidths=2,
                label='Canonical Points', zorder=5)

    # Mark face center
    ax4.scatter(face_center[0], face_center[1], c='yellow', s=150,
                marker='x', linewidths=3, label='Face Center', zorder=5)

    # Add index labels
    for i, (x, y) in enumerate(canonical_pts_2d_3d):
        if -img_w < x < 2 * img_w and -img_h < y < 2 * img_h:
            ax4.text(x, y - 15, str(indices[i]), fontsize=8,
                     ha='center', color='yellow', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7))

    ax4.set_xlim(0, img_w)
    ax4.set_ylim(img_h, 0)
    ax4.axis('off')
    ax4.legend(loc='upper right')

    # 5. Overlay comparison - 3D method
    ax5 = fig.add_subplot(2, 3, 5)
    comparison2 = np.ones((img_h, img_w, 3), dtype=np.uint8) * 240
    ax5.imshow(comparison2)
    ax5.set_title('Overlay - 3D Method\nRed=Original, Green=Canonical',
                  fontsize=12, fontweight='bold')

    ax5.scatter(original_pts_2d[:, 0], original_pts_2d[:, 1],
                c='red', s=80, alpha=0.7, edgecolors='darkred', linewidths=2,
                label='Original', zorder=5)
    ax5.scatter(canonical_pts_2d_3d[:, 0], canonical_pts_2d_3d[:, 1],
                c='green', s=80, alpha=0.7, edgecolors='darkgreen', linewidths=2,
                label='Canonical (3D)', zorder=5)

    # Draw arrows
    for i in range(len(original_pts_2d)):
        if (-img_w < canonical_pts_2d_3d[i, 0] < 2 * img_w and
                -img_h < canonical_pts_2d_3d[i, 1] < 2 * img_h):
            ax5.annotate('', xy=canonical_pts_2d_3d[i], xytext=original_pts_2d[i],
                         arrowprops=dict(arrowstyle='->', color='orange', lw=1.5, alpha=0.6))

    ax5.scatter(face_center[0], face_center[1], c='yellow', s=150,
                marker='x', linewidths=3, zorder=6)

    ax5.set_xlim(0, img_w)
    ax5.set_ylim(img_h, 0)
    ax5.axis('off')
    ax5.legend(loc='upper right')

    # 6. Comparison of both methods
    ax6 = fig.add_subplot(2, 3, 6)
    comparison3 = np.ones((img_h, img_w, 3), dtype=np.uint8) * 240
    ax6.imshow(comparison3)
    ax6.set_title('Method Comparison\nBlue=Simple, Green=3D',
                  fontsize=12, fontweight='bold')

    ax6.scatter(canonical_pts_2d_simple[:, 0], canonical_pts_2d_simple[:, 1],
                c='blue', s=80, alpha=0.6, edgecolors='darkblue', linewidths=2,
                label='Simple (2D rotation)', zorder=5)
    ax6.scatter(canonical_pts_2d_3d[:, 0], canonical_pts_2d_3d[:, 1],
                c='green', s=80, alpha=0.6, edgecolors='darkgreen', linewidths=2,
                label='3D method', zorder=5, marker='^')

    ax6.scatter(face_center[0], face_center[1], c='yellow', s=150,
                marker='x', linewidths=3, zorder=6)

    ax6.set_xlim(0, img_w)
    ax6.set_ylim(img_h, 0)
    ax6.axis('off')
    ax6.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('inverse_transform_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nVisualization saved as 'inverse_transform_visualization.png'")


# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "imgs/face_celian.jpg"

    # Specify landmark indices to transform
    landmark_indices = [33, 263, 1, 61, 291, 199, 130, 359, 10, 152]

    try:
        result = inverse_transform_points(image_path, landmark_indices)

        print("\n" + "=" * 50)
        print("TRANSFORMATION RESULTS")
        print("=" * 50)
        print(f"\nNumber of points transformed: {len(result['landmark_indices'])}")
        print(f"\nHead Pose Angles:")
        print(f"  X (pitch): {result['angles']['x']:.2f}°")
        print(f"  Y (yaw): {result['angles']['y']:.2f}°")
        print(f"  Z (roll): {result['angles']['z']:.2f}°")
        print(f"\nFace Center: {result['face_center']}")

        # Show sample points for both methods
        print(f"\nSample transformations (first 3 points):")
        for i in range(min(3, len(result['original_points_2d']))):
            print(f"\nLandmark {result['landmark_indices'][i]}:")
            print(f"  Original 2D:        {result['original_points_2d'][i]}")
            print(f"  Canonical 2D (2D):  {result['canonical_points_2d_simple'][i]}")
            print(f"  Canonical 2D (3D):  {result['canonical_points_2d_3d'][i]}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()