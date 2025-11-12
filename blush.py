import cv2
import numpy as np


right_cheek_indices = [205, 207, 187, 123, 116, 117, 118, 50, 101, 36, 203, 206]
left_cheek_indices = [425, 427, 411, 352, 345, 346, 347, 280, 330, 266, 423, 426]

left_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
right_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
eye_indices = left_eye_indices + right_eye_indices
nose_indices = [193, 168, 417, 122, 351, 196, 419, 3, 248, 236, 456, 198, 420, 131, 360, 49, 279, 48, 278, 219, 439,
                59, 289, 218, 438, 237, 457, 44, 19, 274]
mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185,
                 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

def apply_blush_to_region(image, landmarks, indices, color, intensity):
    h, w = image.shape[:2]

    points = np.array([landmarks[idx] for idx in indices], dtype=np.int32)
    if len(points) < 3:
        return image

    center = points.mean(axis=0).astype(np.int32)
    center_x, center_y = int(center[0]), int(center[1])
    radius = int(np.max(np.sqrt(np.sum((points - center) ** 2, axis=1))) * 1.3)

    x1 = max(center_x - radius, 0)
    y1 = max(center_y - radius, 0)
    x2 = min(center_x + radius, w)
    y2 = min(center_y + radius, h)
    if x2 <= x1 or y2 <= y1:
        return image

    yy, xx = np.ogrid[y1:y2, x1:x2]
    dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    mask_local = np.clip(1 - (dist / max(radius, 1)), 0, 1)
    mask_local = (mask_local ** 2) * float(intensity)
    if radius %2 == 1:
        ksize = radius
    else:
        ksize = radius + 1
    mask_local = cv2.GaussianBlur(mask_local.astype(np.float32), (ksize, ksize), 0)

    blush_overlay = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.float32)
    blush_overlay[:] = color

    roi = image[y1:y2, x1:x2].astype(np.float32)
    mask3 = np.repeat(mask_local[..., None], 3, axis=2) * 0.6
    blush_effect = roi * (1 - mask3) + blush_overlay * mask3

    # Highlight
    h_radius = int(radius * 0.5)
    hc_y = center_y - int(radius * 0.15)
    yy2, xx2 = np.ogrid[y1:y2, x1:x2]
    dist_h = np.sqrt((xx2 - center_x) ** 2 + (yy2 - hc_y) ** 2)
    h_mask = np.clip(1 - (dist_h / max(h_radius, 1)), 0, 1)
    h_mask = (h_mask ** 2) * float(intensity) * 0.2
    ksize = int(ksize / 2)
    if ksize %2 == 1:
        pass
    else:
        ksize += 1
    h_mask = cv2.GaussianBlur(h_mask.astype(np.float32), (ksize, ksize), 0)
    blush_effect += np.repeat(h_mask[..., None], 3, axis=2) * 30

    out = image.copy().astype(np.float32)
    out[y1:y2, x1:x2] = blush_effect
    return np.clip(out, 0, 255).astype(np.uint8)

def create_region_mask(landmarks, indices, h, w):
    points = []
    for idx in indices:
        x,y = landmarks[idx]
        points.append([x, y])

    if len(points) < 3:
        return np.zeros((h, w), dtype=np.float32)

    points = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(points)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask.astype(np.float32) / 255.0

def apply_blush(image,landmarks, color=(157, 107, 255), intensity=0.6):
    output_image = image.copy()

    h, w = output_image.shape[:2]

    # Apply blush to right cheek
    output_image = apply_blush_to_region(
        output_image, landmarks, right_cheek_indices, color, intensity
    )

    # Apply blush to left cheek
    output_image = apply_blush_to_region(
        output_image, landmarks, left_cheek_indices, color, intensity
    )

    all_points = landmarks
    if len(all_points) >= 3:
        all_points = np.array(all_points, dtype=np.int32)
        face_mask_temp = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(all_points)
        cv2.fillConvexPoly(face_mask_temp, hull, 255)
        face_mask = face_mask_temp.astype(np.float32) / 255.0
    else:
        face_mask = np.ones((h, w), dtype=np.float32)

    eye_mask = create_region_mask(landmarks, eye_indices, h, w)
    nose_mask = create_region_mask(landmarks, nose_indices, h, w)
    mouth_mask = create_region_mask(landmarks, mouth_indices, h, w)

    forbidden_mask = np.maximum(1 - face_mask, np.maximum(eye_mask, np.maximum(nose_mask, mouth_mask)))

    forbidden_3ch = cv2.merge([forbidden_mask, forbidden_mask, forbidden_mask])
    # cv2.imshow('forbidden', forbidden_3ch)
    # cv2.waitKey(0)
    output_float = output_image.astype(np.float32)
    original_float = image.astype(np.float32)
    output_image = output_float * (1 - forbidden_3ch) + original_float * forbidden_3ch
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    return output_image



def main():
    """Demo usage of the FaceBlushEffect class"""

    # Load image
    image_path = "imgs/face_ce.jpg"  # Change this to your image path
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 480))
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print(f"Processing image: {image_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Apply different blush effects
    # Pink blush (default)
    result1 = apply_blush(image, color=(157, 107, 255), intensity=1)

    # Rose blush
    result2 = apply_blush(image, color=(193, 143, 255), intensity=1)

    # Coral blush
    result3 = apply_blush(image, color=(114, 128, 255), intensity=1)

    # Display results
    # Resize for display if image is too large
    max_display_width = 800
    if image.shape[1] > max_display_width:
        scale = max_display_width / image.shape[1]
        display_height = int(image.shape[0] * scale)

        original_display = cv2.resize(image, (max_display_width, display_height))
        result1_display = cv2.resize(result1, (max_display_width, display_height))
        result2_display = cv2.resize(result2, (max_display_width, display_height))
        result3_display = cv2.resize(result3, (max_display_width, display_height))
    else:
        original_display = image
        result1_display = result1
        result2_display = result2
        result3_display = result3

    # # Show images
    # cv2.imshow('Original', original_display)
    # cv2.imshow('Pink Blush', result1_display)
    # cv2.imshow('Rose Blush', result2_display)
    # cv2.imshow('Coral Blush', result3_display)
    #
    # # Save results
    # cv2.imwrite('output_pink_blush.jpg', result1)
    # cv2.imwrite('output_rose_blush.jpg', result2)
    # cv2.imwrite('output_coral_blush.jpg', result3)
    #
    # print("\nResults saved:")
    # print("- output_pink_blush.jpg")
    # print("- output_rose_blush.jpg")
    # print("- output_coral_blush.jpg")
    # print("\nPress any key to close windows...")
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
