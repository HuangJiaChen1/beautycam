import cv2
import numpy as np
from ultralytics import SAM

# Load MobileSAM once (auto-downloads if missing)
model = SAM("mobile_sam.pt")


def get_segmentation_mask(image, click_xy,negatives):
    pts = [click_xy]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(click_xy)
    print(negatives)
    print(pts)
    results = model.predict(
        image,
        points= pts,
        labels=[1],
        verbose=False
    )

    if len(results) == 0 or results[0].masks is None:
        height, width = image.shape[:2]
        return np.zeros((height, width), dtype=np.uint8)

    # Extract mask (first/best one)
    mask_tensor = results[0].masks.data[0]  # Shape: (H, W)
    mask = ((mask_tensor.cpu().numpy() > 0.0)).astype(np.uint8)  # Binary: 1=foreground, 0=background
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    return mask

#
# # Example usage: Load image, click a point, get mask
# if __name__ == "__main__":
#     # Load your image (replace path)
#     image_path = "imgs/2face.jpg"  # Update this!
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Error: Can't load {image_path}")
#         exit(1)
#
#     # Example click point (replace with your (x, y))
#     click_point = (569, 367)  # From your traceback example
#
#     # Get the mask
#     mask = get_segmentation_mask(image, click_point)
#
#     # Optional: Visualize
#     segmented = image.copy()
#     segmented[mask == 0] = 0  # Black out non-masked areas
#     cv2.imwrite("segmented_image.jpg", segmented)
#     cv2.imwrite("mask.png", mask * 255)  # Save mask as grayscale
#
#     print(f"Mask shape: {mask.shape}")
#     print(f"Non-zero pixels in mask: {np.count_nonzero(mask)}")
#     print("Saved 'segmented_image.jpg' and 'mask.png'")