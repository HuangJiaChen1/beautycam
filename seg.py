import cv2
import numpy as np


def canny_edge_detection(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 50, 50)  # Increase the thresholds

    # Apply morphological closing to the edges to fill small gaps
    kernel = np.ones((5, 5), np.uint8)  # Kernel size for closing
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("edges", closed_edges)
    return closed_edges


def is_adjacent_to_edge(y, x, edges, max_diff=1):
    # Check if any neighboring pixel (8-connectivity) is part of the edge
    # Make sure the indices are within bounds
    rows, cols = edges.shape
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols and edges[ny, nx] > 0:
                return True
    return False


def flood_fill_skin_region(image, skin_mask, edges, flooded_mask):
    # Convert the skin mask to a binary mask for processing
    skin_mask = skin_mask.astype(np.uint8)

    # Create a separate mask to visualize the regions that will be filled
    fill_mask = np.zeros_like(skin_mask, dtype=np.uint8)

    # Use flood fill to mark regions enclosed by the edges
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Start a flood fill if the pixel is skin, hasn't been filled yet, and is not adjacent to an edge
            if skin_mask[y, x] == 255 and flooded_mask[y, x] == 0 and not is_adjacent_to_edge(y, x, edges):
                # Flood fill from this skin pixel
                cv2.floodFill(flooded_mask, None, (x, y), 255, loDiff=5, upDiff=5)
                # Mark the regions that are going to be filled
                fill_mask[y, x] = 255

    return flooded_mask, fill_mask


def apply_whitening(hsv_image, mask, increase_v_factor=1.5, decrease_s_factor=0.5):
    # Extract HSV channels
    h_channel = hsv_image[..., 0]
    s_channel = hsv_image[..., 1]
    v_channel = hsv_image[..., 2]

    # Increase the V (brightness) channel and decrease the S (saturation) channel within the mask
    v_channel = np.where(mask == 255, np.clip(v_channel * increase_v_factor, 0, 255), v_channel)
    s_channel = np.where(mask == 255, np.clip(s_channel * decrease_s_factor, 0, 255), s_channel)

    # Rebuild the HSV image
    hsv_image[..., 0] = h_channel
    hsv_image[..., 1] = s_channel
    hsv_image[..., 2] = v_channel

    # Convert back to BGR to visualize the result
    modified_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return modified_image


def remove_outer_layers(mask, num_layers=2):
    """
    Removes the outermost `num_layers` pixels from the mask using morphological erosion.
    """
    # Define the kernel for erosion (a square kernel with a size of 3x3)
    kernel = np.ones((3, 3), np.uint8)

    # Apply erosion `num_layers` times
    for _ in range(num_layers):
        mask = cv2.erode(mask, kernel, iterations=1)

    return mask


def close_mask(mask):
    """
    Apply morphological closing (dilation followed by erosion) to close small gaps.
    """
    kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size to fill gaps
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def process_image(image, skin_mask):
    # Step 1: Perform Canny edge detection and apply morphological closing to the edges
    edges = canny_edge_detection(image)

    # Step 2: Initialize the flooded mask (to track filled regions)
    flooded_mask = np.zeros_like(skin_mask, dtype=np.uint8)

    # Step 3: Apply flood fill to detect regions surrounded by edges (only fill unmarked skin pixels not touching the edge)
    filled_mask, fill_mask = flood_fill_skin_region(image, skin_mask, edges, flooded_mask)

    # Step 4: Remove the outermost two layers from the skin mask
    cleaned_mask = remove_outer_layers(skin_mask, num_layers=2)

    # Step 5: Apply morphological closing to close small gaps
    closed_mask = close_mask(cleaned_mask)

    # Step 6: Apply whitening effect on the skin regions surrounded by edges
    whitened_image = apply_whitening(image, filled_mask)

    return whitened_image, fill_mask, closed_mask


# Example usage:

image = cv2.imread('face_black.jpg')
lower_hsv = np.array([11, 133, 174])  # Example skin tone lower HSV range
upper_hsv = np.array([13, 157, 216])  # Example skin tone upper HSV range

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
skinMask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
cv2.imshow('skinMask', skinMask)
cv2.waitKey(0)

processed_image, fill_mask, closed_mask = process_image(hsv_image, skinMask)

# Highlight regions to be filled in green on the original image
fill_overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization
fill_overlay[fill_mask == 255] = [0, 255, 0]  # Highlight regions to be filled in green

# Show the result
cv2.imshow("Regions to be filled", fill_overlay)
cv2.waitKey(0)

# Display the result
cv2.imshow('Processed Image', processed_image)
cv2.waitKey(0)

# Show cleaned and closed mask
cv2.imshow('Closed Skin Mask', closed_mask)
cv2.waitKey(0)

cv2.destroyAllWindows()
