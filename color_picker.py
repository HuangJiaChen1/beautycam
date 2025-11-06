import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from colorsys import rgb_to_hsv
import numpy as np
import sys

# Load your image (replace with your actual path!)
img_path = 'face_up.jpg'  # e.g., 'C:/Users/You/image.png'
try:
    img = mpimg.imread(img_path)
    print(f"Image loaded successfully: Shape {img.shape}")
except FileNotFoundError:
    print(f"Error: Image not found at '{img_path}'. Please update the path.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading image: {e}")
    sys.exit(1)

# Enable interactive mode
plt.ion()

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(img)
ax.set_title('Click on a point to get color values (close window to exit)')


def on_click(event):
    if event.inaxes != ax:
        return
    try:
        # Get pixel coordinates
        col = int(event.xdata)
        row = int(event.ydata)
        if not (0 <= row < img.shape[0] and 0 <= col < img.shape[1]):
            print("Click outside image bounds—try again.")
            return

        # Extract pixel
        pixel = img[row, col]
        if len(pixel) == 4:
            pixel = pixel[:3]  # Ignore alpha

        # Normalize to [0,1]
        if img.dtype == np.uint8:
            r, g, b = pixel / 255.0
        else:
            r, g, b = pixel

        print(f"\nClicked at pixel ({col}, {row})")
        print(f"RGB (0-255): ({int(r * 255)}, {int(g * 255)}, {int(b * 255)})")
        print(f"RGB (0-1): ({r:.3f}, {g:.3f}, {b:.3f})")

        # HSV (H in degrees 0-360, S/V 0-1)
        h, s, v = rgb_to_hsv(r, g, b)
        print(f"HSV: ({h * 360:.1f}°, {s:.3f}, {v:.3f})")

        # YCbCr (standard BT.601, values 0-1)
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
        print(f"YCbCr: ({y:.3f}, {cb:.3f}, {cr:.3f})")
        print("-" * 50)

    except Exception as e:
        print(f"Click error: {e}")


# Connect event
fig.canvas.mpl_connect('button_press_event', on_click)

# Show and block for interaction
print("Plot displayed. Click on the image! (Right-click or close window to stop.)")
plt.show(block=True)

# Optional fallback: Print colors from a specific point (uncomment and set row/col)
# row, col = 100, 150  # Example coordinates
# print("Fallback: Colors at (row=100, col=150)")
# ... (paste the extraction code from on_click here)