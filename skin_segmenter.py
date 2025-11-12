import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def get_exposed_skin_mask(image: np.ndarray, model_path: str = 'selfie_multiclass_256x256.tflite') -> np.ndarray:
    # Resolve model path relative to this file if not absolute
    model_file = Path(model_path)
    if not model_file.is_absolute():
        model_file = Path(__file__).with_name(model_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    # Use model_asset_buffer to avoid path resolution issues on Windows
    model_bytes = model_file.read_bytes()
    base_options = python.BaseOptions(model_asset_buffer=model_bytes)
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=True,
    )

    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        # Create MediaPipe image (expects RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Segment the image
        segmentation_result = segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask.numpy_view()

        # Condition for exposed skin (categories 2: body-skin, 3: face-skin)
        skin_condition = np.isin(category_mask, [2, 3])

        # Convert to uint8 binary mask (255 for skin, 0 for background)
        skin_mask = skin_condition.astype(np.uint8) * 255

    return skin_mask
#
# input_image = cv2.imread('imgs/2face.jpg')
# input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
# mask = get_exposed_skin_mask(input_image_rgb)
# cv2.imwrite('skin_mask.jpg', mask)
