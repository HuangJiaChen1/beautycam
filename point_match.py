import cv2
import json
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, List


def _landmarks_pixel_coords(image_bgr: np.ndarray, refine: bool = True) -> List[Tuple[int, int]]:
    h, w = image_bgr.shape[:2]
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=refine,
        min_detection_confidence=0.5,
    ) as fm:
        res = fm.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return []
        lm = res.multi_face_landmarks[0].landmark
        pts = [(int(round(p.x * w)), int(round(p.y * h))) for p in lm]
        return pts


def build_mirror_index_map(
    img_path: str = 'imgs/face_celian.jpg',
    target_size: Tuple[int, int] = (1000, 1000),
    refine: bool = True,
) -> Dict[int, int]:
    """
    - Loads the image at `img_path` and resizes to `target_size` (w,h).
    - Gets FaceMesh landmarks on the original image and the horizontally flipped image.
    - Converts flipped landmark coordinates back to the original coordinate system (mirror x).
    - For every landmark index i in the original whose (x,y) exactly matches any mirrored flipped (x,y),
      maps i -> j where j is the representative flipped index sharing the same coordinate.
    Returns a dictionary {original_index: flipped_index}.
    """

    # 1) Load and scale to 8000x8000
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {img_path}")

    target_w, target_h = target_size
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # 2) Landmarks on original
    orig_pts = _landmarks_pixel_coords(img, refine=refine)
    if not orig_pts:
        raise RuntimeError("No face landmarks detected on original image.")

    # 3) Flip and landmarks on flipped
    flipped = cv2.flip(img, 1)
    cv2.imshow("Original", img)
    cv2.imshow("Flipped", flipped)
    cv2.waitKey(0)
    flip_pts = _landmarks_pixel_coords(flipped, refine=refine)
    if not flip_pts:
        raise RuntimeError("No face landmarks detected on flipped image.")

    flip_back_pts = [(8000-x, y) for (x, y) in flip_pts]
    print(orig_pts)
    print(flip_pts)
    # print(flip_back_pts)
    # 4) Build dictionary for exact matches
    coord_to_flip_indices: Dict[Tuple[int, int], List[int]] = {}
    for j, xy in enumerate(flip_back_pts):
        coord_to_flip_indices.setdefault(xy, []).append(j)

    mapping: Dict[int, int] = {}
    for i, xy in enumerate(orig_pts):
        if xy in coord_to_flip_indices:
            # Representative flipped index = first occurrence
            mapping[i] = coord_to_flip_indices[xy][0]

    return mapping


if __name__ == '__main__':
    mapping = build_mirror_index_map()
    print(f"Exact-coordinate mirror matches: {len(mapping)} landmarks")
    # Pretty print a small sample
    sample_items = list(mapping.items())[:20]
    for k, v in sample_items:
        print(f"{k} -> {v}")

    # Save full mapping for inspection
    with open('point_match_map.json', 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print("Saved mapping to point_match_map.json")
