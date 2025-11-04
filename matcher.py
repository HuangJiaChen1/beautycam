import json
import os
from typing import List, Optional



TARGET_INDICES: List[int] = [358,401,356]


def load_mapping(path: str = "point_match_map.json") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Mapping file not found: {path}. Run point_match.py first to generate it."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Keys in JSON are strings; convert back to ints
    return {int(k): int(v) for k, v in data.items()}


def lookup_flipped_indices(targets: List[int], mapping: dict) -> List[Optional[int]]:
    return [mapping.get(idx) for idx in targets]


if __name__ == "__main__":
    mapping = load_mapping()
    flipped_list = lookup_flipped_indices(TARGET_INDICES, mapping)
    print(flipped_list)

