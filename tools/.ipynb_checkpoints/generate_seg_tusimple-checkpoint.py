import argparse
import json
import os
from typing import Iterable, List, Tuple

import cv2
import numpy as np

# TuSimple official splits
TRAIN_SET = ["label_data_0313.json", "label_data_0601.json"]
VAL_SET = ["label_data_0531.json"]
TRAIN_VAL_SET = TRAIN_SET + VAL_SET
TEST_SET = ["test_label.json"]

# Dataset / label generation constants (TuSimple)
TUSIMPLE_H, TUSIMPLE_W = 720, 1280
LANE_SEG_WIDTH_PX = 30  # width used when rasterizing lanes to segmentation mask
MAX_LANES = 6           


Point = Tuple[int, int]


def _ensure_abs_like(path: str) -> str:
    """TuSimple raw_file often is like 'clips/xxx/yyy/zz.jpg'; keep original behavior:
    if it's not starting with '/', prepend '/' for list file.
    """
    return path if path.startswith("/") else ("/" + path)


def _sort_and_fill_lanes(label: dict) -> List[List[Point]]:
    """Clean lanes: remove negative x, then sort lanes by slope and map into 6 slots."""
    lanes_raw: List[List[Point]] = []
    slopes: List[float] = []

    for i in range(len(label["lanes"])):
        pts = [
            (int(x), int(y))
            for x, y in zip(label["lanes"][i], label["h_samples"])
            if x >= 0
        ]
        if len(pts) > 1:
            lanes_raw.append(pts)
            # Same slope definition as original code
            slopes.append(
                np.arctan2(pts[-1][1] - pts[0][1], pts[0][0] - pts[-1][0]) / np.pi * 180
            )

    if not lanes_raw:
        return [[] for _ in range(MAX_LANES)]

    order = np.argsort(slopes)
    lanes_sorted = [lanes_raw[i] for i in order]
    slopes_sorted = [slopes[i] for i in order]

    # Map to 6 lane indices: [0,1,2,3,4,5] based on slope split at 90 degrees (same logic)
    idx = [None for _ in range(MAX_LANES)]
    for i, s in enumerate(slopes_sorted):
        if s <= 90:
            idx[2] = i
            idx[1] = i - 1 if i > 0 else None
            idx[0] = i - 2 if i > 1 else None
        else:
            idx[3] = i
            idx[4] = i + 1 if i + 1 < len(slopes_sorted) else None
            idx[5] = i + 2 if i + 2 < len(slopes_sorted) else None
            break

    lanes_filled: List[List[Point]] = []
    for k in range(MAX_LANES):
        lanes_filled.append([] if idx[k] is None else lanes_sorted[idx[k]])
    return lanes_filled


def gen_label_for_json(root: str, savedir: str, image_set: str) -> None:
    """Generate seg PNGs and list/{split}_gt.txt for a given merged json (train_val/test)."""
    save_dir = os.path.join(root, savedir)
    os.makedirs(os.path.join(save_dir, "list"), exist_ok=True)

    list_path = os.path.join(save_dir, "list", f"{image_set}_gt.txt")
    json_path = os.path.join(save_dir, f"{image_set}.json")

    with open(list_path, "w") as list_f, open(json_path, "r") as f:
        for line in f:
            label = json.loads(line)

            lanes = _sort_and_fill_lanes(label)

            img_path = label["raw_file"]  # e.g. clips/xxx/yyy/zz.jpg
            seg_img = np.zeros((TUSIMPLE_H, TUSIMPLE_W, 3), dtype=np.uint8)

            flags: List[str] = []
            for i, coords in enumerate(lanes):
                if len(coords) < 4:
                    flags.append("0")
                    continue
                for j in range(len(coords) - 1):
                    cv2.line(
                        seg_img,
                        coords[j],
                        coords[j + 1],
                        (i + 1, i + 1, i + 1),
                        LANE_SEG_WIDTH_PX // 2,
                    )
                flags.append("1")

            # Build seg path under savedir: savedir/clips/xxx/yyy/zz.png
            parts = img_path.split("/")
            # Keep original assumption: ['clips', seq, sub_seq, img_name]
            seg_dir = os.path.join(root, savedir, parts[1], parts[2])
            os.makedirs(seg_dir, exist_ok=True)

            img_name = parts[3]
            seg_file = os.path.join(seg_dir, img_name[:-3] + "png")
            cv2.imwrite(seg_file, seg_img)

            seg_rel = "/".join([savedir, *parts[1:3], img_name[:-3] + "png"])
            seg_rel = _ensure_abs_like(seg_rel)
            img_rel = _ensure_abs_like(img_path)

            row = " ".join([img_rel, seg_rel, *flags]) + "\n"
            list_f.write(row)


def generate_json_file(root: str, save_dir: str, out_json_name: str, json_list: Iterable[str]) -> None:
    """Concatenate multiple TuSimple json files into one json file (line-by-line)."""
    out_path = os.path.join(save_dir, out_json_name)
    with open(out_path, "w") as outfile:
        for json_name in json_list:
            in_path = os.path.join(root, json_name)
            with open(in_path, "r") as infile:
                for line in infile:
                    outfile.write(line)


def generate_label(root: str, savedir: str) -> None:
    save_dir = os.path.join(root, savedir)
    os.makedirs(save_dir, exist_ok=True)

    generate_json_file(root, save_dir, "train_val.json", TRAIN_VAL_SET)
    generate_json_file(root, save_dir, "test.json", TEST_SET)

    print("generating train_val set...")
    gen_label_for_json(root, savedir, "train_val")
    print("generating test set...")
    gen_label_for_json(root, savedir, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="The root of the TuSimple dataset")
    parser.add_argument("--savedir", type=str, default="seg_label", help="Output directory name under root")
    args = parser.parse_args()

    generate_label(args.root, args.savedir)