"""
Dataset Preparation, Sanitization & Video-Level Split Index Generator
=====================================================================
Project: Unusual Activity Detection Using Camera by AI

Key Responsibilities:
1. Validates and organizes the dataset into exactly the 8 target classes:
   1. accident
   2. falling
   3. fighting
   4. fire_smoke
   5. normal
   6. shooting
   7. theft
   8. vandalism
2. Performs strict VIDEO-LEVEL train/val/test splitting:
   - 75% Train videos
   - 15% Validation videos
   - 10% Unseen Holdout Test videos
   - Ensures clips from the same original video NEVER cross between train and val/test splits (Zero Data Leakage).
3. Saves the deterministic split to dataset_split.json.
"""

import os
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

SEED = 42
random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
SPLIT_FILE = BASE_DIR / "dataset_split.json"

CLASSES = [
    "accident",
    "falling",
    "fighting",
    "fire_smoke",
    "normal",
    "shooting",
    "theft",
    "vandalism",
]

def prepare_and_split_datasets(train_pct=0.75, val_pct=0.15, test_pct=0.10):
    print("=" * 80)
    print("AUDITING DATASET & GENERATING STRICT VIDEO-LEVEL TRAIN/VAL/TEST SPLITS")
    print("=" * 80)

    dataset_summary = {}
    split_data = {
        "classes": CLASSES,
        "class_to_idx": {c: i for i, c in enumerate(CLASSES)},
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list),
        "counts": {}
    }

    total_videos = 0

    for c in CLASSES:
        cdir = DATASETS_DIR / c
        if not cdir.exists():
            print(f"Error: Directory {cdir} does not exist!")
            continue

        video_files = sorted([
            f.name for f in cdir.iterdir()
            if f.is_file() and f.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}
        ])

        num_vids = len(video_files)
        total_videos += num_vids

        # Shuffle deterministically
        shuffled = list(video_files)
        random.Random(SEED + len(c)).shuffle(shuffled)

        n_train = max(1, int(num_vids * train_pct))
        n_val = max(1, int(num_vids * val_pct))
        n_test = max(1, num_vids - n_train - n_val)

        train_vids = shuffled[:n_train]
        val_vids = shuffled[n_train:n_train + n_val]
        test_vids = shuffled[n_train + n_val:]

        split_data["train"][c] = train_vids
        split_data["val"][c] = val_vids
        split_data["test"][c] = test_vids
        split_data["counts"][c] = {
            "total": num_vids,
            "train": len(train_vids),
            "val": len(val_vids),
            "test": len(test_vids)
        }

        print(f"{c:<14} | Total: {num_vids:>4} | Train: {len(train_vids):>4} | Val: {len(val_vids):>4} | Test: {len(test_vids):>4}")

    # Convert defaultdict to standard dict for JSON serialization
    split_data["train"] = dict(split_data["train"])
    split_data["val"] = dict(split_data["val"])
    split_data["test"] = dict(split_data["test"])

    with open(SPLIT_FILE, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)

    print("-" * 80)
    print(f"Total Valid Videos Audited: {total_videos}")
    print(f"Saved Strict Video-Level Split Index to: {SPLIT_FILE.name}")
    print("=" * 80)

    return split_data


if __name__ == "__main__":
    prepare_and_split_datasets()
