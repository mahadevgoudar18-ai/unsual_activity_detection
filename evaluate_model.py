"""
Comprehensive Offline Model Evaluation & Diagnostic Benchmark (R3D-18)
======================================================================
Project: Unusual Activity Detection Using Camera by AI

Reports:
1. Overall Accuracy, Macro Precision, Macro Recall, Macro F1
2. Per-class Precision, Recall, F1, and Support
3. Full 8x8 Confusion Matrix across strictly unseen holdout test videos
4. Normal False Positive Rate (FPR) and Abnormal False Negative Rate (FNR)
5. Hard-Negative Normal Subcategories (Sleeping/Resting, Walking, Crowd Movement)

Exact 8 Classes:
1. accident
2. falling
3. fighting
4. fire_smoke
5. normal
6. shooting
7. theft
8. vandalism
"""

import os
import cv2
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from activity_detector import ActivityDetector, CLASSES, DISPLAY_NAMES, safe_torch_load, SpatiotemporalActivityModel, MEAN, STD

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
SPLIT_FILE = BASE_DIR / "dataset_split.json"
MODEL_PATH = BASE_DIR / "temporal_model.pth"

def extract_r3d_clip(vpath, clip_length=8, image_size=112):
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = clip_length

    indices = np.linspace(0, max(0, total_frames - 1), clip_length, dtype=int)
    frames_dict = {}
    target_set = set(indices)
    idx = 0
    while cap.isOpened() and idx <= max(indices):
        ret, frame = cap.read()
        if not ret: break
        if idx in target_set:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            frames_dict[idx] = resized
        idx += 1
    cap.release()

    frames = []
    last_v = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    for i in indices:
        if i in frames_dict:
            last_v = frames_dict[i]
        frames.append(last_v.copy())
    while len(frames) < clip_length:
        frames.append(last_v.copy())

    frames_arr = np.array(frames[:clip_length], dtype=np.uint8)
    tensor = torch.from_numpy(frames_arr).permute(3, 0, 1, 2).float() / 255.0
    for c in range(3):
        tensor[c, :, :, :] = (tensor[c, :, :, :] - MEAN[c]) / STD[c]
    return tensor.unsqueeze(0) # (1, 3, T, H, W)


def evaluate_offline():
    device = torch.device("cpu")
    print("=" * 80)
    print("COMPREHENSIVE OFFLINE EVALUATION OF TRAINED R3D-18 SPATIOTEMPORAL MODEL")
    print("=" * 80)

    if not MODEL_PATH.exists():
        print(f"Model file {MODEL_PATH.name} not found!")
        return

    ckpt = safe_torch_load(MODEL_PATH, map_location=device)
    model = SpatiotemporalActivityModel(num_classes=8, clip_length=8)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded Model Checkpoint: {MODEL_PATH.name} | Arch: {ckpt.get('arch')} | Epoch: {ckpt.get('epoch')} | Val F1: {ckpt.get('val_f1', 0):.3f}")

    if not SPLIT_FILE.exists():
        from prepare_dataset import prepare_and_split_datasets
        prepare_and_split_datasets()

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split_dict = json.load(f)

    # =========================================================================
    # PART 1: MANDATORY NORMAL TEST (SLEEPING, LYING, SITTING, WALKING, CROWD)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: EVALUATING NORMAL ACTIVITY TEST SET (SLEEPING / LYING / SITTING / WALKING)")
    print("=" * 80)

    test_normal_vids = split_dict["test"].get("normal", [])
    sleeping_vids = [v for v in test_normal_vids if "B_N" in v]
    movement_vids = [v for v in test_normal_vids if "B_M" in v]
    cctv_vids = [v for v in test_normal_vids if "Normal_Videos" in v]

    normal_test_samples = [
        ("Sleeping / Resting in Bed", sleeping_vids),
        ("Normal Walking / Moving in Room", movement_vids),
        ("Outdoor Scene / Normal Crowd Movement", cctv_vids)
    ]

    normal_fp_counts = defaultdict(int)
    total_normal_tested = 0
    normal_correct = 0

    for category, vids in normal_test_samples:
        print(f"\n--- Subcategory: {category} ({len(vids)} holdout test videos) ---")
        for vname in vids[:6]:
            vpath = str((DATASETS_DIR / "normal" / vname).resolve())
            clip_t = extract_r3d_clip(vpath)
            if clip_t is None: continue
            with torch.inference_mode():
                logits = model(clip_t)
                probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

            pred_idx = np.argmax(probs)
            pred_class = CLASSES[pred_idx]
            conf = probs[pred_idx] * 100.0
            norm_p = probs[CLASSES.index("normal")] * 100.0

            is_correct = (pred_class == "normal")
            total_normal_tested += 1
            if is_correct:
                normal_correct += 1
            else:
                normal_fp_counts[pred_class] += 1

            status = "[PASS]" if is_correct else f"[FAIL -> {pred_class}]"
            print(f"  Video: {vname:<28} | Pred: {pred_class:<12} | Conf: {conf:>5.1f}% | Normal Prob: {norm_p:>5.1f}% | {status}")

    # =========================================================================
    # PART 2: 8-CLASS BATCH CONFUSION MATRIX ACROSS UNSEEN TEST SPLITS
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: 8-CLASS BENCHMARK & CONFUSION MATRIX ACROSS UNSEEN TEST SPLIT")
    print("=" * 80)

    confusion = np.zeros((8, 8), dtype=int)

    for true_idx, cname in enumerate(CLASSES):
        test_vids = split_dict["test"].get(cname, [])
        for vname in test_vids:
            vpath = str((DATASETS_DIR / cname / vname).resolve())
            clip_t = extract_r3d_clip(vpath)
            if clip_t is None: continue
            with torch.inference_mode():
                logits = model(clip_t)
                probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

            pred_idx = int(np.argmax(probs))
            confusion[true_idx, pred_idx] += 1

    print("\n" + "-" * 80)
    print(f"{'Class':<14} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support'}")
    print("-" * 80)

    f1_scores, prec_scores, rec_scores = [], [], []
    for i, cname in enumerate(CLASSES):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
        prec_scores.append(prec)
        rec_scores.append(rec)
        support = confusion[i, :].sum()
        print(f"{cname:<14} | {prec*100:>8.1f}% | {rec*100:>8.1f}% | {f1*100:>8.1f}% | {support:>7d}")

    macro_f1 = np.mean(f1_scores)
    macro_prec = np.mean(prec_scores)
    macro_rec = np.mean(rec_scores)
    accuracy = np.trace(confusion) / max(1, confusion.sum())

    # Detailed NORMAL Class Metrics
    norm_idx = CLASSES.index("normal")
    tp_norm = confusion[norm_idx, norm_idx]
    fp_norm = confusion[:, norm_idx].sum() - tp_norm
    fn_norm = confusion[norm_idx, :].sum() - tp_norm
    tn_norm = confusion.sum() - (tp_norm + fp_norm + fn_norm)
    
    prec_norm = tp_norm / (tp_norm + fp_norm) if (tp_norm + fp_norm) > 0 else 0.0
    rec_norm = tp_norm / (tp_norm + fn_norm) if (tp_norm + fn_norm) > 0 else 0.0
    spec_norm = tn_norm / (tn_norm + fp_norm) if (tn_norm + fp_norm) > 0 else 0.0
    fpr_norm = fp_norm / (fp_norm + tn_norm) if (fp_norm + tn_norm) > 0 else 0.0
    f1_norm = 2 * (prec_norm * rec_norm) / (prec_norm + rec_norm) if (prec_norm + rec_norm) > 0 else 0.0

    print("-" * 80)
    print(f"Overall Test Accuracy : {accuracy*100:.1f}%")
    print(f"Macro Precision       : {macro_prec*100:.1f}%")
    print(f"Macro Recall          : {macro_rec*100:.1f}%")
    print(f"Macro F1-Score        : {macro_f1*100:.1f}%")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("MANDATORY NORMAL CLASS HIGH-PRIORITY METRICS:")
    print(f"  - Precision           : {prec_norm*100:.1f}%")
    print(f"  - Recall              : {rec_norm*100:.1f}%")
    print(f"  - Specificity         : {spec_norm*100:.1f}%")
    print(f"  - False Positive Rate : {fpr_norm*100:.1f}%")
    print(f"  - F1-Score            : {f1_norm*100:.1f}%")
    print("=" * 80)

    print("\nCONFUSION MATRIX (Rows: True Class, Columns: Predicted Class):")
    header = "             " + " ".join([f"{c[:4]:>6}" for c in CLASSES])
    print(header)
    for i, cname in enumerate(CLASSES):
        row_str = f"{cname:<12} " + " ".join([f"{confusion[i, j]:>6d}" for j in range(8)])
        print(row_str)

    print("\n" + "=" * 80)
    print("NORMAL FALSE POSITIVE SUMMARY:")
    print(f"Total Normal Subsamples Tested: {total_normal_tested} | Correct (Normal): {normal_correct} ({(normal_correct/max(1, total_normal_tested))*100:.1f}%)")
    for c in CLASSES:
        if c != "normal":
            fp = normal_fp_counts[c]
            print(f"  Normal -> {c:<12}: {fp}")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_offline()
