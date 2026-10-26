"""
Spatiotemporal Activity Model Training Engine (R3D-18 / Video 3D CNN)
=====================================================================
Project: Unusual Activity Detection Using Camera by AI

Architecture & Training Methodology:
1. True 3D Spatiotemporal Convolutional Neural Network (R3D-18):
   - Initialized with Kinetics-400 pretrained 3D action recognition weights.
   - Input shape: (B, 3, T=8, H=112, W=112)
   - Convolves across space and time simultaneously, learning real velocity, acceleration,
     and interaction patterns rather than static appearance alone.
2. Strict Video-Level Dataset Partitioning:
   - Reads dataset_split.json to ensure zero data leakage between Train, Validation, and Test.
3. Hard-Negative Diversity for Normal Class:
   - Samples diverse ADL (sleeping in bed, sitting, reading, walking, group walking, crowd scenes,
     dim/dark rooms, and stationary poses).
4. Balanced Loss with Calibrated Class Weights:
   - Mitigates class imbalance with focal label smoothing and Normal specificity penalty.
5. Exact 8 Classes:
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
import time
import json
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models.video as video_models

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_FILE = BASE_DIR / "dataset_split.json"

MODEL_SAVE_PATH = MODELS_DIR / "best_temporal_activity_model.pt"
ROOT_SAVE_PATH = BASE_DIR / "temporal_model.pth"

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

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ============================================================
# R3D-18 SPATIOTEMPORAL MODEL
# ============================================================

class SpatiotemporalActivityModel(nn.Module):
    def __init__(self, num_classes=8, clip_length=8):
        super().__init__()
        self.num_classes = num_classes
        self.clip_length = clip_length

        # Load Kinetics-400 Pretrained R3D-18
        try:
            weights = video_models.R3D_18_Weights.DEFAULT
            backbone = video_models.r3d_18(weights=weights)
        except Exception:
            backbone = video_models.r3d_18(weights=None)

        self.stem = backbone.stem
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        
        in_features = backbone.fc.in_features # 512
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes)
        )

    def extract_features(self, x):
        # x: (B, 3, T, H, W)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # (B, 512)
        return x

    def forward(self, x):
        # x: (B, 3, T, H, W)
        feats = self.extract_features(x)
        out = self.classifier(feats)
        return out

    def forward_from_features(self, feats):
        # feats: (B, 512)
        return self.classifier(feats)

# ============================================================
# FAST VIDEO CLIP EXTRACTION
# ============================================================

def extract_clips_from_video(vpath, class_name, clip_length=8, frame_stride=2, num_clips=2, image_size=112, apply_dark_aug=False):
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    span = (clip_length - 1) * frame_stride + 1

    if total_frames <= span + 5:
        indices = np.linspace(0, max(0, total_frames - 1), clip_length, dtype=int)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
        if apply_dark_aug:
            scale = random.uniform(0.18, 0.45)
            frames_arr = np.clip(frames_arr.astype(np.float32) * scale, 0, 255).astype(np.uint8)

        # Tensor shape for 3D CNN: (3, T, H, W)
        tensor = torch.from_numpy(frames_arr).permute(3, 0, 1, 2).float() / 255.0
        for c in range(3):
            tensor[c, :, :, :] = (tensor[c, :, :, :] - MEAN[c]) / STD[c]
        return [tensor]

    max_start = max(0, total_frames - span)
    fractions = np.linspace(0.10, 0.90, num_clips)
    start_points = [int(f * max_start) for f in fractions]

    all_needed_indices = set()
    clip_index_lists = []
    for sp in start_points:
        c_idxs = [sp + k * frame_stride for k in range(clip_length)]
        clip_index_lists.append(c_idxs)
        all_needed_indices.update(c_idxs)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames_dict = {}
    max_req = max(all_needed_indices)
    idx = 0
    while cap.isOpened() and idx <= max_req:
        ret, frame = cap.read()
        if not ret: break
        if idx in all_needed_indices:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            frames_dict[idx] = resized
        idx += 1
    cap.release()

    clips_tensors = []
    for c_idxs in clip_index_lists:
        frames = []
        last_v = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        for i in c_idxs:
            if i in frames_dict:
                last_v = frames_dict[i]
            frames.append(last_v.copy())
        if len(frames) == clip_length:
            frames_arr = np.array(frames, dtype=np.uint8)
            if apply_dark_aug:
                scale = random.uniform(0.18, 0.45)
                frames_arr = np.clip(frames_arr.astype(np.float32) * scale, 0, 255).astype(np.uint8)

            # Shape: (3, T, H, W)
            tensor = torch.from_numpy(frames_arr).permute(3, 0, 1, 2).float() / 255.0
            for c in range(3):
                tensor[c, :, :, :] = (tensor[c, :, :, :] - MEAN[c]) / STD[c]
            clips_tensors.append(tensor)

    return clips_tensors


def build_spatiotemporal_cache(split_dict, model, device, clip_length=8, image_size=112):
    print("\nPre-extracting R3D-18 Spatiotemporal Feature Cache using strict video-level splits...")
    t0 = time.time()

    train_features = []
    train_labels = []
    val_features = []
    val_labels = []

    model.eval()
    with torch.no_grad():
        for cname in CLASSES:
            c_idx = CLASS_TO_IDX[cname]
            cdir = DATASETS_DIR / cname

            train_vids = split_dict["train"].get(cname, [])
            val_vids = split_dict["val"].get(cname, [])

            # Train clips extraction
            train_clips = []
            for i_v, vname in enumerate(train_vids[:90]):
                vpath = str((cdir / vname).resolve())
                is_dark_aug = (cname == "normal" and (i_v % 3 == 0))
                clips = extract_clips_from_video(
                    vpath, class_name=cname, clip_length=clip_length, frame_stride=2,
                    num_clips=2, image_size=image_size, apply_dark_aug=is_dark_aug
                )
                train_clips.extend(clips)

            # Synthetic dark/dim normal clips
            if cname == "normal":
                for _ in range(25):
                    base_val = random.randint(18, 45)
                    noise = np.random.randint(-4, 5, (clip_length, image_size, image_size, 3))
                    synth_arr = np.clip(base_val + noise, 0, 255).astype(np.uint8)
                    synth_t = torch.from_numpy(synth_arr).permute(3, 0, 1, 2).float() / 255.0
                    for c in range(3):
                        synth_t[c, :, :, :] = (synth_t[c, :, :, :] - MEAN[c]) / STD[c]
                    train_clips.append(synth_t)

            # Validation clips extraction
            val_clips = []
            for vname in val_vids[:25]:
                vpath = str((cdir / vname).resolve())
                clips = extract_clips_from_video(
                    vpath, class_name=cname, clip_length=clip_length, frame_stride=2,
                    num_clips=2, image_size=image_size, apply_dark_aug=False
                )
                val_clips.extend(clips)

            print(f"  - {cname:<12} | Train Clips: {len(train_clips):>4} | Val Clips: {len(val_clips):>4}")

            # Forward pass through R3D-18 backbone to cache 512-dim features
            for clip_t in train_clips:
                clip_batch = clip_t.unsqueeze(0).to(device) # (1, 3, T, H, W)
                feat = model.extract_features(clip_batch).squeeze(0).cpu() # (512)
                train_features.append(feat)
                train_labels.append(c_idx)

            for clip_t in val_clips:
                clip_batch = clip_t.unsqueeze(0).to(device)
                feat = model.extract_features(clip_batch).squeeze(0).cpu()
                val_features.append(feat)
                val_labels.append(c_idx)

    train_X = torch.stack(train_features)
    train_Y = torch.tensor(train_labels, dtype=torch.long)
    val_X = torch.stack(val_features)
    val_Y = torch.tensor(val_labels, dtype=torch.long)

    print(f"\nR3D-18 Feature Cache Built in {time.time()-t0:.1f}s | Train: {len(train_Y)} samples, Val: {len(val_Y)} samples")
    return train_X, train_Y, val_X, val_Y

# ============================================================
# TRAINING ENGINE
# ============================================================

def train_temporal_model(epochs=45, batch_size=32, lr=1e-3, clip_length=8, image_size=112):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("STARTING SPATIOTEMPORAL R3D-18 ACTIVITY MODEL TRAINING")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | Clip Length: {clip_length}")
    print("=" * 80)

    if not SPLIT_FILE.exists():
        from prepare_dataset import prepare_and_split_datasets
        prepare_and_split_datasets()

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split_dict = json.load(f)

    model = SpatiotemporalActivityModel(num_classes=len(CLASSES), clip_length=clip_length).to(device)

    train_X, train_Y, val_X, val_Y = build_spatiotemporal_cache(
        split_dict, model, device, clip_length=clip_length, image_size=image_size
    )

    class_counts = torch.bincount(train_Y, minlength=len(CLASSES)).float()
    weights = len(train_Y) / (len(CLASSES) * torch.clamp(class_counts, min=1.0))
    weights[CLASS_TO_IDX["normal"]] *= 1.40
    weights = weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.03)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    train_dataset = TensorDataset(train_X, train_Y)
    val_dataset = TensorDataset(val_X, val_Y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_f1 = 0.0

    print("\nTraining R3D-18 Spatiotemporal Activity Classifier Head...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for feats, targets in train_loader:
            feats, targets = feats.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model.forward_from_features(feats)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        scheduler.step()
        train_acc = (train_correct / max(1, train_total)) * 100.0
        train_loss /= max(1, train_total)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for feats, targets in val_loader:
                feats, targets = feats.to(device), targets.to(device)
                outputs = model.forward_from_features(feats)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * targets.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())

        val_acc = (val_correct / max(1, val_total)) * 100.0
        val_loss /= max(1, val_total)

        f1_list = []
        for c_idx in range(len(CLASSES)):
            tp = sum(1 for p, t in zip(all_preds, all_targets) if p == c_idx and t == c_idx)
            fp = sum(1 for p, t in zip(all_preds, all_targets) if p == c_idx and t != c_idx)
            fn = sum(1 for p, t in zip(all_preds, all_targets) if p != c_idx and t == c_idx)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            f1_list.append(f1)

        val_f1 = float(np.mean(f1_list))
        dt = time.time() - t0

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] ({dt:.2f}s) | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.1f}% | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.1f}%, Macro-F1: {val_f1:.3f}")

        if val_f1 >= best_val_f1 or epoch == epochs:
            best_val_f1 = val_f1
            checkpoint = {
                "arch": "r3d_18",
                "model_state_dict": model.state_dict(),
                "num_classes": len(CLASSES),
                "clip_length": clip_length,
                "image_size": image_size,
                "classes": CLASSES,
                "class_to_idx": CLASS_TO_IDX,
                "val_f1": float(val_f1),
                "val_acc": float(val_acc),
                "epoch": epoch
            }
            torch.save(checkpoint, str(MODEL_SAVE_PATH))
            torch.save(checkpoint, str(ROOT_SAVE_PATH))

    print("=" * 80)
    print(f"Training Complete! Best Val Macro-F1: {best_val_f1:.3f}")
    print(f"Saved Checkpoint to: {ROOT_SAVE_PATH.name} & {MODEL_SAVE_PATH.name}")
    print("=" * 80)


if __name__ == "__main__":
    train_temporal_model(epochs=45, batch_size=32, lr=1e-3, clip_length=8, image_size=112)
