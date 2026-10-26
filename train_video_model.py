"""
Temporal Video Activity Recognition Model Training Script
Architecture: Pretrained CNN (MobileNetV3) + Bidirectional LSTM with Temporal Attention
Dataset: UCF-Crime, Fall Detection Dataset, and Normal Activity Videos
Classes: accident, falling, fighting, fire_smoke, normal, shooting, theft, vandalism
"""

import os
import cv2
import time
import math
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODELS_DIR / "best_activity_model.pt"

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
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASSES)}

# Normalization constants (ImageNet standard)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ============================================================
# DATASET SCANNING & MAPPING
# ============================================================

def normalize_name(name):
    name = name.lower().strip()
    for old, new in [(" ", ""), ("-", ""), ("_", "")]:
        name = name.replace(old, new)
    return name


def map_folder_to_class(folder_name, parent_name=""):
    n = normalize_name(folder_name)
    p = normalize_name(parent_name)

    # Normal
    if n in {"normal", "normalvideos", "normalevent", "normalevents", "normalvideosforeventrecognition"} or "normal" in n:
        return "normal"
    # Falling
    if n in {"fmaskb1", "fmaskb2", "fmaskb3", "fmaskb4", "fall", "falls", "falling", "falldetection", "falldetectiondataset"}:
        return "falling"
    # Accident
    if n in {"roadaccident", "roadaccidents", "accident", "accidents", "roadaccidentvideos"}:
        return "accident"
    # Fighting
    if n in {"fighting", "fight", "assault", "abuse"}:
        return "fighting"
    # Theft
    if n in {"theft", "stealing", "steal", "robbery", "shoplifting", "burglary"}:
        return "theft"
    # Vandalism
    if n in {"vandalism"}:
        return "vandalism"
    # Fire / smoke
    if n in {"arson", "explosion", "fire", "fires", "smoke", "firesmoke", "fireandsmoke"}:
        return "fire_smoke"
    # Shooting
    if n in {"shooting", "shootings"}:
        return "shooting"

    return None


def scan_dataset(datasets_dir):
    """Scan directory and return mapping of class -> list of video file paths."""
    class_videos = defaultdict(list)
    skipped = 0

    for root, dirs, files in os.walk(datasets_dir):
        folder_name = Path(root).name
        parent_name = Path(root).parent.name

        cls = map_folder_to_class(folder_name, parent_name)
        if not cls:
            cls = map_folder_to_class(parent_name)

        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
                full_path = os.path.join(root, f)
                if cls in CLASS_TO_IDX:
                    class_videos[cls].append(full_path)
                else:
                    skipped += 1

    return class_videos, skipped

# ============================================================
# TEMPORAL VIDEO DATASET
# ============================================================

class VideoClipDataset(Dataset):
    """
    Samples temporal clips from video files.
    Extracts T frames, applies data augmentation, and returns (T, C, H, W) tensor with label.
    """
    def __init__(self, clip_records, clip_length=16, image_size=160, is_training=True):
        self.records = clip_records
        self.clip_length = clip_length
        self.image_size = image_size
        self.is_training = is_training

    def __len__(self):
        return len(self.records)

    def _load_frames(self, video_path, start_frame, total_frames, stride):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        if not cap.isOpened():
            # Return dummy blank tensor if video cannot be opened
            return np.zeros((self.clip_length, self.image_size, self.image_size, 3), dtype=np.uint8)

        indices = [min(total_frames - 1, start_frame + i * stride) for i in range(self.clip_length)]
        indices_set = set(indices)
        current_idx = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while current_idx < total_frames and len(frames) < self.clip_length:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            target_idx = start_frame + len(frames) * stride
            if current_idx >= target_idx:
                resized = cv2.resize(frame, (self.image_size, self.image_size))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
            current_idx += 1

        cap.release()

        # If clip is short, pad by repeating the last valid frame
        if len(frames) == 0:
            return np.zeros((self.clip_length, self.image_size, self.image_size, 3), dtype=np.uint8)

        while len(frames) < self.clip_length:
            frames.append(frames[-1].copy())

        return np.array(frames[:self.clip_length], dtype=np.uint8)

    def __getitem__(self, idx):
        record = self.records[idx]
        video_path = record["path"]
        class_idx = record["label"]
        total_frames = record["frames"]

        # Calculate stride and start frame
        stride = max(1, total_frames // (self.clip_length * 2)) if total_frames > self.clip_length * 2 else 1
        clip_span = self.clip_length * stride

        if total_frames > clip_span:
            if self.is_training:
                max_start = total_frames - clip_span
                start_frame = random.randint(0, max_start)
            else:
                start_frame = max(0, (total_frames - clip_span) // 2)
        else:
            start_frame = 0
            stride = 1

        frames = self._load_frames(video_path, start_frame, total_frames, stride)

        # Video-level data augmentation (applied consistently across all frames in clip)
        if self.is_training:
            # Random horizontal flip
            if random.random() > 0.5:
                frames = np.flip(frames, axis=2).copy()

            # Slight brightness adjustment
            brightness_factor = random.uniform(0.85, 1.15)
            frames = np.clip(frames * brightness_factor, 0, 255).astype(np.uint8)

        # Convert to Tensor (T, C, H, W) normalized to ImageNet mean/std
        # frames: (T, H, W, C) -> (T, C, H, W)
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        for c in range(3):
            tensor[:, c, :, :] = (tensor[:, c, :, :] - MEAN[c]) / STD[c]

        return tensor, torch.tensor(class_idx, dtype=torch.long)

# ============================================================
# TEMPORAL VIDEO MODEL ARCHITECTURE
# ============================================================

class TemporalAttention(nn.Module):
    """Computes soft attention weights across time steps."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (Batch, Time, Hidden)
        scores = self.attn(x) # (Batch, Time, 1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights, dim=1) # (Batch, Hidden)
        return context, weights


class TemporalActivityModel(nn.Module):
    """
    Spatial Backbone (MobileNetV3) + Bidirectional LSTM + Temporal Attention
    """
    def __init__(self, num_classes=8, clip_length=16, hidden_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.clip_length = clip_length
        self.hidden_dim = hidden_dim

        # Pretrained spatial CNN backbone
        weights = MobileNet_V3_Small_Weights.DEFAULT
        backbone = mobilenet_v3_small(weights=weights)
        self.spatial_features = backbone.features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        feature_dim = 576  # MobileNetV3-Small feature dimension

        # Temporal sequence modeling (Bi-LSTM)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Temporal attention layer
        self.attention = TemporalAttention(hidden_dim * 2)

        # Final activity classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # Fold time into batch for efficient parallel 2D CNN forward pass
        x_reshaped = x.view(B * T, C, H, W)
        spatial_feats = self.spatial_features(x_reshaped)
        spatial_pooled = self.global_pool(spatial_feats).view(B * T, -1)

        # Unfold back to temporal sequence: (B, T, feature_dim)
        temporal_seq = spatial_pooled.view(B, T, -1)

        # Temporal LSTM processing
        lstm_out, _ = self.lstm(temporal_seq) # (B, T, hidden_dim * 2)

        # Temporal attention pooling across the 16 frames
        context, _ = self.attention(lstm_out) # (B, hidden_dim * 2)

        # Activity logits
        logits = self.classifier(context)
        return logits

# ============================================================
# EVALUATION & METRICS
# ============================================================

def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model and compute multi-class metrics + confusion matrix."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(dataloader.dataset))
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Accuracy
    accuracy = float(np.mean(all_preds == all_targets)) if len(all_targets) > 0 else 0.0

    # Per-class precision, recall, F1
    per_class_metrics = {}
    num_classes = len(CLASSES)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for p, t in zip(all_preds, all_targets):
        conf_matrix[t, p] += 1

    f1_list = []
    for idx, cls_name in enumerate(CLASSES):
        tp = conf_matrix[idx, idx]
        fp = np.sum(conf_matrix[:, idx]) - tp
        fn = np.sum(conf_matrix[idx, :]) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_metrics[cls_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(np.sum(conf_matrix[idx, :]))
        }
        if np.sum(conf_matrix[idx, :]) > 0:
            f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    return avg_loss, accuracy, macro_f1, per_class_metrics, conf_matrix

# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train Temporal Video Activity Model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (8 recommended for CPU)")
    parser.add_argument("--clip_length", type=int, default=16, help="Frames per video clip")
    parser.add_argument("--image_size", type=int, default=160, help="Frame height/width")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--samples_per_class", type=int, default=60, help="Max videos per class to train on")
    args = parser.parse_args()

    print("=" * 70)
    print("TEMPORAL VIDEO ACTIVITY RECOGNITION - TRAINING")
    print("=" * 70)
    print(f"Clip Length:        {args.clip_length} frames")
    print(f"Input Resolution:   {args.image_size}x{args.image_size}")
    print(f"Batch Size:         {args.batch_size}")
    print(f"Epochs:             {args.epochs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device:     {device} ({torch.get_num_threads()} CPU threads)")

    # 1. Scan datasets
    print("\nScanning datasets directory...")
    class_videos, skipped = scan_dataset(DATASETS_DIR)
    print(f"{'Class':<15} | {'Available Videos':<18}")
    print("-" * 38)
    for cls in CLASSES:
        print(f"{cls:<15} | {len(class_videos[cls]):<18}")

    # 2. Filter corrupted videos and create video-level splits
    print("\nValidating videos and splitting 70% Train / 15% Val / 15% Test...")
    train_records = []
    val_records = []
    test_records = []
    class_weights = []

    for cls_name in CLASSES:
        vids = class_videos[cls_name]
        random.shuffle(vids)

        # Select balanced subset per class to prevent large classes from dominating
        selected_vids = vids[:args.samples_per_class]

        valid_vids_info = []
        for vpath in selected_vids:
            try:
                cap = cv2.VideoCapture(vpath)
                if cap.isOpened():
                    fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if fcount >= args.clip_length:
                        valid_vids_info.append({"path": vpath, "frames": fcount, "label": CLASS_TO_IDX[cls_name]})
                cap.release()
            except Exception:
                continue

        # Video-level split (Prevents clip duplication between train and validation)
        n = len(valid_vids_info)
        n_train = max(1, int(n * 0.70))
        n_val = max(1, int(n * 0.15))

        c_train = valid_vids_info[:n_train]
        c_val = valid_vids_info[n_train:n_train + n_val]
        c_test = valid_vids_info[n_train + n_val:]

        train_records.extend(c_train)
        val_records.extend(c_val)
        test_records.extend(c_test)

        class_weights.append(1.0 / max(1, len(c_train)))

    # Normalize class weights for balanced loss
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * len(CLASSES)
    class_weights = class_weights.to(device)

    print(f"\nDataset Splits:")
    print(f"  Training Videos:   {len(train_records)}")
    print(f"  Validation Videos: {len(val_records)}")
    print(f"  Testing Videos:    {len(test_records)}")

    # 3. Create DataLoaders
    train_dataset = VideoClipDataset(train_records, clip_length=args.clip_length, image_size=args.image_size, is_training=True)
    val_dataset = VideoClipDataset(val_records, clip_length=args.clip_length, image_size=args.image_size, is_training=False)
    test_dataset = VideoClipDataset(test_records, clip_length=args.clip_length, image_size=args.image_size, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Initialize Model
    print("\nInitializing TemporalActivityModel with pretrained MobileNetV3 backbone...")
    model = TemporalActivityModel(num_classes=len(CLASSES), clip_length=args.clip_length)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Differential learning rate: lower lr for pretrained backbone, higher for temporal head
    optimizer = optim.AdamW([
        {"params": model.spatial_features.parameters(), "lr": args.lr * 0.1},
        {"params": model.lstm.parameters(), "lr": args.lr},
        {"params": model.attention.parameters(), "lr": args.lr},
        {"params": model.classifier.parameters(), "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # 5. Training Loop
    best_val_f1 = 0.0
    print("\n" + "=" * 70)
    print("TRAINING STARTED")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        start_time = time.time()

        for step, (inputs, targets) in enumerate(train_loader, 1):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping to stabilize LSTM gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += inputs.size(0)

        epoch_time = time.time() - start_time
        avg_train_loss = train_loss / max(1, train_total)
        train_acc = (train_correct / max(1, train_total)) * 100.0

        # Validation
        val_loss, val_acc, val_f1, _, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_f1)

        print(f"Epoch [{epoch:2d}/{args.epochs:2d}] - {epoch_time:.1f}s | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% Macro-F1: {val_f1*100:.1f}%")

        # Save Best Model Checkpoint
        if val_f1 > best_val_f1 or epoch == 1:
            best_val_f1 = val_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": CLASSES,
                "class_to_idx": CLASS_TO_IDX,
                "clip_length": args.clip_length,
                "image_size": args.image_size,
                "hidden_dim": 128,
                "epoch": epoch,
                "val_f1": val_f1,
                "val_acc": val_acc
            }, str(MODEL_SAVE_PATH))
            print(f"  >>> Checkpoint Saved: {MODEL_SAVE_PATH.name} (Val F1: {val_f1*100:.1f}%)")

    # 6. Final Evaluation on Hold-Out Test Set
    print("\n" + "=" * 70)
    print("EVALUATION ON INDEPENDENT TEST SET")
    print("=" * 70)

    checkpoint = torch.load(str(MODEL_SAVE_PATH), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_f1, per_class, conf_matrix = evaluate_model(model, test_loader, criterion, device)

    print(f"\nOverall Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Overall Test Macro-F1: {test_f1 * 100:.2f}%")
    print("-" * 70)
    print(f"{'Class':<14} | {'Precision':<11} | {'Recall':<11} | {'F1-Score':<11} | {'Samples':<8}")
    print("-" * 70)
    for cls_name in CLASSES:
        m = per_class[cls_name]
        print(f"{cls_name:<14} | {m['precision']*100:>9.1f}% | {m['recall']*100:>9.1f}% | {m['f1']*100:>9.1f}% | {m['support']:>7d}")
    print("-" * 70)

    print("\nCONFUSION MATRIX (Row: Actual Class, Column: Predicted Class):")
    print(" " * 14 + "  ".join(f"{c[:4]:>4}" for c in CLASSES))
    for i, row in enumerate(conf_matrix):
        row_str = "  ".join(f"{val:>4d}" for val in row)
        print(f"{CLASSES[i]:<14} {row_str}")

    print("\nKey Confusion Analysis:")
    # Fighting vs Normal / Theft
    fight_idx = CLASS_TO_IDX["fighting"]
    norm_idx = CLASS_TO_IDX["normal"]
    theft_idx = CLASS_TO_IDX["theft"]
    vand_idx = CLASS_TO_IDX["vandalism"]

    print(f"  - Fighting classified as Normal: {conf_matrix[fight_idx, norm_idx]}")
    print(f"  - Fighting classified as Theft:  {conf_matrix[fight_idx, theft_idx]}")
    print(f"  - Normal classified as Vandalism:{conf_matrix[norm_idx, vand_idx]}")
    print(f"  - Normal classified as Theft:    {conf_matrix[norm_idx, theft_idx]}")
    print(f"  - Theft classified as Fighting:  {conf_matrix[theft_idx, fight_idx]}")

    print("\nSUCCESS: Model successfully trained and saved to:")
    print(f"  {MODEL_SAVE_PATH}")
    print("=" * 70)

if __name__ == "__main__":
    main()

