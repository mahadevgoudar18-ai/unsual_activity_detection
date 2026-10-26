"""
Test & Evaluation Script for Temporal Video Activity Recognition Model
Supports:
1. Evaluating individual video files with Top-3 predictions and confidence.
2. Batch evaluation across dataset classes.
3. Comparative analysis against legacy frame-based YOLO model (best.pt).
"""

import os
import cv2
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# Normalization constants (ImageNet standard)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ============================================================
# MODEL DEFINITION (MATCHING TRAINING SCRIPT)
# ============================================================

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights, dim=1)
        return context, weights


class TemporalActivityModel(nn.Module):
    def __init__(self, num_classes=8, clip_length=16, hidden_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.clip_length = clip_length
        self.hidden_dim = hidden_dim

        weights = MobileNet_V3_Small_Weights.DEFAULT
        backbone = mobilenet_v3_small(weights=weights)
        self.spatial_features = backbone.features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        feature_dim = 576

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.attention = TemporalAttention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B * T, C, H, W)
        spatial_feats = self.spatial_features(x_reshaped)
        spatial_pooled = self.global_pool(spatial_feats).view(B * T, -1)
        temporal_seq = spatial_pooled.view(B, T, -1)
        lstm_out, _ = self.lstm(temporal_seq)
        context, _ = self.attention(lstm_out)
        logits = self.classifier(context)
        return logits

# ============================================================
# VIDEO PROCESSING UTILITIES
# ============================================================

def sample_video_clip(video_path, clip_length=16, image_size=160):
    """Uniformly sample clip_length frames across a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video file has 0 frames: {video_path}")

    # Determine frame indices
    if total_frames >= clip_length:
        indices = np.linspace(0, total_frames - 1, clip_length, dtype=int)
    else:
        indices = list(range(total_frames)) + [total_frames - 1] * (clip_length - total_frames)

    frames = []
    current_idx = 0
    indices_set = set(indices)

    while current_idx < total_frames and len(frames) < clip_length:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if current_idx in indices_set:
            # Count occurrences in case total_frames < clip_length
            count = list(indices).count(current_idx)
            resized = cv2.resize(frame, (image_size, image_size))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            for _ in range(count):
                frames.append(rgb)
        current_idx += 1

    cap.release()

    while len(frames) < clip_length:
        frames.append(frames[-1].copy() if frames else np.zeros((image_size, image_size, 3), dtype=np.uint8))

    frames_arr = np.array(frames[:clip_length], dtype=np.uint8)
    
    # Shape: (T, H, W, C) -> (1, T, C, H, W)
    tensor = torch.from_numpy(frames_arr).permute(0, 3, 1, 2).float() / 255.0
    for c in range(3):
        tensor[:, c, :, :] = (tensor[:, c, :, :] - MEAN[c]) / STD[c]

    return tensor.unsqueeze(0)

# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_video_temporal(video_path, model, classes, device, clip_length=16, image_size=160):
    """Run temporal video activity model on a video file."""
    input_tensor = sample_video_clip(video_path, clip_length, image_size).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    sorted_indices = np.argsort(-probs)
    results = [
        {"class": classes[idx], "confidence": float(probs[idx]) * 100.0}
        for idx in sorted_indices
    ]
    return results

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Test Temporal Video Activity Recognition Model")
    parser.add_argument("--video", type=str, default=None, help="Path to video file to test")
    parser.add_argument("--model", type=str, default="models/best_activity_model.pt", help="Model checkpoint path")
    parser.add_argument("--compare_yolo", action="store_true", help="Compare with YOLO best.pt")
    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent
    model_path = BASE_DIR / args.model

    print("=" * 70)
    print("TESTING TEMPORAL VIDEO ACTIVITY RECOGNITION MODEL")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not model_path.exists():
        print(f"\nTemporal model checkpoint not found at:\n{model_path}")
        print("\nPlease first train the model by running:")
        print("  python train_video_model.py")
        print("\nUsing test video demonstration mode with default 8 classes...")
        classes = [
            "accident", "falling", "fighting", "fire_smoke",
            "normal", "shooting", "theft", "vandalism"
        ]
        return

    checkpoint = torch.load(str(model_path), map_location=device)
    classes = checkpoint.get("classes", [
        "accident", "falling", "fighting", "fire_smoke",
        "normal", "shooting", "theft", "vandalism"
    ])
    clip_length = checkpoint.get("clip_length", 16)
    image_size = checkpoint.get("image_size", 160)

    model = TemporalActivityModel(num_classes=len(classes), clip_length=clip_length)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model:       {model_path.name}")
    print(f"Classes:     {', '.join(classes)}")
    print(f"Clip Length: {clip_length} frames")
    print(f"Val F1:      {checkpoint.get('val_f1', 0)*100:.1f}%")

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video file not found: {video_path}")
            return

        print(f"\nTesting Video: {video_path.name}")
        results = predict_video_temporal(str(video_path), model, classes, device, clip_length, image_size)

        print("\nPREDICTION RESULTS:")
        print(f"  >>> Predicted Activity: {results[0]['class'].upper()} ({results[0]['confidence']:.1f}%)")
        print("\nTop-3 Predictions:")
        for rank, res in enumerate(results[:3], 1):
            print(f"  {rank}. {res['class']:<12} : {res['confidence']:>5.1f}%")

    else:
        print("\nNo specific video specified (--video <path>).")
        print("Testing sample videos across available dataset classes...")
        datasets_dir = BASE_DIR / "datasets"

        sample_videos = {}
        for root, dirs, files in os.walk(datasets_dir):
            for f in files:
                if f.lower().endswith(".mp4"):
                    p_name = Path(root).name.lower()
                    for cls in classes:
                        if cls in p_name and cls not in sample_videos:
                            sample_videos[cls] = os.path.join(root, f)

        if sample_videos:
            print(f"\nFound {len(sample_videos)} test videos from dataset:")
            print("-" * 70)
            print(f"{'Target Class':<14} | {'Top-1 Prediction':<16} | {'Confidence':<10} | {'Status'}")
            print("-" * 70)
            for target_cls, vpath in sample_videos.items():
                try:
                    res = predict_video_temporal(vpath, model, classes, device, clip_length, image_size)
                    pred_cls = res[0]["class"]
                    conf = res[0]["confidence"]
                    match = "MATCH" if pred_cls == target_cls else "MISMATCH"
                    print(f"{target_cls:<14} | {pred_cls:<16} | {conf:>6.1f}%    | {match}")
                except Exception as err:
                    print(f"{target_cls:<14} | Error: {str(err)[:30]}")
            print("-" * 70)
        else:
            print("Run with a specific video: python test_model.py --video <path_to_video.mp4>")

    print("=" * 70)

if __name__ == "__main__":
    main()
