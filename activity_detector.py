"""
Spatiotemporal Activity Detection Engine (R3D-18 / 3D CNN)
===========================================================
Project: Unusual Activity Detection Using Camera by AI

Calibrated Robust Decision Pipeline:
1. True 3D Spatiotemporal Convolutional Model (R3D-18):
   - Input shape: (1, 3, T=8, H=112, W=112)
   - Evaluates consecutive temporal frames as a coherent spatiotemporal tensor.
2. Rolling Temporal Frame Buffer:
   - Continuously receives frames from the live camera stream.
   - Preserves temporal order and normalizes consistently with ImageNet RGB stats.
3. Motion, Liveness & Hard-Negative Normal Guard:
   - Stationary scenes (sleeping, resting in bed, sitting still, reading, looking at camera)
     strictly anchor to NORMAL with 0.0% anomaly score.
   - Ordinary indoor hand/body movement (walking, typing, phone use, standing, gestures)
     remains strictly in NORMAL / ANALYZING unless sustained abnormal evidence is confirmed.
4. Calibrated Confirmation Policy:
   - Abnormal confirmation requires high confidence (p >= 0.60), high margin over Normal (>= 0.25),
     and sustained temporal streak (>= 3 consecutive windows).
   - Uncertain predictions default safely to NORMAL.
5. Separation of Live State and Alert History:
   - Live camera display reflects latest temporal window.
   - Alert history stores confirmed past events.

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
import time
import uuid
import psutil
import threading
import numpy as np
from collections import deque
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models.video as video_models

# ============================================================
# EXACT 8 ACTIVITIES
# ============================================================

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

DISPLAY_NAMES = {
    "accident": "Accident",
    "falling": "Falling",
    "fighting": "Fighting",
    "fire_smoke": "Fire / Smoke",
    "normal": "Normal",
    "shooting": "Shooting",
    "theft": "Theft",
    "vandalism": "Vandalism",
}

ABNORMAL_CLASSES = [
    "accident",
    "falling",
    "fighting",
    "fire_smoke",
    "shooting",
    "theft",
    "vandalism",
]

ACTIVITY_MIN_MOTION = {
    "falling": 0.015,
    "accident": 0.020,
    "fighting": 0.025,
    "theft": 0.015,
    "vandalism": 0.020,
    "fire_smoke": 0.008,
    "shooting": 0.012,
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)

# ============================================================
# R3D-18 SPATIOTEMPORAL MODEL ARCHITECTURE
# ============================================================

class SpatiotemporalActivityModel(nn.Module):
    def __init__(self, num_classes=8, clip_length=8):
        super().__init__()
        self.num_classes = num_classes
        self.clip_length = clip_length

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
        
        in_features = backbone.fc.in_features
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
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        feats = self.extract_features(x)
        out = self.classifier(feats)
        return out

    def forward_from_features(self, feats):
        return self.classifier(feats)

# ============================================================
# ROLLING TEMPORAL FRAME BUFFER
# ============================================================

class RollingFrameBuffer:
    def __init__(self, clip_length=8, image_size=112):
        self.clip_length = clip_length
        self.image_size = image_size
        self.frame_buffer = deque(maxlen=clip_length)
        self.motion_buffer = deque(maxlen=clip_length)
        self.original_frames = deque(maxlen=clip_length)
        self.lock = threading.Lock()
        
        self.mean = torch.tensor(MEAN).view(3, 1, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1, 1)

    def append(self, bgr_frame, motion_score):
        resized = cv2.resize(bgr_frame, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        with self.lock:
            self.frame_buffer.append(rgb)
            self.motion_buffer.append(motion_score)
            self.original_frames.append(bgr_frame)

    def get_clip_tensor(self, device):
        with self.lock:
            count = len(self.frame_buffer)
            if count == 0:
                return None
            
            frames_list = list(self.frame_buffer)
            if count >= self.clip_length:
                seq = frames_list[-self.clip_length:]
            else:
                seq = [frames_list[0]] * (self.clip_length - count) + frames_list

            arr = np.array(seq, dtype=np.uint8)
            tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).float() / 255.0
            tensor = (tensor - self.mean) / self.std
            return tensor.unsqueeze(0).to(device)

    def get_recent_motion_energy(self):
        with self.lock:
            if not self.motion_buffer:
                return 0.0
            return float(np.mean(self.motion_buffer))

    def get_original_frames(self):
        with self.lock:
            return list(self.original_frames)

    def clear(self):
        with self.lock:
            self.frame_buffer.clear()
            self.motion_buffer.clear()
            self.original_frames.clear()

    def __len__(self):
        with self.lock:
            return len(self.frame_buffer)

# ============================================================
# FAST MOTION & LIVENESS ANALYZER
# ============================================================

class FastMotionAnalyzer:
    def __init__(self):
        self.prev_small_gray = None

    def update_and_check(self, bgr_frame):
        small = cv2.resize(bgr_frame, (64, 48), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        brightness = float(np.mean(gray))
        contrast_std = float(np.std(gray))
        
        is_low_light = (brightness < 45.0 or contrast_std < 8.0)
        is_pitch_black = brightness < 20.0

        if self.prev_small_gray is None:
            self.prev_small_gray = gray
            return 0.0, True, is_low_light, is_pitch_black

        delta = cv2.absdiff(self.prev_small_gray, gray)
        thresh = (delta > 15).astype(np.uint8)
        diff_ratio = float(np.sum(thresh) / thresh.size)
        self.prev_small_gray = gray

        # Motion threshold for stationary scenes (sleeping, resting, sitting still)
        is_stationary = (diff_ratio < 0.010)
        return diff_ratio, is_stationary, is_low_light, is_pitch_black

    def reset(self):
        self.prev_small_gray = None

# ============================================================
# PROBABILITY SMOOTHER & CALIBRATED EVENT STATE MANAGER
# ============================================================

class ProbabilitySmoother:
    def __init__(self, num_classes=8, alpha=0.35):
        self.num_classes = num_classes
        self.alpha = alpha
        self.smoothed_probs = None

    def update(self, raw_probs):
        if self.smoothed_probs is None:
            self.smoothed_probs = raw_probs.copy()
        else:
            self.smoothed_probs = (self.alpha * raw_probs) + ((1.0 - self.alpha) * self.smoothed_probs)

        s_sum = self.smoothed_probs.sum()
        if s_sum > 0:
            self.smoothed_probs = self.smoothed_probs / s_sum
        return self.smoothed_probs.copy()

    def reset(self):
        self.smoothed_probs = None


class EventStateManager:
    """
    4-State Decoupled Lifecycle Manager:
    - NORMAL: Regular daily activities, sleeping/sitting in bed, walking, phone use (0.0% anomaly)
    - ANALYZING: Transitioning or uncertain scenes
    - POSSIBLE_ACTIVITY: Elevated candidate
    - CONFIRMED_ACTIVITY: Confirmed abnormal event (streak >= 3, conf >= 60%, margin >= 25%)
    """
    def __init__(self, required_streak=3, cooldown_seconds=12.0):
        self.current_state = "NORMAL"
        self.active_activity = "normal"
        self.candidate_activity = None
        self.candidate_streak = 0
        self.normal_streak = 0
        self.confirmation_progress = 0
        self.required_streak = required_streak
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_time = 0.0
        self.last_alert_activity = None
        self.current_event_id = None

    def update(self, probs, motion_energy=0.0, is_stationary=False, is_low_light=False, is_pitch_black=False):
        probs_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
        
        # 1. STATIONARY SCENE GUARD (Sleeping, Resting in Bed, Sitting Still)
        if is_stationary or motion_energy < 0.008:
            probs_dict["normal"] = max(probs_dict["normal"], 0.98)
            for c in ABNORMAL_CLASSES:
                probs_dict[c] = min(probs_dict[c], 0.02)

        # 2. ACTIVITY-SPECIFIC MOTION VALIDATION
        for c in ABNORMAL_CLASSES:
            min_motion = ACTIVITY_MIN_MOTION.get(c, 0.015)
            if motion_energy < min_motion:
                damp = max(0.1, motion_energy / max(0.001, min_motion))
                probs_dict[c] *= damp

        # 3. LOW-LIGHT & PITCH-BLACK GUARDS
        if is_low_light:
            if is_stationary or probs_dict["normal"] >= 0.25:
                probs_dict["fighting"] = min(probs_dict["fighting"], 0.06)
                probs_dict["theft"] = min(probs_dict["theft"], 0.06)
                probs_dict["fire_smoke"] = min(probs_dict["fire_smoke"], 0.03)
                probs_dict["shooting"] = min(probs_dict["shooting"], 0.03)

        if is_pitch_black:
            probs_dict["fire_smoke"] = 0.0
            probs_dict["shooting"] = 0.0

        # Renormalize
        tot = sum(probs_dict.values())
        if tot > 0:
            for k in probs_dict:
                probs_dict[k] /= tot

        normal_p = probs_dict["normal"]
        top_abnormal_class = max(ABNORMAL_CLASSES, key=lambda c: probs_dict[c])
        top_abnormal_p = probs_dict[top_abnormal_class]

        # Calibrated Eligibility for abnormal candidate:
        min_req_motion = ACTIVITY_MIN_MOTION.get(top_abnormal_class, 0.012)
        margin_over_normal = top_abnormal_p - normal_p
        
        is_abnormal_candidate = (
            not is_stationary and
            (motion_energy >= min_req_motion) and
            (top_abnormal_class == max(CLASSES, key=lambda c: probs_dict[c])) and
            (top_abnormal_p >= 0.60) and
            (margin_over_normal >= 0.25)
        )

        anomaly_score = 0.0

        if is_abnormal_candidate:
            self.normal_streak = 0
            if self.candidate_activity == top_abnormal_class:
                self.candidate_streak += 1
            else:
                self.candidate_activity = top_abnormal_class
                self.candidate_streak = 1

            self.active_activity = top_abnormal_class
            progress_pct = int(min(100, (self.candidate_streak / self.required_streak) * 100))
            self.confirmation_progress = progress_pct
            anomaly_score = round(min(100.0, top_abnormal_p * 100.0 * (progress_pct / 100.0)), 1)

            early_confirm = (top_abnormal_p >= 0.85 and margin_over_normal >= 0.40 and self.candidate_streak >= 2)

            if self.candidate_streak >= self.required_streak or early_confirm:
                if self.current_state != "EVENT_ACTIVE":
                    self.current_state = "CONFIRMED_ACTIVITY"
            else:
                if self.current_state != "EVENT_ACTIVE":
                    self.current_state = "POSSIBLE_ACTIVITY"

        elif (normal_p >= 0.30) or (top_abnormal_p < 0.35) or is_stationary:
            self.normal_streak += 1
            self.candidate_streak = max(0, self.candidate_streak - 1)
            self.confirmation_progress = max(0, self.confirmation_progress - 40)
            anomaly_score = 0.0

            if self.normal_streak >= 2 or is_stationary:
                self.current_state = "NORMAL"
                self.active_activity = "normal"
                self.candidate_activity = None
                self.last_alert_activity = None
                self.current_event_id = None
            else:
                if self.current_state != "EVENT_ACTIVE":
                    self.current_state = "ANALYZING"

        else:
            if self.current_state != "EVENT_ACTIVE":
                self.current_state = "ANALYZING"
                self.active_activity = "normal"
                self.confirmation_progress = max(0, self.confirmation_progress - 20)
                anomaly_score = round(max(0.0, (top_abnormal_p - 0.30) * 40.0), 1)

        return {
            "state": self.current_state,
            "active_activity": self.active_activity,
            "candidate_activity": self.candidate_activity,
            "confidence": round(top_abnormal_p * 100.0 if is_abnormal_candidate else normal_p * 100.0, 1),
            "normal_prob": round(normal_p * 100.0, 1),
            "anomaly_score": anomaly_score,
            "progress": self.confirmation_progress
        }

    def can_trigger_alert(self, now_time):
        if self.current_state != "CONFIRMED_ACTIVITY":
            return False
        in_cooldown = (now_time - self.last_alert_time) < self.cooldown_seconds
        same_locked = (self.last_alert_activity == self.active_activity and (now_time - self.last_alert_time) < 25.0)
        return not in_cooldown and not same_locked

    def mark_alert_sent(self, event_id, now_time):
        self.current_event_id = event_id
        self.last_alert_time = now_time
        self.last_alert_activity = self.active_activity
        self.current_state = "EVENT_ACTIVE"

# ============================================================
# MASTER ACTIVITY DETECTOR CLASS
# ============================================================

class ActivityDetector:
    def __init__(self, model_path=None, evidence_dir="evidence", clip_length=8, image_size=112):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = CLASSES
        self.display_names = DISPLAY_NAMES
        self.clip_length = clip_length
        self.image_size = image_size
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        self.buffer = RollingFrameBuffer(clip_length=clip_length, image_size=image_size)
        self.motion_analyzer = FastMotionAnalyzer()
        self.smoother = ProbabilitySmoother(num_classes=len(CLASSES), alpha=0.35)
        self.state_manager = EventStateManager(required_streak=3, cooldown_seconds=12.0)
        self.lock = threading.Lock()

        self.process = psutil.Process()
        self.latest_latency = {
            "inference_ms": 0.0,
            "decision_ms": 0.0,
            "total_ms": 0.0,
            "fps": 0.0,
            "cpu_percent": 0.0,
            "ram_mb": 0.0,
            "gpu_device": str(self.device)
        }
        self.last_frame_timestamp = time.perf_counter()

        self.model = None
        self._load_model(model_path)

    def _load_model(self, model_path):
        target_path = Path(model_path) if model_path else Path("temporal_model.pth")
        if not target_path.exists():
            alt_path = Path("models/best_temporal_activity_model.pt")
            if alt_path.exists():
                target_path = alt_path

        if target_path.exists():
            try:
                ckpt = safe_torch_load(target_path, map_location=self.device)
                self.clip_length = ckpt.get("clip_length", 8)
                self.image_size = ckpt.get("image_size", 112)
                self.buffer = RollingFrameBuffer(clip_length=self.clip_length, image_size=self.image_size)

                self.model = SpatiotemporalActivityModel(num_classes=len(CLASSES), clip_length=self.clip_length)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.model.to(self.device)
                self.model.eval()
                print(f"[ActivityDetector] Loaded R3D-18 Spatiotemporal Model from {target_path.name} ({self.device})")
            except Exception as e:
                print(f"[ActivityDetector] Warning loading checkpoint ({e}). Initializing architecture.")

        if self.model is None:
            self.model = SpatiotemporalActivityModel(num_classes=len(CLASSES), clip_length=self.clip_length)
            self.model.to(self.device)
            self.model.eval()

    def process_frame(self, frame, location=None):
        if not self.lock.acquire(blocking=False):
            return None

        try:
            t_start = time.perf_counter()
            now_wall = time.time()

            fps = 1.0 / max(0.001, (t_start - self.last_frame_timestamp))
            self.last_frame_timestamp = t_start

            # 1. Fast Motion & Liveness Check (< 0.1 ms)
            diff_ratio, is_stationary, is_low_light, is_pitch_black = self.motion_analyzer.update_and_check(frame)

            # 2. Append to rolling buffer
            self.buffer.append(frame, diff_ratio)

            if len(self.buffer) < 3:
                t_total = (time.perf_counter() - t_start) * 1000.0
                return {
                    "success": True,
                    "activity": "Normal",
                    "activity_code": "normal",
                    "status": "NORMAL",
                    "confidence": 98.0,
                    "normal_prob": 98.0,
                    "anomaly_score": 0.0,
                    "confirmation_progress": 0,
                    "message": "Initializing temporal buffer...",
                    "alert_created": False,
                    "alert_data": None,
                    "latency": {
                        "inference_ms": 0.0,
                        "decision_ms": round(t_total, 1),
                        "total_ms": round(t_total, 1),
                        "fps": round(fps, 1),
                        "cpu_percent": 0.0,
                        "ram_mb": 0.0,
                        "gpu_device": str(self.device)
                    }
                }

            # 3. Spatiotemporal Clip Inference (1, 3, T=8, H=112, W=112)
            t_inf_start = time.perf_counter()
            clip_tensor = self.buffer.get_clip_tensor(self.device)

            with torch.inference_mode():
                logits = self.model(clip_tensor)
                raw_probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            t_inf_end = time.perf_counter()
            inf_ms = (t_inf_end - t_inf_start) * 1000.0

            # 4. Temporal Smoothing
            t_dec_start = time.perf_counter()
            smoothed_probs = self.smoother.update(raw_probs)
            motion_energy = self.buffer.get_recent_motion_energy()

            # 5. State Machine Evaluation with Motion & Liveness Guard
            state_info = self.state_manager.update(
                smoothed_probs,
                motion_energy=motion_energy,
                is_stationary=is_stationary,
                is_low_light=is_low_light,
                is_pitch_black=is_pitch_black
            )

            # 6. Alert & Evidence Generation with Unique EVENT ID
            alert_created = False
            alert_data = None

            if self.state_manager.can_trigger_alert(now_wall):
                act_code = state_info["active_activity"]
                disp_name = DISPLAY_NAMES.get(act_code, act_code.title())
                
                date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                rand_suffix = uuid.uuid4().hex[:4].upper()
                event_id = f"EVENT-{date_str}-{rand_suffix}"

                evidence_file = self._save_best_evidence_image(act_code, event_id)
                if evidence_file:
                    alert_data = {
                        "id": event_id,
                        "event_id": event_id,
                        "activity": disp_name,
                        "activity_code": act_code,
                        "confidence": state_info["confidence"],
                        "anomaly_score": round(max(60.0, state_info["confidence"]), 1),
                        "status": "CONFIRMED",
                        "time": datetime.now().strftime("%I:%M:%S %p"),
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "message": f"Confirmed {disp_name} detected ({state_info['confidence']}% confidence).",
                        "evidence": f"/evidence/{evidence_file}",
                        "location": location or {"lat": None, "lng": None}
                    }
                    self.state_manager.mark_alert_sent(event_id, now_wall)
                    alert_created = True

            t_dec_end = time.perf_counter()
            dec_ms = (t_dec_end - t_dec_start) * 1000.0
            tot_ms = (t_dec_end - t_start) * 1000.0

            try:
                mem_mb = self.process.memory_info().rss / (1024 * 1024)
                num_cpus = psutil.cpu_count() or 1
                cpu_pct = self.process.cpu_percent(interval=None) / num_cpus
            except Exception:
                mem_mb = 0.0
                cpu_pct = 0.0

            self.latest_latency = {
                "inference_ms": round(inf_ms, 1),
                "decision_ms": round(dec_ms, 1),
                "total_ms": round(tot_ms, 1),
                "fps": round(fps, 1),
                "cpu_percent": round(cpu_pct, 1),
                "ram_mb": round(mem_mb, 1),
                "gpu_device": str(self.device)
            }

            cur_st = state_info["state"]
            act_code = state_info["active_activity"]
            disp_act = DISPLAY_NAMES.get(act_code, act_code.title())

            if cur_st in {"CONFIRMED_ACTIVITY", "EVENT_ACTIVE"}:
                disp_label = disp_act
                status_tag = "CONFIRMED"
                status_msg = f"EMERGENCY ALERT: {disp_act} confirmed ({state_info['confidence']}% confidence)."
            elif cur_st == "POSSIBLE_ACTIVITY":
                disp_label = f"Possible {disp_act}"
                status_tag = "POSSIBLE"
                status_msg = f"Confirming {disp_act} ({state_info['confidence']}%)..."
            elif cur_st == "ANALYZING":
                disp_label = "Analyzing Scene"
                status_tag = "ANALYZING"
                status_msg = "Analyzing temporal dynamics. Scene stable."
            else:
                disp_label = "Normal"
                status_tag = "NORMAL"
                status_msg = "Scene is Normal. Continuous monitoring active."

            return {
                "success": True,
                "activity": disp_label,
                "activity_code": act_code,
                "status": status_tag,
                "confidence": state_info["confidence"],
                "normal_prob": state_info["normal_prob"],
                "anomaly_score": state_info["anomaly_score"],
                "confirmation_progress": state_info["progress"],
                "message": status_msg,
                "alert_created": alert_created,
                "alert_data": alert_data,
                "latency": self.latest_latency
            }

        finally:
            self.lock.release()

    def _save_best_evidence_image(self, activity_code, event_id):
        frames = self.buffer.get_original_frames()
        if not frames:
            return None
        try:
            best_frame = None
            best_sharpness = -1.0
            for f in frames:
                if f is not None and f.size > 0:
                    small_g = cv2.cvtColor(cv2.resize(f, (160, 120)), cv2.COLOR_BGR2GRAY)
                    sharpness = float(cv2.Laplacian(small_g, cv2.CV_64F).var())
                    if sharpness > best_sharpness:
                        best_sharpness = sharpness
                        best_frame = f

            if best_frame is None:
                best_frame = frames[-1]

            filename = f"{activity_code}_{event_id}.jpg"
            filepath = self.evidence_dir / filename
            cv2.imwrite(str(filepath), best_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return filename
        except Exception as err:
            print(f"[ActivityDetector] Error saving evidence image: {err}")
            return None

    def reset(self):
        with self.lock:
            self.buffer.clear()
            self.motion_analyzer.reset()
            self.smoother.reset()
            self.state_manager = EventStateManager(required_streak=3, cooldown_seconds=12.0)