"""
Comprehensive Hard-Negative & 8-Class Verification Test Suite (TEST 1 to TEST 10)
=================================================================================
Project: Unusual Activity Detection Using Camera by AI

Tests Performed:
TEST 1: Normal sleeping/lying footage (Expected: Normal)
TEST 2: Normal sitting/walking/talking footage (Expected: Normal)
TEST 3: Normal group movement (Expected: Normal)
TEST 4: Actual falling video (Expected: Falling)
TEST 5: Actual fighting video (Expected: Fighting)
TEST 6: Actual theft-related video (Expected: Theft)
TEST 7: Actual vandalism video (Expected: Vandalism)
TEST 8: Actual accident video (Expected: Accident)
TEST 9: Actual fire/smoke video (Expected: Fire_Smoke)
TEST 10: Actual shooting video (Expected: Shooting)

Outputs for every test:
- VIDEO NAME
- GROUND TRUTH
- PREDICTED CLASS
- CONFIDENCE
- TOP-3 PREDICTIONS
- RESULT = PASS/FAIL
"""

import os
import cv2
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

from activity_detector import ActivityDetector, CLASSES, DISPLAY_NAMES

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
SPLIT_FILE = BASE_DIR / "dataset_split.json"

def evaluate_video_stream(detector, video_path, stride=2, max_frames=600):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    detector.reset()
    latencies = []
    results = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx > max_frames:
            break

        if frame_idx % stride == 0:
            t0 = time.perf_counter()
            res = detector.process_frame(frame)
            t_ms = (time.perf_counter() - t0) * 1000.0
            if res:
                latencies.append(t_ms)
                results.append(res)
        frame_idx += 1

    cap.release()

    if not results:
        return None

    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    confirmed_steps = [r for r in results if r["status"] == "CONFIRMED"]
    last_res = results[-1]

    if len(confirmed_steps) >= 2 and last_res["anomaly_score"] > 25.0:
        confirmed_codes = [r["activity_code"] for r in confirmed_steps]
        detected_code = Counter(confirmed_codes).most_common(1)[0][0]
        conf_scores = [r["confidence"] for r in confirmed_steps if r["activity_code"] == detected_code]
        avg_conf = float(np.mean(conf_scores)) if conf_scores else last_res["confidence"]
        status = "CONFIRMED"
    else:
        all_codes = [r["activity_code"] for r in results]
        code_counts = Counter(all_codes)
        if code_counts.get("normal", 0) >= len(results) * 0.35 or last_res["normal_prob"] >= 45.0:
            detected_code = "normal"
            avg_conf = last_res["normal_prob"]
            status = "NORMAL"
        else:
            detected_code = Counter(all_codes).most_common(1)[0][0]
            avg_conf = last_res["confidence"]
            status = last_res["status"]

    detected_display = DISPLAY_NAMES.get(detected_code, detected_code.title())

    # Build Top-3 summary
    counts = Counter([r["activity_code"] for r in results])
    top3_list = []
    for c, cnt in counts.most_common(3):
        pct = round((cnt / max(1, len(results))) * 100.0, 1)
        top3_list.append(f"{DISPLAY_NAMES.get(c, c)} ({pct}%)")
    top3_str = ", ".join(top3_list)

    return {
        "video": Path(video_path).name,
        "prediction": detected_code,
        "display": detected_display,
        "confidence": avg_conf,
        "top3": top3_str,
        "avg_latency_ms": avg_latency,
        "status": status,
        "anomaly_score": last_res["anomaly_score"],
        "normal_prob": last_res["normal_prob"]
    }


def run_comprehensive_tests(detector):
    print("=" * 80)
    print("RUNNING COMPREHENSIVE 10-TEST VERIFICATION SUITE")
    print("=" * 80)

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split_dict = json.load(f)

    # Test cases mapping
    tests_plan = [
        ("TEST 1", "Normal sleeping/lying footage", "normal", "B_N_70_resized.mp4"),
        ("TEST 2", "Normal sitting/walking footage", "normal", "B_M_87_resized.mp4"),
        ("TEST 3", "Normal group movement/crowd", "normal", "Normal_Videos_831_x264.mp4"),
        ("TEST 4", "Actual falling video", "falling", split_dict["test"].get("falling", ["B_D_0001.mp4"])[0]),
        ("TEST 5", "Actual fighting video", "fighting", split_dict["test"].get("fighting", ["Abuse001_x264.mp4"])[0]),
        ("TEST 6", "Actual theft-related video", "theft", split_dict["test"].get("theft", ["Burglary001_x264.mp4"])[0]),
        ("TEST 7", "Actual vandalism video", "vandalism", split_dict["test"].get("vandalism", ["Vandalism001_x264.mp4"])[0]),
        ("TEST 8", "Actual accident video", "accident", split_dict["test"].get("accident", ["RoadAccidents001_x264.mp4"])[0]),
        ("TEST 9", "Actual fire/smoke video", "fire_smoke", split_dict["test"].get("fire_smoke", ["Arson001_x264.mp4"])[0]),
        ("TEST 10", "Actual shooting-class video", "shooting", split_dict["test"].get("shooting", ["Shooting001_x264.mp4"])[0]),
    ]

    summary_records = []

    for test_id, description, ground_truth, vname in tests_plan:
        cdir = DATASETS_DIR / ground_truth
        vpath = cdir / vname
        if not vpath.exists():
            # Fallback to first video in folder
            vids = list(cdir.glob("*.mp4")) + list(cdir.glob("*.avi"))
            if vids:
                vpath = vids[0]
                vname = vpath.name

        print(f"\n--- {test_id}: {description} ---")
        res = evaluate_video_stream(detector, str(vpath), max_frames=350)
        if res:
            is_pass = (res["prediction"].lower() == ground_truth.lower())
            result_tag = "PASS" if is_pass else "FAIL"

            print(f"VIDEO NAME        : {res['video']}")
            print(f"GROUND TRUTH      : {DISPLAY_NAMES.get(ground_truth, ground_truth)}")
            print(f"PREDICTED CLASS   : {res['display']}")
            print(f"CONFIDENCE        : {res['confidence']:.1f}%")
            print(f"TOP-3 PREDICTIONS : {res['top3']}")
            print(f"RESULT            : {result_tag}")

            summary_records.append({
                "test": test_id,
                "desc": description,
                "video": res["video"],
                "ground_truth": ground_truth,
                "prediction": res["prediction"],
                "display": res["display"],
                "confidence": res["confidence"],
                "top3": res["top3"],
                "pass": is_pass
            })
        else:
            print(f"Failed to read video: {vname}")

    print("\n" + "=" * 80)
    print(f"{'Test':<8} | {'Ground Truth':<12} | {'Predicted':<12} | {'Conf':<8} | {'Result'}")
    print("-" * 80)
    passed_count = sum(1 for s in summary_records if s["pass"])
    for s in summary_records:
        tag = "[PASS]" if s["pass"] else "[FAIL]"
        print(f"{s['test']:<8} | {s['ground_truth']:<12} | {s['display']:<12} | {s['confidence']:>5.1f}%  | {tag}")
    print("=" * 80)
    print(f"Total Passed: {passed_count}/{len(summary_records)} ({passed_count/max(1, len(summary_records))*100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 10-Test Activity Verification Suite")
    parser.add_argument("--model", type=str, default="temporal_model.pth")
    parser.add_argument("--video", type=str, default=None)
    args = parser.parse_args()

    detector = ActivityDetector(model_path=args.model, clip_length=8, image_size=112)

    if args.video:
        res = evaluate_video_stream(detector, args.video)
        if res:
            print(res)
    else:
        run_comprehensive_tests(detector)
