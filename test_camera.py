# ============================================================
# AI UNUSUAL ACTIVITY DETECTION
# COMPLETE REPLACEMENT app.py
#
# 8 classes:
#   accident
#   falling
#   fighting
#   fire_smoke
#   normal
#   shooting
#   theft
#   vandalism
#
# IMPORTANT:
# - Normal is the default displayed activity.
# - Raw model predictions do NOT immediately create alerts.
# - Abnormal activities require temporal confirmation.
# - Falling requires a movement transition.
# - Evidence is captured only after confirmation.
# - No continuous video recording.
# - No automatic emergency dispatch.
# ============================================================

from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    send_from_directory
)

from ultralytics import YOLO

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from collections import deque
from threading import Lock
from werkzeug.utils import secure_filename


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
ALERTS_FILE = os.path.join(BASE_DIR, "alerts.json")

os.makedirs(EVIDENCE_DIR, exist_ok=True)


# ============================================================
# EXPECTED MODEL CLASSES
# ============================================================

EXPECTED_CLASSES = [
    "accident",
    "falling",
    "fighting",
    "fire_smoke",
    "normal",
    "shooting",
    "theft",
    "vandalism",
]


# ============================================================
# DISPLAY NAMES
# ============================================================

DISPLAY_NAMES = {
    "normal": "Normal",
    "falling": "Falling",
    "accident": "Accident",
    "fighting": "Fighting",
    "fire_smoke": "Fire / Smoke",
    "shooting": "Shooting",
    "theft": "Theft",
    "vandalism": "Vandalism",
}


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nERROR: best.pt was not found:\n{MODEL_PATH}\n"
    )

print("\n" + "=" * 70)
print("LOADING 8-CLASS UNUSUAL ACTIVITY MODEL")
print("=" * 70)

model = YOLO(MODEL_PATH)


# ============================================================
# VERIFY MODEL CLASSES
# ============================================================

raw_names = model.names

if isinstance(raw_names, dict):
    MODEL_CLASSES = {
        int(k): str(v).lower().strip()
        for k, v in raw_names.items()
    }
else:
    MODEL_CLASSES = {
        i: str(v).lower().strip()
        for i, v in enumerate(raw_names)
    }

print("\nMODEL CLASSES:")

for class_id in sorted(MODEL_CLASSES):
    print(f"{class_id} -> {MODEL_CLASSES[class_id]}")


missing_classes = [
    c for c in EXPECTED_CLASSES
    if c not in MODEL_CLASSES.values()
]

if missing_classes:
    raise RuntimeError(
        "\nERROR: MODEL CLASS VERIFICATION FAILED\n"
        f"Missing classes: {missing_classes}\n"
    )

print("\nSUCCESS: All 8 expected classes are available.")


# Reverse lookup
CLASS_TO_ID = {
    name: class_id
    for class_id, name in MODEL_CLASSES.items()
}


# ============================================================
# THREAD LOCK
# ============================================================

state_lock = Lock()


# ============================================================
# CAMERA / MONITORING STATE
# ============================================================

camera_started = False
frames_processed = 0

latest_location = {
    "lat": None,
    "lng": None,
    "accuracy": None,
    "updated": None
}


# ============================================================
# TEMPORAL SETTINGS
#
# These values are intentionally conservative.
# The purpose is to stop one-frame false positives.
# ============================================================

HISTORY_SIZE = 12

prediction_history = deque(maxlen=HISTORY_SIZE)

frame_motion_history = deque(maxlen=HISTORY_SIZE)

last_frame_gray = None


# ============================================================
# NORMAL PRIORITY
# ============================================================

NORMAL_MIN_PROB = 0.35

# If normal is reasonably strong, abnormal predictions need
# a significant advantage before they can be confirmed.
NORMAL_MARGIN = 0.12


# ============================================================
# GENERAL ABNORMAL SETTINGS
# ============================================================

ABNORMAL_THRESHOLDS = {
    "falling": 0.88,
    "accident": 0.88,
    "fighting": 0.88,
    "theft": 0.88,
    "vandalism": 0.90,
    "fire_smoke": 0.90,
    "shooting": 0.92,
}


# Number of recent frames that must support the activity.
REQUIRED_SUPPORT = {
    "falling": 4,
    "accident": 6,
    "fighting": 7,
    "theft": 6,
    "vandalism": 6,
    "fire_smoke": 5,
    "shooting": 6,
}


# ============================================================
# EVENT COOLDOWN
# ============================================================

ALERT_COOLDOWN = 20

last_alert_time = 0
last_alert_activity = None


# ============================================================
# FALL DETECTION STATE
# ============================================================

fall_prediction_history = deque(maxlen=12)

fall_motion_history = deque(maxlen=12)

previous_posture_like_state = None

fall_transition_detected_recently = False

fall_transition_timer = 0


# ============================================================
# MOTION SETTINGS
# ============================================================

MOTION_LOW = 2.0
MOTION_MEDIUM = 5.0
MOTION_HIGH = 9.0
MOTION_SUDDEN = 14.0


# ============================================================
# ALERT CONFIRMATION STATE
# ============================================================

candidate_activity = "normal"
candidate_count = 0

last_confirmed_activity = "normal"

confirmation_required = 1
confirmation_current = 0


# ============================================================
# FILE HELPERS
# ============================================================

def load_alerts():
    """Load saved alerts."""

    if not os.path.exists(ALERTS_FILE):
        return []

    try:
        with open(
            ALERTS_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

    except Exception as e:
        print("WARNING: Could not read alerts.json:", e)

    return []


def save_alerts(alerts):
    """Save alerts safely."""

    temp_file = ALERTS_FILE + ".tmp"

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            alerts,
            f,
            indent=2,
            ensure_ascii=False
        )

    os.replace(temp_file, ALERTS_FILE)


# ============================================================
# LOCATION FORMAT
# ============================================================

def get_location_text():
    lat = latest_location.get("lat")
    lng = latest_location.get("lng")

    if lat is None or lng is None:
        return "Location unavailable"

    return f"{lat:.6f}, {lng:.6f}"


# ============================================================
# ACTIVITY CATEGORY
# ============================================================

def get_event_status(activity):
    """
    Classify event type for notification/history display.
    """

    if activity in {
        "falling",
        "accident"
    }:
        return "SAFETY"

    if activity in {
        "fighting",
        "theft",
        "vandalism",
        "shooting"
    }:
        return "SECURITY"

    if activity == "fire_smoke":
        return "FIRE / SMOKE"

    return "NORMAL"


# ============================================================
# MOTION CALCULATION
# ============================================================

def calculate_motion(frame):
    """
    Calculate a simple frame-to-frame motion score.

    This is NOT used alone to declare an abnormal event.
    It is only supporting evidence.
    """

    global last_frame_gray

    try:
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.resize(
            gray,
            (320, 240)
        )

        if last_frame_gray is None:
            last_frame_gray = gray
            return 0.0

        difference = cv2.absdiff(
            gray,
            last_frame_gray
        )

        motion = float(
            np.mean(difference)
        )

        last_frame_gray = gray

        return motion

    except Exception:
        return 0.0


# ============================================================
# RAW MODEL PREDICTION
# ============================================================

def get_raw_prediction(frame):
    """
    Run the YOLO classification model.

    Returns:
        top_class
        top_confidence
        all_probabilities
    """

    results = model.predict(
        source=frame,
        verbose=False
    )

    result = results[0]

    if result.probs is None:
        return (
            "normal",
            0.0,
            {
                c: 0.0
                for c in EXPECTED_CLASSES
            }
        )

    probabilities = (
        result.probs.data
        .detach()
        .cpu()
        .numpy()
    )

    all_probs = {}

    for class_id, class_name in MODEL_CLASSES.items():

        if class_id < len(probabilities):
            all_probs[class_name] = float(
                probabilities[class_id]
            )
        else:
            all_probs[class_name] = 0.0

    ranked = sorted(
        all_probs.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_class = ranked[0][0]
    top_confidence = ranked[0][1]

    return (
        top_class,
        top_confidence,
        all_probs
    )


# ============================================================
# NORMAL PRIORITY FILTER
# ============================================================

def normal_is_dominant(all_probs):
    """
    Decide whether the current frame should be treated
    as normal.

    This prevents weak abnormal predictions from becoming
    the displayed activity.
    """

    normal_prob = all_probs.get(
        "normal",
        0.0
    )

    abnormal_values = [
        all_probs.get(c, 0.0)
        for c in ABNORMAL_THRESHOLDS
    ]

    highest_abnormal = (
        max(abnormal_values)
        if abnormal_values
        else 0.0
    )

    # Strong normal evidence.
    if normal_prob >= 0.50:
        return True

    # Normal has a meaningful advantage.
    if normal_prob >= (
        highest_abnormal - NORMAL_MARGIN
    ):
        return True

    return False


# ============================================================
# TEMPORAL SUPPORT
# ============================================================

def calculate_support(activity):
    """
    Calculate how many recent frames support an activity.
    """

    if not prediction_history:
        return 0

    count = 0

    threshold = ABNORMAL_THRESHOLDS.get(
        activity,
        1.0
    )

    for item in prediction_history:

        if item["activity"] != activity:
            continue

        if item["confidence"] >= threshold:
            count += 1

    return count


# ============================================================
# AVERAGE CONFIDENCE
# ============================================================

def average_confidence(activity):
    values = []

    for item in prediction_history:

        if item["activity"] == activity:
            values.append(
                item["confidence"]
            )

    if not values:
        return 0.0

    return float(
        sum(values) / len(values)
    )


# ============================================================
# FALL TRANSITION DETECTION
# ============================================================

def detect_fall_transition(
    falling_probability,
    motion
):
    """
    Detect a possible fall using a transition rather than
    simply detecting a person lying down.

    Normal sleeping/lying should therefore not automatically
    become Falling.
    """

    global fall_transition_detected_recently
    global fall_transition_timer

    fall_prediction_history.append(
        falling_probability
    )

    fall_motion_history.append(
        motion
    )

    # --------------------------------------------------------
    # Strong sudden movement
    # --------------------------------------------------------

    sudden_motion = (
        motion >= MOTION_SUDDEN
    )

    medium_motion = (
        motion >= MOTION_MEDIUM
    )

    # --------------------------------------------------------
    # Recent falling predictions
    # --------------------------------------------------------

    strong_fall_predictions = sum(
        1
        for value in fall_prediction_history
        if value >= 0.88
    )

    # --------------------------------------------------------
    # Check recent motion burst
    # --------------------------------------------------------

    recent_motion = list(
        fall_motion_history
    )[-6:]

    high_motion_frames = sum(
        1
        for value in recent_motion
        if value >= MOTION_HIGH
    )

    # --------------------------------------------------------
    # Transition condition
    # --------------------------------------------------------

    if (
        sudden_motion
        and strong_fall_predictions >= 2
    ):
        fall_transition_detected_recently = True
        fall_transition_timer = 12

    elif (
        high_motion_frames >= 2
        and strong_fall_predictions >= 3
    ):
        fall_transition_detected_recently = True
        fall_transition_timer = 12

    # --------------------------------------------------------
    # Keep transition alive temporarily
    # --------------------------------------------------------

    if fall_transition_timer > 0:

        fall_transition_timer -= 1

        if (
            strong_fall_predictions >= 2
            and (
                medium_motion
                or fall_transition_detected_recently
            )
        ):
            return True

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    if fall_transition_timer <= 0:
        fall_transition_detected_recently = False

    return False


# ============================================================
# ACTIVITY-SPECIFIC VALIDATION
# ============================================================

def validate_activity(
    activity,
    confidence,
    all_probs,
    motion
):
    """
    This is the main anti-false-positive system.
    """

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    if activity == "normal":
        return True

    normal_prob = all_probs.get(
        "normal",
        0.0
    )

    # --------------------------------------------------------
    # Never accept weak abnormal confidence.
    # --------------------------------------------------------

    threshold = ABNORMAL_THRESHOLDS.get(
        activity,
        0.90
    )

    if confidence < threshold:
        return False

    # --------------------------------------------------------
    # Normal must not be close to abnormal.
    # --------------------------------------------------------

    if (
        normal_prob >= 0.35
        and confidence <
        normal_prob + NORMAL_MARGIN
    ):
        return False

    # --------------------------------------------------------
    # Temporal support
    # --------------------------------------------------------

    support = calculate_support(
        activity
    )

    required = REQUIRED_SUPPORT.get(
        activity,
        6
    )

    if support < required:
        return False

    # --------------------------------------------------------
    # Average confidence
    # --------------------------------------------------------

    avg_conf = average_confidence(
        activity
    )

    if avg_conf < threshold:
        return False

    # --------------------------------------------------------
    # FALLING
    # --------------------------------------------------------

    if activity == "falling":

        transition = detect_fall_transition(
            all_probs.get(
                "falling",
                0.0
            ),
            motion
        )

        if not transition:
            return False

        return True

    # --------------------------------------------------------
    # FIGHTING
    #
    # Requires sustained movement.
    # Movement alone does NOT trigger fighting.
    # --------------------------------------------------------

    if activity == "fighting":

        recent_motion = list(
            frame_motion_history
        )[-8:]

        motion_frames = sum(
            1
            for value in recent_motion
            if value >= MOTION_MEDIUM
        )

        if motion_frames < 3:
            return False

        return True

    # --------------------------------------------------------
    # ACCIDENT
    #
    # Requires stronger motion.
    # --------------------------------------------------------

    if activity == "accident":

        recent_motion = list(
            frame_motion_history
        )[-8:]

        motion_frames = sum(
            1
            for value in recent_motion
            if value >= MOTION_HIGH
        )

        if motion_frames < 2:
            return False

        return True

    # --------------------------------------------------------
    # THEFT
    #
    # Does not require motion because a theft-related scene
    # may become visually static.
    # --------------------------------------------------------

    if activity == "theft":
        return True

    # --------------------------------------------------------
    # VANDALISM
    # --------------------------------------------------------

    if activity == "vandalism":
        return True

    # --------------------------------------------------------
    # FIRE / SMOKE
    # --------------------------------------------------------

    if activity == "fire_smoke":
        return True

    # --------------------------------------------------------
    # SHOOTING
    # --------------------------------------------------------

    if activity == "shooting":
        return True

    return False


# ============================================================
# CONFIRMATION ENGINE
# ============================================================

def update_confirmation(
    raw_activity,
    raw_confidence,
    all_probs,
    motion
):
    """
    Update the temporal confirmation state.

    Returns:

        display_activity
        confirmed_activity
        confirmation_current
        confirmation_required
        message
    """

    global candidate_activity
    global candidate_count
    global last_confirmed_activity
    global confirmation_required
    global confirmation_current

    # --------------------------------------------------------
    # If Normal is dominant, immediately show Normal.
    # --------------------------------------------------------

    if normal_is_dominant(
        all_probs
    ):

        candidate_activity = "normal"
        candidate_count = 0

        confirmation_current = 0
        confirmation_required = 1

        last_confirmed_activity = "normal"

        return (
            "normal",
            None,
            0,
            1,
            "Normal activity."
        )

    # --------------------------------------------------------
    # Weak abnormal prediction:
    # Do not display it as confirmed.
    # --------------------------------------------------------

    threshold = ABNORMAL_THRESHOLDS.get(
        raw_activity,
        1.0
    )

    if (
        raw_activity == "normal"
        or raw_confidence < threshold
    ):

        candidate_activity = "normal"
        candidate_count = 0

        confirmation_current = 0
        confirmation_required = 1

        last_confirmed_activity = "normal"

        return (
            "normal",
            None,
            0,
            1,
            "Normal activity."
        )

    # --------------------------------------------------------
    # Activity changed.
    # --------------------------------------------------------

    if raw_activity != candidate_activity:

        candidate_activity = raw_activity
        candidate_count = 1

    else:
        candidate_count += 1

    # --------------------------------------------------------
    # Activity-specific required frames.
    # --------------------------------------------------------

    required = REQUIRED_SUPPORT.get(
        raw_activity,
        6
    )

    # Falling has a separate transition condition.
    if raw_activity == "falling":
        required = 4

    confirmation_required = required
    confirmation_current = min(
        candidate_count,
        required
    )

    # --------------------------------------------------------
    # Validate current activity.
    # --------------------------------------------------------

    valid = validate_activity(
        raw_activity,
        raw_confidence,
        all_probs,
        motion
    )

    if not valid:

        # Keep UI safe.
        last_confirmed_activity = "normal"

        return (
            "normal",
            None,
            confirmation_current,
            required,
            f"Possible {DISPLAY_NAMES.get(raw_activity, raw_activity)} - confirming..."
        )

    # --------------------------------------------------------
    # Confirm only after enough consecutive frames.
    # --------------------------------------------------------

    if candidate_count >= required:

        last_confirmed_activity = raw_activity

        return (
            raw_activity,
            raw_activity,
            confirmation_current,
            required,
            f"{DISPLAY_NAMES.get(raw_activity, raw_activity)} detected and confirmed."
        )

    # --------------------------------------------------------
    # Still confirming.
    # --------------------------------------------------------

    last_confirmed_activity = "normal"

    return (
        "normal",
        None,
        confirmation_current,
        required,
        f"Possible {DISPLAY_NAMES.get(raw_activity, raw_activity)} - confirming..."
    )


# ============================================================
# SAVE EVIDENCE
# ============================================================

def save_evidence(
    frame,
    activity
):
    """
    Save exactly one JPG for a confirmed event.
    """

    timestamp = datetime.now()

    safe_activity = secure_filename(
        activity
    ).replace(
        "_",
        "_"
    )

    filename = (
        f"{safe_activity}_"
        f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    )

    filepath = os.path.join(
        EVIDENCE_DIR,
        filename
    )

    success = cv2.imwrite(
        filepath,
        frame
    )

    if not success:
        print(
            "ERROR: Could not save evidence:",
            filepath
        )
        return None

    return filename


# ============================================================
# CREATE ALERT
# ============================================================

def create_alert(
    activity,
    confidence,
    anomaly_score,
    message,
    frame
):
    """
    Create one alert and one evidence image.
    """

    global last_alert_time
    global last_alert_activity

    now = time.time()

    # --------------------------------------------------------
    # Cooldown
    # --------------------------------------------------------

    if (
        now - last_alert_time
        < ALERT_COOLDOWN
    ):
        return None

    # --------------------------------------------------------
    # Do not create repeated same activity alerts.
    # --------------------------------------------------------

    if (
        last_alert_activity == activity
        and (
            now - last_alert_time
            < ALERT_COOLDOWN
        )
    ):
        return None

    # --------------------------------------------------------
    # Save evidence
    # --------------------------------------------------------

    evidence_filename = save_evidence(
        frame,
        activity
    )

    if evidence_filename is None:
        return None

    timestamp = datetime.now()

    confidence_percent = round(
        confidence * 100,
        2
    )

    anomaly_percent = round(
        anomaly_score * 100,
        2
    )

    alert = {
        "id": int(
            timestamp.timestamp() * 1000
        ),

        "activity": DISPLAY_NAMES.get(
            activity,
            activity
        ),

        "activity_key": activity,

        "confidence": confidence_percent,

        "anomaly_score": anomaly_percent,

        "status": get_event_status(
            activity
        ),

        "message": message,

        "location": get_location_text(),

        "latitude": latest_location.get(
            "lat"
        ),

        "longitude": latest_location.get(
            "lng"
        ),

        "accuracy": latest_location.get(
            "accuracy"
        ),

        "time": timestamp.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "evidence": (
            f"/evidence/"
            f"{evidence_filename}"
        ),

        "notification": {
            "type": get_event_status(
                activity
            ),
            "location": get_location_text()
        }
    }

    # --------------------------------------------------------
    # Save alert
    # --------------------------------------------------------

    with state_lock:

        alerts = load_alerts()

        # Newest alert first.
        alerts.insert(
            0,
            alert
        )

        # Keep history manageable.
        alerts = alerts[:100]

        save_alerts(
            alerts
        )

    last_alert_time = now
    last_alert_activity = activity

    print("\n" + "=" * 70)
    print("CONFIRMED ALERT CREATED")
    print("=" * 70)
    print(
        "Activity:",
        alert["activity"]
    )
    print(
        "Confidence:",
        alert["confidence"],
        "%"
    )
    print(
        "Status:",
        alert["status"]
    )
    print(
        "Location:",
        alert["location"]
    )
    print(
        "Evidence:",
        alert["evidence"]
    )
    print("=" * 70)

    return alert


# ============================================================
# ROOT PAGE
# ============================================================

@app.route("/")
def index():
    return render_template(
        "index.html"
    )


# ============================================================
# EVIDENCE FILE ROUTE
# ============================================================

@app.route(
    "/evidence/<path:filename>"
)
def evidence_file(filename):

    safe_filename = secure_filename(
        os.path.basename(filename)
    )

    return send_from_directory(
        EVIDENCE_DIR,
        safe_filename
    )


# ============================================================
# STATUS API
# ============================================================

@app.route(
    "/api/status",
    methods=["GET"]
)
def api_status():

    with state_lock:

        alerts = load_alerts()

        return jsonify({
            "success": True,

            "camera": (
                "Online"
                if camera_started
                else "Offline"
            ),

            "ai": "Ready",

            "model": "8-Class YOLO",

            "frames": frames_processed,

            "monitoring": True,

            "alerts": len(alerts),

            "activity": DISPLAY_NAMES.get(
                last_confirmed_activity,
                "Normal"
            ),

            "display_name": DISPLAY_NAMES.get(
                last_confirmed_activity,
                "Normal"
            )
        })


# ============================================================
# CAMERA START
# ============================================================

@app.route(
    "/api/camera/start",
    methods=["POST"]
)
def camera_start():

    global camera_started

    camera_started = True

    return jsonify({
        "success": True,
        "camera": "Online",
        "message": "Camera monitoring started."
    })


# ============================================================
# CAMERA STOP
# ============================================================

@app.route(
    "/api/camera/stop",
    methods=["POST"]
)
def camera_stop():

    global camera_started

    camera_started = False

    return jsonify({
        "success": True,
        "camera": "Offline",
        "message": "Camera monitoring stopped."
    })


# ============================================================
# LOCATION API
# ============================================================

@app.route(
    "/api/location",
    methods=["POST"]
)
def update_location():

    global latest_location

    try:

        data = request.get_json(
            silent=True
        ) or {}

        lat = data.get(
            "lat",
            data.get("latitude")
        )

        lng = data.get(
            "lng",
            data.get("longitude")
        )

        accuracy = data.get(
            "accuracy"
        )

        if lat is not None:
            latest_location["lat"] = float(
                lat
            )

        if lng is not None:
            latest_location["lng"] = float(
                lng
            )

        if accuracy is not None:
            latest_location["accuracy"] = float(
                accuracy
            )

        latest_location["updated"] = (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        return jsonify({
            "success": True,
            "location": latest_location
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


# ============================================================
# FRAME API
# ============================================================

@app.route(
    "/api/frame",
    methods=["POST"]
)
def process_frame():

    global frames_processed

    try:

        # ----------------------------------------------------
        # Accept raw JPEG sent by current index.html.
        # ----------------------------------------------------

        raw_data = request.get_data()

        if not raw_data:
            return jsonify({
                "success": False,
                "error": "No image data received."
            }), 400

        image_array = np.frombuffer(
            raw_data,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return jsonify({
                "success": False,
                "error": "Could not decode image."
            }), 400

        frames_processed += 1

        # ----------------------------------------------------
        # Motion
        # ----------------------------------------------------

        motion = calculate_motion(
            frame
        )

        frame_motion_history.append(
            motion
        )

        # ----------------------------------------------------
        # Raw AI prediction
        # ----------------------------------------------------

        (
            raw_activity,
            raw_confidence,
            all_probs
        ) = get_raw_prediction(
            frame
        )

        # ----------------------------------------------------
        # Store prediction
        # ----------------------------------------------------

        prediction_history.append({
            "activity": raw_activity,
            "confidence": raw_confidence,
            "time": time.time()
        })

        # ----------------------------------------------------
        # Update temporal confirmation
        # ----------------------------------------------------

        (
            display_activity,
            confirmed_activity,
            confirmation_current,
            confirmation_required_value,
            message
        ) = update_confirmation(
            raw_activity,
            raw_confidence,
            all_probs,
            motion
        )

        # ----------------------------------------------------
        # Normal probability
        # ----------------------------------------------------

        normal_probability = all_probs.get(
            "normal",
            0.0
        )

        # ----------------------------------------------------
        # Anomaly score
        #
        # This is informational. It does not itself create
        # an alert.
        # ----------------------------------------------------

        anomaly_score = max(
            [
                all_probs.get(
                    activity,
                    0.0
                )
                for activity in ABNORMAL_THRESHOLDS
            ]
            or [0.0]
        )

        # ----------------------------------------------------
        # Is currently abnormal?
        # ----------------------------------------------------

        is_abnormal = (
            confirmed_activity is not None
        )

        # ----------------------------------------------------
        # Alert creation
        # ----------------------------------------------------

        alert_created = False
        alert_data = None

        if confirmed_activity is not None:

            alert_data = create_alert(
                confirmed_activity,
                all_probs.get(
                    confirmed_activity,
                    raw_confidence
                ),
                anomaly_score,
                message,
                frame
            )

            if alert_data is not None:
                alert_created = True

        # ----------------------------------------------------
        # Confirmation response
        # ----------------------------------------------------

        response = {

            "success": True,

            "frames": frames_processed,

            # ------------------------------------------------
            # SAFE DISPLAY ACTIVITY
            # ------------------------------------------------

            "activity": (
                DISPLAY_NAMES.get(
                    display_activity,
                    "Normal"
                )
            ),

            "display_name": (
                DISPLAY_NAMES.get(
                    display_activity,
                    "Normal"
                )
            ),

            # ------------------------------------------------
            # Raw prediction
            # ------------------------------------------------

            "raw_activity": DISPLAY_NAMES.get(
                raw_activity,
                raw_activity
            ),

            "raw_activity_key": raw_activity,

            "confidence": round(
                raw_confidence * 100,
                2
            ),

            "normal_prob": round(
                normal_probability * 100,
                2
            ),

            "anomaly_score": round(
                anomaly_score * 100,
                2
            ),

            # ------------------------------------------------
            # State
            # ------------------------------------------------

            "status": (
                "ABNORMAL"
                if is_abnormal
                else "NORMAL"
            ),

            "is_abnormal": is_abnormal,

            "confirmed": (
                confirmed_activity
                is not None
            ),

            "alert_created": alert_created,

            "alert_data": alert_data,

            # ------------------------------------------------
            # Confirmation
            # ------------------------------------------------

            "confirmation": {
                "current": confirmation_current,
                "required": confirmation_required_value
            },

            # ------------------------------------------------
            # Message
            # ------------------------------------------------

            "message": message,

            # ------------------------------------------------
            # Motion
            # ------------------------------------------------

            "motion": round(
                motion,
                2
            ),

            # ------------------------------------------------
            # All raw probabilities
            # ------------------------------------------------

            "probabilities": {
                DISPLAY_NAMES.get(
                    name,
                    name
                ): round(
                    value * 100,
                    2
                )
                for name, value
                in all_probs.items()
            }
        }

        return jsonify(
            response
        )

    except Exception as e:

        print(
            "\nFRAME PROCESSING ERROR:",
            repr(e)
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# ALERT HISTORY API
# ============================================================

@app.route(
    "/api/alerts",
    methods=["GET"]
)
def get_alerts():

    with state_lock:

        alerts = load_alerts()

    return jsonify({
        "success": True,
        "alerts": alerts
    })


# ============================================================
# CLEAR ALERT HISTORY
#
# IMPORTANT:
# This clears alert messages/history ONLY.
# Evidence JPG files are NOT deleted.
# ============================================================

@app.route(
    "/api/alerts/clear",
    methods=["DELETE", "POST"]
)
def clear_alerts():

    with state_lock:

        save_alerts([])

    print(
        "\nAlert history cleared."
    )

    return jsonify({
        "success": True,
        "message": "Alert history cleared. Evidence images were kept."
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({
        "success": True,
        "status": "healthy",
        "model": "8-class YOLO",
        "classes": EXPECTED_CLASSES,
        "model_path": MODEL_PATH
    })


# ============================================================
# SERVER START
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("AI UNUSUAL ACTIVITY DETECTION")
    print("=" * 70)

    print("\nModel:")
    print(MODEL_PATH)

    print("\nClasses:")
    for class_id in sorted(MODEL_CLASSES):
        print(
            f"{class_id} -> "
            f"{MODEL_CLASSES[class_id]}"
        )

    print("\n" + "=" * 70)
    print("SERVER READY")
    print("=" * 70)

    print("\nLocal:")
    print("http://127.0.0.1:5000")

    print("\nNetwork:")
    print("http://0.0.0.0:5000")

    print("\nOpen the website and allow camera permission.")

    print("\n" + "=" * 70)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )