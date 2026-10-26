
from flask import Flask, Response, jsonify, request, send_from_directory
from activity_detector import ActivityDetector
from email_alert import send_activity_email

from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import threading
import json
import time


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "temporal_model.pth"
EVIDENCE_DIR = BASE_DIR / "evidence"
ALERT_FILE = BASE_DIR / "alerts.json"

EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# NEW NETWORK SETTINGS
# ============================================================

LAPTOP_IP = "192.168.1.115"
PORT = 5000

SSL_CERT = BASE_DIR / "192.168.1.115.pem"
SSL_KEY = BASE_DIR / "192.168.1.115-key.pem"


# ============================================================
# AUTOMATIC GMAIL ALERT SETTINGS
# ============================================================

EMAIL_COOLDOWN_SECONDS = 120

ALERT_ACTIVITIES = {
    "accident",
    "falling",
    "fighting",
    "fire_smoke",
    "shooting",
    "theft",
    "vandalism"
}

email_lock = threading.Lock()
last_email_time = {}
emailed_event_ids = set()


# ============================================================
# CAMERA STATE
# ============================================================

latest_frame = None
frame_lock = threading.Lock()

mobile_online = False
last_frame_time = None

state_lock = threading.Lock()

latest_state = {
    "success": True,
    "activity": "Normal",
    "activity_code": "normal",
    "status": "NORMAL",
    "confidence": 0.0,
    "normal_prob": 100.0,
    "anomaly_score": 0.0,
    "confirmation_progress": 0,
    "temporal_buffer": 0,
    "temporal_required": 8,
    "message": "Waiting for mobile camera...",
    "event_active": False,
    "event_id": None,
    "alert_created": False,
    "alert_data": None,
    "latency": {},
    "risk_level": 0.0,
    "risk_stage": "Waiting",
    "mobile_online": False
}


# ============================================================
# ALERT HISTORY
# ============================================================

alert_lock = threading.Lock()


def load_alerts():
    if not ALERT_FILE.exists():
        return []

    try:
        with open(ALERT_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

    except Exception as error:
        print("Could not load alerts.json:", error)

    return []


alerts = load_alerts()


def save_alerts():
    try:
        with open(ALERT_FILE, "w", encoding="utf-8") as file:
            json.dump(alerts, file, indent=4)

    except Exception as error:
        print("Could not save alerts.json:", error)


# ============================================================
# LOAD R3D-18 MODEL
# ============================================================

print()
print("=" * 70)
print("UNUSUAL ACTIVITY DETECTION USING CAMERA BY AI")
print("=" * 70)

print("Model path:", MODEL_PATH)
print("Laptop IP:", LAPTOP_IP)
print("Loading R3D-18 temporal model...")

detector = ActivityDetector(
    model_path=str(MODEL_PATH),
    evidence_dir=str(EVIDENCE_DIR),
    clip_length=8,
    image_size=112
)

print("R3D-18 model loaded successfully.")


# ============================================================
# RISK CALCULATION
# ============================================================

def calculate_risk(result):
    activity_code = result.get("activity_code", "normal")
    status = result.get("status", "NORMAL")

    anomaly_score = float(
        result.get("anomaly_score", 0.0) or 0.0
    )

    if activity_code == "normal" or status == "NORMAL":
        return 0.0

    risk = anomaly_score / 10.0
    risk = max(0.0, min(risk, 10.0))

    return round(risk, 1)


def get_risk_stage(risk):
    if risk <= 2:
        return "Early indication"

    if risk <= 5:
        return "Starting stage"

    if risk <= 8:
        return "Moderate severity"

    return "Severe / High"


# ============================================================
# AUTOMATIC GMAIL ALERTS
# ============================================================

def find_evidence_path(alert_data):
    """
    Find the saved evidence image from ActivityDetector.
    """

    possible_keys = [
        "evidence_path",
        "evidence_file",
        "evidence",
        "image_path",
        "snapshot_path"
    ]

    for key in possible_keys:
        value = alert_data.get(key)

        if not value or not isinstance(value, str):
            continue

        value = value.split("?", 1)[0]

        if value.startswith("/evidence/"):
            value = value[len("/evidence/"):]

        candidate = Path(value)

        if not candidate.is_absolute():
            candidate = EVIDENCE_DIR / candidate.name

        try:
            candidate = candidate.resolve()
            evidence_root = EVIDENCE_DIR.resolve()

            if (
                candidate.is_file()
                and candidate.is_relative_to(evidence_root)
                and candidate.suffix.lower()
                in {".jpg", ".jpeg", ".png"}
            ):
                return str(candidate)

        except (OSError, ValueError):
            continue

    return None


def queue_email_alert(alert_data, result):
    """
    Send a Gmail alert only for a new confirmed event.
    """

    activity = str(
        alert_data.get(
            "activity_code",
            result.get("activity_code", "normal")
        )
    ).strip().lower().replace(" ", "_")

    if activity not in ALERT_ACTIVITIES:
        return

    event_id = str(
        alert_data.get("event_id")
        or result.get("event_id")
        or ""
    )

    evidence_path = find_evidence_path(alert_data)

    if not evidence_path:
        print(
            "[GMAIL] Evidence image not found. "
            "Email skipped."
        )
        return

    confidence = float(
        alert_data.get(
            "confidence",
            result.get("confidence", 0.0)
        ) or 0.0
    )

    risk = float(
        result.get("risk_level", 0.0) or 0.0
    )

    now = time.monotonic()

    with email_lock:
        if event_id and event_id in emailed_event_ids:
            return

        last_time = last_email_time.get(activity)

        if last_time is not None:
            if now - last_time < EMAIL_COOLDOWN_SECONDS:
                print(
                    "[GMAIL] Cooldown active for:",
                    activity
                )
                return

        last_email_time[activity] = now

        if event_id:
            emailed_event_ids.add(event_id)

    def send_in_background():
        try:
            print()
            print("=" * 60)
            print("AUTOMATIC GMAIL ALERT")
            print("Activity:", activity)
            print("Confidence:", confidence)
            print("Risk:", risk)
            print("Evidence:", evidence_path)
            print("=" * 60)

            sent = send_activity_email(
                activity=activity,
                confidence=confidence,
                risk=risk,
                evidence_path=evidence_path,
                location="Mobile Camera"
            )

            if sent:
                print("[GMAIL] Email sent successfully.")
            else:
                print("[GMAIL] Email sending failed.")

        except Exception as error:
            print("[GMAIL] Error:", error)

    thread = threading.Thread(
        target=send_in_background,
        daemon=True
    )

    thread.start()


# ============================================================
# MOBILE CAMERA PAGE
# ============================================================

MOBILE_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>Mobile Camera</title>

<style>
body {
    margin: 0;
    background: #111;
    color: white;
    font-family: Arial, sans-serif;
    text-align: center;
}

h2 {
    margin-top: 20px;
}

video {
    width: 95%;
    max-width: 800px;
    border-radius: 12px;
    background: black;
    margin-top: 10px;
}

#status {
    margin: 15px;
    padding: 12px;
    background: #222;
    border-radius: 8px;
}
</style>
</head>

<body>

<h2>Mobile Camera</h2>

<video id="video"
       autoplay
       playsinline
       muted>
</video>

<div id="status">
    Connecting camera...
</div>

<script>
const video = document.getElementById("video");
const statusText = document.getElementById("status");
const canvas = document.createElement("canvas");

let sending = false;

async function startCamera() {
    try {
        const stream =
            await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: {
                        ideal: "environment"
                    },
                    width: {
                        ideal: 1280
                    },
                    height: {
                        ideal: 720
                    }
                },
                audio: false
            });

        video.srcObject = stream;

        statusText.innerText =
            "Camera connected - sending video to laptop";

        await fetch("/api/camera/start", {
            method: "POST"
        });

        sendFrame();

    } catch (error) {
        console.error("Camera error:", error);

        statusText.innerText =
            "Camera permission failed. " +
            "Allow camera access and reload.";
    }
}

function sendFrame() {
    if (!video.videoWidth) {
        setTimeout(sendFrame, 300);
        return;
    }

    if (sending) {
        setTimeout(sendFrame, 150);
        return;
    }

    sending = true;

    canvas.width = 640;
    canvas.height = 480;

    const context = canvas.getContext("2d");

    context.drawImage(
        video,
        0,
        0,
        640,
        480
    );

    canvas.toBlob(
        async function(blob) {
            if (!blob) {
                sending = false;
                setTimeout(sendFrame, 150);
                return;
            }

            try {
                await fetch("/api/frame", {
                    method: "POST",
                    headers: {
                        "Content-Type": "image/jpeg"
                    },
                    body: blob
                });

            } catch (error) {
                console.error(
                    "Frame send error:",
                    error
                );
            }

            sending = false;
            setTimeout(sendFrame, 150);
        },
        "image/jpeg",
        0.75
    );
}

window.addEventListener(
    "beforeunload",
    function() {
        fetch("/api/camera/stop", {
            method: "POST",
            keepalive: true
        });
    }
);

startCamera();
</script>

</body>
</html>
"""


# ============================================================
# LAPTOP ADMIN PAGE
# ============================================================

ADMIN_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>Unusual Activity Detection</title>

<style>
body {
    margin: 0;
    background: #f4f6f8;
    color: #222;
    font-family: Arial, sans-serif;
}

header {
    background: #17202a;
    color: white;
    padding: 20px;
}

header h1 {
    margin: 0;
}

header p {
    margin-top: 8px;
}

.container {
    width: 94%;
    max-width: 1200px;
    margin: 20px auto;
}

.card {
    background: white;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.video-box {
    text-align: center;
}

#liveVideo {
    width: 100%;
    max-height: 600px;
    object-fit: contain;
    background: black;
    border-radius: 10px;
}

.grid {
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
}

.metric {
    background: #f7f7f7;
    border-radius: 10px;
    padding: 15px;
}

.metric-title {
    font-size: 14px;
    color: #666;
}

.metric-value {
    font-size: 25px;
    font-weight: bold;
    margin-top: 7px;
}

.normal {
    color: green;
}

.warning {
    color: orange;
}

.danger {
    color: red;
}

.alert {
    border-left: 5px solid red;
    padding: 12px;
    margin-top: 10px;
    background: #fff4f4;
    border-radius: 6px;
}

button {
    padding: 10px 18px;
    border: none;
    border-radius: 7px;
    background: #17202a;
    color: white;
    cursor: pointer;
}
</style>
</head>

<body>

<header>
    <h1>
        Unusual Activity Detection Using Camera by AI
    </h1>
    <p>
        Laptop Admin Panel — Mobile Camera → R3D-18 AI
    </p>
</header>

<div class="container">

    <div class="card video-box">
        <h2>Live Mobile Camera</h2>

        <img id="liveVideo" src="/video_feed">

        <p id="mobileStatus">
            Checking mobile camera...
        </p>
    </div>

    <div class="card">
        <h2>Current AI Detection</h2>

        <div class="grid">

            <div class="metric">
                <div class="metric-title">Activity</div>
                <div id="activity" class="metric-value">
                    Normal
                </div>
            </div>

            <div class="metric">
                <div class="metric-title">Status</div>
                <div id="status" class="metric-value">
                    NORMAL
                </div>
            </div>

            <div class="metric">
                <div class="metric-title">Confidence</div>
                <div id="confidence" class="metric-value">
                    0%
                </div>
            </div>

            <div class="metric">
                <div class="metric-title">Temporal Frames</div>
                <div id="frames" class="metric-value">
                    0 / 8
                </div>
            </div>

            <div class="metric">
                <div class="metric-title">Risk Level</div>
                <div id="risk" class="metric-value">
                    0.0 / 10
                </div>
            </div>

            <div class="metric">
                <div class="metric-title">Risk Stage</div>
                <div id="riskStage" class="metric-value">
                    Waiting
                </div>
            </div>

        </div>

        <p id="message">
            Waiting for mobile camera...
        </p>
    </div>

    <div class="card">
        <h2>Alert History</h2>

        <button onclick="clearAlerts()">
            Clear History
        </button>

        <div id="alerts">
            No alerts.
        </div>
    </div>

</div>

<script>
async function updateState() {
    try {
        const response =
            await fetch("/api/latest-state");

        const data = await response.json();

        document.getElementById("activity").innerText =
            data.activity || "Normal";

        document.getElementById("status").innerText =
            data.status || "NORMAL";

        document.getElementById("confidence").innerText =
            Number(data.confidence || 0).toFixed(1) + "%";

        const buffer = data.temporal_buffer || 0;
        const required = data.temporal_required || 8;

        document.getElementById("frames").innerText =
            buffer + " / " + required;

        document.getElementById("message").innerText =
            data.message || "";

        const risk = Number(data.risk_level || 0);

        document.getElementById("risk").innerText =
            risk.toFixed(1) + " / 10";

        document.getElementById("riskStage").innerText =
            data.risk_stage || "Waiting";

        const activityElement =
            document.getElementById("activity");

        activityElement.className = "metric-value";

        if (
            data.activity_code &&
            data.activity_code !== "normal"
        ) {
            activityElement.classList.add("danger");
        } else {
            activityElement.classList.add("normal");
        }

        const mobileStatus =
            document.getElementById("mobileStatus");

        if (data.mobile_online) {
            mobileStatus.innerText =
                "Mobile camera connected";
        } else {
            mobileStatus.innerText =
                "Mobile Offline";
        }

    } catch (error) {
        console.error("State error:", error);
    }
}

async function loadAlerts() {
    try {
        const response = await fetch("/api/alerts");
        const data = await response.json();

        const container =
            document.getElementById("alerts");

        container.replaceChildren();

        if (!data || data.length === 0) {
            container.innerText = "No alerts.";
            return;
        }

        data.forEach(function(alert) {
            const box = document.createElement("div");
            box.className = "alert";

            const title = document.createElement("b");
            title.textContent =
                alert.activity || "Activity";

            box.appendChild(title);

            const details = document.createElement("p");
            details.textContent =
                "Time: " + (alert.time || "") +
                " | Confidence: " +
                (alert.confidence || 0) + "%";

            box.appendChild(details);

            if (alert.evidence) {
                const link = document.createElement("a");
                link.href = alert.evidence;
                link.target = "_blank";
                link.rel = "noopener noreferrer";
                link.textContent = "View Evidence";

                box.appendChild(link);
            }

            container.appendChild(box);
        });

    } catch (error) {
        console.error("Alert loading error:", error);
    }
}

async function clearAlerts() {
    try {
        await fetch("/api/alerts/clear", {
            method: "POST"
        });

        loadAlerts();

    } catch (error) {
        console.error("Clear alerts error:", error);
    }
}

setInterval(updateState, 300);
setInterval(loadAlerts, 1000);

updateState();
loadAlerts();
</script>

</body>
</html>
"""


# ============================================================
# ADMIN HOME
# ============================================================

@app.route("/")
def index():
    return ADMIN_HTML


# ============================================================
# MOBILE CAMERA PAGE
# ============================================================

@app.route("/mobile")
def mobile():
    return MOBILE_HTML


# ============================================================
# RECEIVE FRAMES FROM MOBILE
# ============================================================

@app.route("/api/frame", methods=["POST"])
def receive_frame():
    global latest_frame
    global latest_state
    global mobile_online
    global last_frame_time

    try:
        if not request.data:
            return jsonify({
                "success": False,
                "message": "Empty frame"
            }), 400

        image_array = np.frombuffer(
            request.data,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return jsonify({
                "success": False,
                "message": "Could not decode mobile frame"
            }), 400

        with frame_lock:
            latest_frame = frame.copy()

        mobile_online = True
        last_frame_time = datetime.now()

        result = detector.process_frame(
            frame,
            location={
                "source": "mobile_camera",
                "device": "phone"
            }
        )

        if result is None:
            result = {
                "success": True,
                "activity": "Analyzing Scene",
                "activity_code": "normal",
                "status": "ANALYZING",
                "confidence": 0.0,
                "normal_prob": 0.0,
                "anomaly_score": 0.0,
                "confirmation_progress": 0,
                "temporal_buffer": 0,
                "temporal_required": 8,
                "message": "Collecting temporal frames...",
                "event_active": False,
                "event_id": None,
                "alert_created": False,
                "alert_data": None,
                "latency": {}
            }

        risk = calculate_risk(result)

        result["risk_level"] = risk
        result["risk_stage"] = get_risk_stage(risk)
        result["mobile_online"] = True

        # ====================================================
        # SAVE ALERT + SEND AUTOMATIC GMAIL
        # ====================================================

        if result.get("alert_created", False):
            alert_data = result.get("alert_data")

            if isinstance(alert_data, dict):
                should_send_email = False

                with alert_lock:
                    event_id = str(
                        alert_data.get("event_id")
                        or result.get("event_id")
                        or ""
                    )

                    existing_ids = {
                        str(item.get("event_id", ""))
                        for item in alerts
                    }

                    if not event_id or event_id not in existing_ids:
                        alerts.append(alert_data)
                        save_alerts()
                        should_send_email = True

                if should_send_email:
                    queue_email_alert(alert_data, result)

        with state_lock:
            latest_state = result.copy()

        return jsonify(result)

    except Exception as error:
        print()
        print("=" * 70)
        print("FRAME PROCESSING ERROR")
        print(error)
        print("=" * 70)

        return jsonify({
            "success": False,
            "activity": "Analyzing Scene",
            "activity_code": "normal",
            "status": "ERROR",
            "confidence": 0.0,
            "normal_prob": 0.0,
            "anomaly_score": 0.0,
            "confirmation_progress": 0,
            "temporal_buffer": 0,
            "temporal_required": 8,
            "message": str(error),
            "event_active": False,
            "event_id": None,
            "alert_created": False,
            "alert_data": None,
            "mobile_online": False,
            "risk_level": 0.0,
            "risk_stage": "Waiting"
        }), 500


# ============================================================
# LAPTOP LIVE VIDEO
# ============================================================

@app.route("/video_feed")
def video_feed():

    def generate():
        while True:
            with frame_lock:
                if latest_frame is None:
                    frame = None
                else:
                    frame = latest_frame.copy()

            if frame is None:
                time.sleep(0.1)
                continue

            success, encoded = cv2.imencode(
                ".jpg",
                frame,
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    80
                ]
            )

            if not success:
                time.sleep(0.05)
                continue

            frame_bytes = encoded.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

            time.sleep(0.03)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# LATEST AI STATE
# ============================================================

@app.route("/api/latest-state")
def latest_state_api():
    with state_lock:
        data = latest_state.copy()

    data["mobile_online"] = mobile_online

    if last_frame_time is not None:
        elapsed = (
            datetime.now() - last_frame_time
        ).total_seconds()

        if elapsed > 3:
            data["mobile_online"] = False

    return jsonify(data)


# ============================================================
# SYSTEM STATUS
# ============================================================

@app.route("/api/status")
def system_status():
    return jsonify({
        "success": True,
        "system": "ready",
        "model": "R3D-18",
        "classes": 8,
        "clip_length": 8,
        "image_size": 112,
        "mobile_online": mobile_online,
        "laptop_ip": LAPTOP_IP,
        "port": PORT,
        "gmail_alerts": True
    })


# ============================================================
# CAMERA START
# ============================================================

@app.route("/api/camera/start", methods=["POST"])
def camera_start():
    global mobile_online

    mobile_online = True

    try:
        detector.reset()

    except Exception as error:
        print("Detector reset error:", error)

    return jsonify({
        "success": True,
        "message": "Mobile camera started"
    })


# ============================================================
# CAMERA STOP
# ============================================================

@app.route("/api/camera/stop", methods=["POST"])
def camera_stop():
    global mobile_online

    mobile_online = False

    return jsonify({
        "success": True,
        "message": "Mobile camera stopped"
    })


# ============================================================
# ALERT HISTORY
# ============================================================

@app.route("/api/alerts")
def get_alerts():
    with alert_lock:
        data = list(alerts)

    data.reverse()

    return jsonify(data)


# ============================================================
# CLEAR ALERT HISTORY
# ============================================================

@app.route("/api/alerts/clear", methods=["POST"])
def clear_alerts():
    global alerts

    with alert_lock:
        alerts = []
        save_alerts()

    return jsonify({
        "success": True,
        "message": "Alert history cleared"
    })


# ============================================================
# EVIDENCE FILES
# ============================================================

@app.route("/evidence/<path:filename>")
def evidence(filename):
    return send_from_directory(
        str(EVIDENCE_DIR),
        filename
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("UNUSUAL ACTIVITY DETECTION - LAPTOP ADMIN")
    print("=" * 70)

    print()
    print("Laptop Admin:")
    print(f"https://{LAPTOP_IP}:{PORT}/")

    print()
    print("Mobile Camera:")
    print(f"https://{LAPTOP_IP}:{PORT}/mobile")

    print()
    print("Model: R3D-18")
    print("Classes: 8")
    print("Temporal clip: 8 frames")
    print("Image size: 112 x 112")
    print("Automatic Gmail Alerts: Enabled")
    print("Email cooldown: 120 seconds")
    print()
    print("=" * 70)

    if not SSL_CERT.is_file() or not SSL_KEY.is_file():
        raise FileNotFoundError(
            "HTTPS certificate or server key is missing. "
            "Check SSL_CERT and SSL_KEY paths."
        )

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        threaded=True,
        use_reloader=False,
        ssl_context=(
            str(SSL_CERT),
            str(SSL_KEY)
        )
    )
