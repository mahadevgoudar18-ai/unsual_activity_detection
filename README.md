# Unusual Activity Detection System Using AI

An intelligent, real-time spatiotemporal activity and anomaly detection system built with PyTorch, R3D-18 (3D CNN), YOLO, and Flask. The system monitors live camera streams to detect emergency events, display live analytics, and automatically dispatch immediate alert notifications to designated emergency responders.

---

## 🚀 Features

- **Spatiotemporal 3D CNN Engine (R3D-18)**:
  - Processes rolling 8-frame temporal video buffers to evaluate motion and actions coherently across time.
  - Recognizes 8 distinct classes:
    - `accident`
    - `falling`
    - `fighting`
    - `fire_smoke`
    - `normal`
    - `shooting`
    - `theft`
    - `vandalism`

- **Robust False-Alarm Suppression**:
  - Motion energy and stationary scene analysis to avoid false positives during resting, sleeping, or idle periods.
  - Streak-based temporal confirmation policy to prevent erratic single-frame triggers.

- **Real-Time Web Dashboard**:
  - Live video streaming with dynamic bounding boxes, anomaly confidence gauges, and latency monitoring.
  - Alert timeline with captured evidence snapshots.
  - Mobile camera streaming support via HTTPS/WebRTC for portable surveillance.

- **Automated Emergency Dispatch**:
  - Email notification pipeline routing specific alerts to relevant responders (Police, Medical, Fire).
  - Snapshot evidence attachment sent directly with the alert.

---

## 📁 Project Structure

```
├── app.py                      # Flask web application & streaming server
├── activity_detector.py        # Core R3D-18 spatiotemporal detector & state manager
├── detector.py                 # Object detection engine
├── email_alert.py              # Automated email notification dispatcher
├── train_temporal_model.py     # Training script for R3D-18 spatiotemporal model
├── train_video_model.py        # Video model training pipeline
├── evaluate_model.py           # Model evaluation and metrics calculation
├── requirements.txt            # Python dependencies
├── .env.example                # Example environment configuration
├── templates/
│   ├── index.html              # Main monitoring dashboard
│   └── mobile.html             # Mobile camera capture interface
├── detection/
│   ├── activity.py             # Activity analysis modules
│   ├── anomaly.py              # Anomaly detection logic
│   └── yolo_detector.py        # YOLO object detection wrapper
└── style.css                   # Dashboard stylesheet
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/mahadevgoudar18-ai/unsual_activity_detection.git
cd unsual_activity_detection
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your email configuration:
```bash
cp .env.example .env
```

Edit `.env`:
```ini
GMAIL_SENDER=your_email@gmail.com
GMAIL_APP_PASSWORD=your_gmail_app_password

POLICE_DEMO_EMAIL=police_alerts@example.com
MEDICAL_DEMO_EMAIL=medical_alerts@example.com
FIRE_DEMO_EMAIL=fire_alerts@example.com
```

---

## 🚦 Running the Application

Start the Flask server:
```bash
python app.py
```

Open your browser and navigate to:
```
http://localhost:5000
```
For secure mobile camera streaming (HTTPS), generate local certificates using `mkcert` and run the HTTPS service.

---

## 🧠 Training & Evaluation

To train or fine-tune the temporal model on custom video datasets:
```bash
python train_temporal_model.py
```

To evaluate the trained model:
```bash
python evaluate_model.py
```

---

## 📄 License
This project is for educational and research purposes.
