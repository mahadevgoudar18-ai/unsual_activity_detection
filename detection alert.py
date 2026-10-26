# ============================================================
# AI UNUSUAL ACTIVITY DETECTION SYSTEM
# ALERT MANAGER
# ============================================================

import time
from datetime import datetime


class AlertManager:

    def __init__(self, cooldown=10):

        self.cooldown = cooldown

        self.last_alert_time = 0

        self.last_activity = ""

        self.alert_history = []

    # ========================================================
    # CHECK WHETHER NEW ALERT CAN BE CREATED
    # ========================================================

    def can_create_alert(self, activity):

        current_time = time.time()

        # First alert
        if self.last_alert_time == 0:
            return True

        # Different activity
        if activity != self.last_activity:
            return True

        # Cooldown
        if current_time - self.last_alert_time >= self.cooldown:
            return True

        return False

    # ========================================================
    # DETERMINE EMERGENCY TYPE
    # ========================================================

    def get_emergency_type(self, activity):

        if not activity:
            return "none"

        activity_lower = activity.lower()

        # ----------------------------------------------------
        # MEDICAL EMERGENCIES
        # ----------------------------------------------------

        medical_activities = [
            "fall",
            "falling",
            "collapsed",
            "collapse",
            "medical emergency",
            "person down",
            "injury"
        ]

        for item in medical_activities:

            if item in activity_lower:
                return "medical"

        # ----------------------------------------------------
        # POLICE / SECURITY EMERGENCIES
        # ----------------------------------------------------

        police_activities = [
            "fight",
            "fighting",
            "violence",
            "attack",
            "dog attack",
            "weapon",
            "gun",
            "knife",
            "theft",
            "robbery",
            "intrusion",
            "trespassing",
            "burglary",
            "aggressive"
        ]

        for item in police_activities:

            if item in activity_lower:
                return "police"

        # ----------------------------------------------------
        # OTHER UNUSUAL ACTIVITY
        # ----------------------------------------------------

        unusual_activities = [
            "running",
            "running in restricted area",
            "loitering",
            "suspicious movement",
            "unusual activity"
        ]

        for item in unusual_activities:

            if item in activity_lower:
                return "security"

        return "none"

    # ========================================================
    # CREATE ALERT
    # ========================================================

    def create_alert(
        self,
        activity,
        confidence=0,
        anomaly_score=0,
        camera_id="Mobile Camera"
    ):

        if not self.can_create_alert(activity):
            return None

        emergency_type = self.get_emergency_type(activity)

        # Do not create an alert for normal activity
        if emergency_type == "none":
            return None

        current_time = time.time()

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # ----------------------------------------------------
        # ALERT MESSAGE
        # ----------------------------------------------------

        if emergency_type == "medical":

            message = (
                "MEDICAL EMERGENCY DETECTED: "
                + activity
            )

            destination = "Nearest Hospital"

        elif emergency_type == "police":

            message = (
                "POLICE EMERGENCY DETECTED: "
                + activity
            )

            destination = "Nearest Police Station"

        else:

            message = (
                "SECURITY ALERT DETECTED: "
                + activity
            )

            destination = "Security / Police"

        # ----------------------------------------------------
        # ALERT OBJECT
        # ----------------------------------------------------

        alert = {

            "id": len(self.alert_history) + 1,

            "activity": activity,

            "confidence": round(
                float(confidence),
                2
            ),

            "anomaly_score": round(
                float(anomaly_score),
                2
            ),

            "camera_id": camera_id,

            "emergency_type": emergency_type,

            "destination": destination,

            "message": message,

            "timestamp": timestamp,

            "status": "ACTIVE"

        }

        # ----------------------------------------------------
        # SAVE ALERT
        # ----------------------------------------------------

        self.alert_history.insert(
            0,
            alert
        )

        # Keep only latest 100 alerts
        self.alert_history = (
            self.alert_history[:100]
        )

        self.last_alert_time = current_time

        self.last_activity = activity

        return alert

    # ========================================================
    # GET ALL ALERTS
    # ========================================================

    def get_alerts(self):

        return self.alert_history

    # ========================================================
    # GET LATEST ALERT
    # ========================================================

    def get_latest_alert(self):

        if not self.alert_history:

            return None

        return self.alert_history[0]

    # ========================================================
    # CLEAR ALERTS
    # ========================================================

    def clear_alerts(self):

        self.alert_history = []

        self.last_alert_time = 0

        self.last_activity = ""

    # ========================================================
    # GET ALERT COUNT
    # ========================================================

    def get_alert_count(self):

        return len(self.alert_history)