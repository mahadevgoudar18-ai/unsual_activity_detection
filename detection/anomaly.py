class AnomalyDetector:

    def __init__(self, alert_threshold=70):
        self.alert_threshold = alert_threshold

    def detect(
        self,
        frame=None,
        detections=None,
        activity="Normal",
        movement_score=0,
        confidence=0,
        **kwargs
    ):

        if detections is None:
            detections = []

        try:
            movement_score = float(movement_score)
        except:
            movement_score = 0

        try:
            confidence = float(confidence)
        except:
            confidence = 0

        movement_score = max(
            0,
            min(100, movement_score)
        )

        confidence = max(
            0,
            min(100, confidence)
        )

        # -----------------------------------------
        # Count people
        # -----------------------------------------

        people_detected = 0

        for detection in detections:

            if not isinstance(detection, dict):
                continue

            class_name = str(
                detection.get(
                    "class",
                    detection.get(
                        "name",
                        ""
                    )
                )
            ).lower()

            if class_name == "person":
                people_detected += 1

        # -----------------------------------------
        # Activity
        # -----------------------------------------

        activity_text = str(
            activity or "Normal"
        ).lower()

        # -----------------------------------------
        # Anomaly score
        # -----------------------------------------

        normal_activities = [
            "normal",
            "standing",
            "walking",
            "moving",
            "sitting",
            "idle",
            "waiting"
        ]

        if activity_text in normal_activities:

            anomaly_score = movement_score * 0.25

        elif "running" in activity_text:

            anomaly_score = max(
                60,
                movement_score * 0.85
            )

        elif "fight" in activity_text:

            anomaly_score = 90

        elif "fall" in activity_text:

            anomaly_score = 90

        elif "dog attack" in activity_text:

            anomaly_score = 92

        elif (
            "dog" in activity_text
            and "attack" in activity_text
        ):

            anomaly_score = 92

        elif "robbery" in activity_text:

            anomaly_score = 95

        elif "stealing" in activity_text:

            anomaly_score = 90

        elif "shoplifting" in activity_text:

            anomaly_score = 90

        elif "weapon" in activity_text:

            anomaly_score = 98

        elif "gun" in activity_text:

            anomaly_score = 98

        elif "knife" in activity_text:

            anomaly_score = 98

        elif "shooting" in activity_text:

            anomaly_score = 99

        elif "accident" in activity_text:

            anomaly_score = 96

        elif "fire" in activity_text:

            anomaly_score = 99

        elif "explosion" in activity_text:

            anomaly_score = 99

        elif "vandalism" in activity_text:

            anomaly_score = 90

        elif "suspicious" in activity_text:

            anomaly_score = max(
                70,
                movement_score
            )

        elif "unusual" in activity_text:

            anomaly_score = max(
                70,
                movement_score
            )

        elif "abnormal" in activity_text:

            anomaly_score = max(
                70,
                movement_score
            )

        else:

            anomaly_score = movement_score * 0.5

        # -----------------------------------------
        # Movement contribution
        # -----------------------------------------

        if movement_score >= 85:

            anomaly_score += 8

        elif movement_score >= 70:

            anomaly_score += 4

        # -----------------------------------------
        # Multiple people
        # -----------------------------------------

        if (
            people_detected >= 2
            and movement_score >= 70
        ):

            anomaly_score += 5

        # -----------------------------------------
        # Confidence
        # -----------------------------------------

        if confidence >= 80:

            anomaly_score += 2

        # -----------------------------------------
        # Limit
        # -----------------------------------------

        anomaly_score = max(
            0,
            min(
                100,
                round(anomaly_score)
            )
        )

        # -----------------------------------------
        # Is anomaly?
        # -----------------------------------------

        is_anomaly = (
            anomaly_score >=
            self.alert_threshold
        )

        # -----------------------------------------
        # Emergency type
        # -----------------------------------------

        emergency_type = "none"

        if is_anomaly:

            if any(
                word in activity_text
                for word in [
                    "fall",
                    "collapse",
                    "medical",
                    "accident",
                    "injury"
                ]
            ):

                emergency_type = "medical"

            elif any(
                word in activity_text
                for word in [
                    "fight",
                    "robbery",
                    "stealing",
                    "shoplifting",
                    "weapon",
                    "gun",
                    "knife",
                    "shooting",
                    "dog attack",
                    "vandalism",
                    "suspicious"
                ]
            ):

                emergency_type = "police"

            elif any(
                word in activity_text
                for word in [
                    "fire",
                    "explosion"
                ]
            ):

                emergency_type = "fire"

            else:

                emergency_type = "police"

        return {

            "anomaly_score":
                anomaly_score,

            "is_anomaly":
                is_anomaly,

            "activity":
                activity,

            "movement_score":
                round(movement_score),

            "confidence":
                round(confidence),

            "people_detected":
                people_detected,

            "emergency_type":
                emergency_type
        }

    def analyze(
        self,
        frame=None,
        detections=None,
        activity="Normal",
        movement_score=0,
        confidence=0,
        **kwargs
    ):

        return self.detect(
            frame=frame,
            detections=detections,
            activity=activity,
            movement_score=movement_score,
            confidence=confidence,
            **kwargs
        )