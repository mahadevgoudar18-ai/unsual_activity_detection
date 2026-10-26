import cv2
import numpy as np


class ActivityDetector:

    def __init__(self):

        self.previous_gray = None

        self.movement_score = 0

        self.activity = "Normal"

        self.frame_count = 0

    def calculate_movement(self, frame):

        if frame is None:
            return 0

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.resize(
            gray,
            (320, 180)
        )

        if self.previous_gray is None:

            self.previous_gray = gray

            return 0

        difference = cv2.absdiff(
            self.previous_gray,
            gray
        )

        movement = np.mean(
            difference
        )

        self.previous_gray = gray

        # Convert movement to percentage

        score = movement * 2.5

        score = max(
            0,
            min(
                100,
                score
            )
        )

        self.movement_score = score

        return round(score)

    def detect_activity(
        self,
        frame,
        detections=None
    ):

        if detections is None:
            detections = []

        movement = self.calculate_movement(
            frame
        )

        labels = []

        for detection in detections:

            if not isinstance(
                detection,
                dict
            ):
                continue

            label = str(
                detection.get(
                    "class",
                    ""
                )
            ).lower()

            labels.append(label)

        # -----------------------------------------
        # Object based activity
        # -----------------------------------------

        people = labels.count(
            "person"
        )

        dogs = labels.count(
            "dog"
        )

        # -----------------------------------------
        # Basic activity rules
        # -----------------------------------------

        if people == 0:

            activity = "Normal"

        elif dogs > 0 and movement > 75:

            activity = "Dog Activity"

        elif people >= 2 and movement > 80:

            activity = "Possible Fight"

        elif movement > 85:

            activity = "Running / Rapid Movement"

        elif movement > 55:

            activity = "Moving"

        elif movement > 20:

            activity = "Walking"

        else:

            activity = "Standing / Idle"

        self.activity = activity

        return {
            "activity": activity,
            "movement_score": movement,
            "people_detected": people
        }

    def analyze(
        self,
        frame,
        detections=None
    ):

        return self.detect_activity(
            frame,
            detections
        )