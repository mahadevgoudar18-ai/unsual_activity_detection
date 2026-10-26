class AnomalyDetector:

    def __init__(self, threshold=0.70):
        self.threshold = threshold

    def calculate_score(self, activity_data):

        score = 0.0

        if not activity_data:
            return score

        # Get values safely
        movement = float(activity_data.get("movement_score", 0))
        people = int(activity_data.get("people_detected", 0))
        objects = activity_data.get("detections", [])

        # -----------------------------------------
        # 1. High movement
        # -----------------------------------------
        if movement >= 80:
            score += 0.35

        elif movement >= 60:
            score += 0.20

        elif movement >= 40:
            score += 0.10

        # -----------------------------------------
        # 2. Multiple people with high movement
        # -----------------------------------------
        if people >= 2 and movement >= 60:
            score += 0.30

        # -----------------------------------------
        # 3. Check detected objects
        # -----------------------------------------
        detected_names = []

        for obj in objects:
            if isinstance(obj, dict):
                name = obj.get("class", obj.get("name", ""))
                detected_names.append(str(name).lower())

        # -----------------------------------------
        # 4. Person + dog interaction
        # -----------------------------------------
        has_person = "person" in detected_names
        has_dog = "dog" in detected_names

        if has_person and has_dog:
            score += 0.40

        # -----------------------------------------
        # 5. Several people + high movement
        # -----------------------------------------
        if people >= 2 and movement >= 70:
            score += 0.25

        # Keep score between 0 and 1
        score = min(score, 1.0)

        return round(score, 2)

    def is_unusual(self, score):

        return score >= self.threshold