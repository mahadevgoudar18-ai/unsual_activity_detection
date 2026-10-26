from ultralytics import YOLO
import cv2


class YOLODetector:

    def __init__(self):

        print("Loading YOLO model...")

        self.model = YOLO("yolo11n.pt")

        print("YOLO model loaded successfully.")


    def detect(self, frame):

        results = self.model(
            frame,
            verbose=False
        )

        result = results[0]

        detections = []
        people_count = 0

        if result.boxes is not None:

            for box in result.boxes:

                confidence = float(box.conf[0])

                if confidence < 0.40:
                    continue

                class_id = int(box.cls[0])

                class_name = self.model.names[class_id]

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                if class_name == "person":
                    people_count += 1

                detections.append({
                    "class": class_name,
                    "confidence": round(
                        confidence * 100,
                        2
                    ),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })

                # Draw bounding box

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = (
                    f"{class_name} "
                    f"{confidence * 100:.1f}%"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        return {
            "frame": frame,
            "detections": detections,
            "people_count": people_count
        }