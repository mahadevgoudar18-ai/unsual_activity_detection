from ultralytics import YOLO
import cv2


class YOLODetector:

    def __init__(self, model_path="yolov8n.pt", confidence=0.35):

        self.confidence = confidence

        print("Loading YOLO model...")

        self.model = YOLO(model_path)

        print("YOLO model loaded successfully.")


    def detect(self, frame):

        detections = []

        # Run YOLO detection
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False
        )

        annotated_frame = frame.copy()

        # Process YOLO results
        for result in results:

            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:

                try:

                    # Bounding box
                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                    # Confidence
                    conf = float(
                        box.conf[0]
                        .cpu()
                        .numpy()
                    )

                    # Class ID
                    class_id = int(
                        box.cls[0]
                        .cpu()
                        .numpy()
                    )

                    # Class name
                    class_name = self.model.names[
                        class_id
                    ]

                    # Store detection
                    detection = {

                        "class": class_name,

                        "confidence": round(
                            conf,
                            3
                        ),

                        "bbox": [

                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2)

                        ]

                    }

                    detections.append(
                        detection
                    )

                    # ---------------------------------
                    # DRAW BOUNDING BOX
                    # ---------------------------------

                    cv2.rectangle(

                        annotated_frame,

                        (x1, y1),

                        (x2, y2),

                        (0, 255, 0),

                        2

                    )

                    # Label
                    label = (
                        f"{class_name} "
                        f"{conf * 100:.0f}%"
                    )

                    cv2.putText(

                        annotated_frame,

                        label,

                        (x1, max(y1 - 10, 20)),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.55,

                        (0, 255, 0),

                        2

                    )

                except Exception as error:

                    print(
                        "YOLO detection error:",
                        error
                    )

        return annotated_frame, detections