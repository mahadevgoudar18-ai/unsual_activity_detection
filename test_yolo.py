from ultralytics import YOLO
from pathlib import Path

print("=" * 60)
print("TESTING 8-CLASS UNUSUAL ACTIVITY MODEL")
print("=" * 60)

MODEL_PATH = Path("best.pt")

EXPECTED_CLASSES = {
    "accident",
    "falling",
    "fighting",
    "fire_smoke",
    "normal",
    "shooting",
    "theft",
    "vandalism"
}

print("\nLoading model...")

try:
    model = YOLO(str(MODEL_PATH))
    print("Model loaded successfully!")
except Exception as e:
    print(f"ERROR: Could not load model: {e}")
    raise SystemExit(1)

print("\nModel Classes:")

for class_id, class_name in model.names.items():
    print(f"{class_id} -> {class_name}")

print("\nChecking classes...")

found_classes = set(model.names.values())

if found_classes != EXPECTED_CLASSES:
    print("ERROR: Class list is incorrect.")
    print("Expected:", EXPECTED_CLASSES)
    print("Found:", found_classes)
    raise SystemExit(1)

print("SUCCESS: All 8 classes are correct!")

print("\n" + "=" * 60)
print("MODEL TEST PASSED")
print("=" * 60)

print("\nYour 8-class model is ready for camera integration.")

print("\nClasses available:")
for class_id, class_name in model.names.items():
    print(f"{class_id} -> {class_name}")