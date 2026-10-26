# ============================================================
# VERSION 3 - 8 CLASS UNUSUAL ACTIVITY TRAINING
# ============================================================
#
# Goal:
# Reduce confusion between:
#   Fighting
#   Fire / Smoke
#   Theft
#
# Improve:
#   Normal
#   Falling
#   Accident
#   Vandalism
#   Shooting
#
# Classes:
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
# This script trains the model only.
# It does NOT run the camera.
# ============================================================

import os
import cv2
import shutil
import random
import hashlib

from pathlib import Path
from collections import defaultdict

from ultralytics import YOLO


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42

random.seed(SEED)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASETS_DIR = BASE_DIR / "datasets"

OUTPUT_DATASET = BASE_DIR / "activity_dataset_v3"

RUNS_DIR = BASE_DIR / "runs"

MODEL_OUTPUT = BASE_DIR / "best.pt"


# ============================================================
# TRAINING SETTINGS
# ============================================================

MODEL_NAME = "yolov8s-cls.pt"

IMAGE_SIZE = 224

EPOCHS = 30

BATCH_SIZE = 16

WORKERS = 0

PATIENCE = 0

TRAIN_RATIO = 0.80


# ============================================================
# SOURCE SETTINGS
# ============================================================

# We have only around 50 Normal source videos.
# To avoid one class dominating, use 50 source videos
# for each class.
MAX_SOURCES_PER_CLASS = 50


# More frames than Version 2.
FRAMES_PER_SOURCE = 32


# ============================================================
# IMAGE QUALITY SETTINGS
# ============================================================

MIN_IMAGE_WIDTH = 80

MIN_IMAGE_HEIGHT = 80

# Reject almost completely black/empty frames.
MIN_MEAN_BRIGHTNESS = 8.0

# Reject frames that are almost entirely one color/value.
MIN_STDDEV = 4.0


# ============================================================
# 8 CLASSES
# ============================================================

CLASSES = [
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
# VIDEO EXTENSIONS
# ============================================================

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".mpeg",
    ".mpg",
    ".m4v",
    ".webm",
    ".3gp",
    ".gif",
}


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# CLASS MAPPING
# ============================================================

def normalize_name(name):
    """
    Normalize a folder/file name for matching.
    """

    name = name.lower().strip()

    replacements = [
        (" ", ""),
        ("-", ""),
        ("_", ""),
    ]

    for old, new in replacements:
        name = name.replace(old, new)

    return name


def map_name_to_class(name):
    """
    Convert dataset category names into one of our 8 classes.
    """

    n = normalize_name(name)

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    normal_names = {
        "normal",
        "normalvideos",
        "normalevent",
        "normalevents",
        "normalvideosforeventrecognition",
    }

    if n in normal_names:
        return "normal"

    # --------------------------------------------------------
    # FALLING
    # FallVision folders:
    # f_mask_b_1
    # f_mask_b_2
    # f_mask_b_3
    # f_mask_b_4
    # --------------------------------------------------------

    if n in {
        "fmaskb1",
        "fmaskb2",
        "fmaskb3",
        "fmaskb4",
        "fall",
        "falls",
        "falling",
        "falldetection",
        "falldetectiondataset",
    }:
        return "falling"

    # --------------------------------------------------------
    # ACCIDENT
    # --------------------------------------------------------

    if n in {
        "roadaccident",
        "roadaccidents",
        "accident",
        "accidents",
        "roadaccidentvideos",
    }:
        return "accident"

    # --------------------------------------------------------
    # FIGHTING
    # --------------------------------------------------------

    if n in {
        "fighting",
        "fight",
        "assault",
        "abuse",
    }:
        return "fighting"

    # --------------------------------------------------------
    # THEFT
    # --------------------------------------------------------

    if n in {
        "theft",
        "stealing",
        "steal",
        "robbery",
        "shoplifting",
        "burglary",
    }:
        return "theft"

    # --------------------------------------------------------
    # VANDALISM
    # --------------------------------------------------------

    if n in {
        "vandalism",
    }:
        return "vandalism"

    # --------------------------------------------------------
    # FIRE / SMOKE
    # --------------------------------------------------------

    if n in {
        "arson",
        "explosion",
        "fire",
        "fires",
        "smoke",
        "firesmoke",
        "fireandsmoke",
    }:
        return "fire_smoke"

    # --------------------------------------------------------
    # SHOOTING
    # --------------------------------------------------------

    if n in {
        "shooting",
        "shootings",
    }:
        return "shooting"

    # --------------------------------------------------------
    # IGNORE
    # --------------------------------------------------------

    return None


# ============================================================
# SOURCE RECORD
# ============================================================

class SourceRecord:

    def __init__(
        self,
        path,
        class_name
    ):
        self.path = Path(path)
        self.class_name = class_name


# ============================================================
# DISCOVER DATASET
# ============================================================

def discover_sources():

    print("\n" + "=" * 70)
    print("SCANNING DATASETS")
    print("=" * 70)

    if not DATASETS_DIR.exists():
        print(
            "\nERROR: datasets folder was not found:"
        )
        print(DATASETS_DIR)
        raise SystemExit(1)

    sources = defaultdict(list)

    all_files = list(
        DATASETS_DIR.rglob("*")
    )

    for path in all_files:

        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if (
            suffix not in VIDEO_EXTENSIONS
            and suffix not in IMAGE_EXTENSIONS
        ):
            continue

        # ----------------------------------------------------
        # Check path components.
        #
        # This is important because UCF-Crime category names
        # are usually folder names.
        # ----------------------------------------------------

        parts = list(path.parts)

        detected_class = None

        # Search from deepest folder toward root.
        for part in reversed(parts):

            detected_class = map_name_to_class(
                part
            )

            if detected_class:
                break

        # ----------------------------------------------------
        # File stem can also contain category information.
        # ----------------------------------------------------

        if detected_class is None:

            detected_class = map_name_to_class(
                path.stem
            )

        if detected_class is None:
            continue

        sources[
            detected_class
        ].append(
            SourceRecord(
                path,
                detected_class
            )
        )

    # --------------------------------------------------------
    # Print discovered source counts
    # --------------------------------------------------------

    print("\nDISCOVERED SOURCES:")

    for class_name in CLASSES:

        print(
            f"{class_name:<15}: "
            f"{len(sources[class_name])}"
        )

    print(
        "\nTotal discovered:",
        sum(
            len(v)
            for v in sources.values()
        )
    )

    return sources


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicate_paths(
    sources
):

    print("\nRemoving duplicate source paths...")

    result = defaultdict(list)

    for class_name in CLASSES:

        seen = set()

        for record in sources[class_name]:

            absolute_path = str(
                record.path.resolve()
            ).lower()

            if absolute_path in seen:
                continue

            seen.add(
                absolute_path
            )

            result[
                class_name
            ].append(
                record
            )

    return result


# ============================================================
# CREATE FILE SIGNATURE
# ============================================================

def file_signature(path):

    """
    Create a lightweight signature from path + file size.
    """

    try:

        stat = path.stat()

        value = (
            str(path.resolve())
            + "|"
            + str(stat.st_size)
        )

        return hashlib.md5(
            value.encode(
                "utf-8",
                errors="ignore"
            )
        ).hexdigest()

    except Exception:
        return str(path)


# ============================================================
# REMOVE POSSIBLE DUPLICATE SOURCES
# ============================================================

def remove_duplicate_files(
    sources
):

    print(
        "\nChecking duplicate source files..."
    )

    result = defaultdict(list)

    for class_name in CLASSES:

        seen = set()

        for record in sources[class_name]:

            signature = file_signature(
                record.path
            )

            if signature in seen:
                continue

            seen.add(
                signature
            )

            result[
                class_name
            ].append(
                record
            )

    return result


# ============================================================
# BALANCE SOURCES
# ============================================================

def balance_sources(
    sources
):

    print("\n" + "=" * 70)
    print("BALANCING SOURCE VIDEOS")
    print("=" * 70)

    balanced = defaultdict(list)

    for class_name in CLASSES:

        available = list(
            sources[class_name]
        )

        random.shuffle(
            available
        )

        if len(available) < MAX_SOURCES_PER_CLASS:

            print(
                f"\nWARNING: {class_name} "
                f"has only {len(available)} sources."
            )

            selected = available

        else:

            selected = available[
                :MAX_SOURCES_PER_CLASS
            ]

        balanced[
            class_name
        ] = selected

        print(
            f"{class_name:<15}: "
            f"{len(selected)} sources"
        )

    missing = [
        c
        for c in CLASSES
        if len(balanced[c]) == 0
    ]

    if missing:

        print(
            "\nERROR: Missing classes:"
        )

        for c in missing:
            print(
                " -",
                c
            )

        raise SystemExit(1)

    return balanced


# ============================================================
# SOURCE LEVEL TRAIN/VAL SPLIT
# ============================================================

def split_sources(
    balanced
):

    print("\n" + "=" * 70)
    print("SOURCE-LEVEL TRAIN / VALIDATION SPLIT")
    print("=" * 70)

    train_sources = defaultdict(list)

    val_sources = defaultdict(list)

    for class_name in CLASSES:

        items = list(
            balanced[class_name]
        )

        random.shuffle(
            items
        )

        split_index = int(
            len(items)
            * TRAIN_RATIO
        )

        train_part = items[
            :split_index
        ]

        val_part = items[
            split_index:
        ]

        train_sources[
            class_name
        ] = train_part

        val_sources[
            class_name
        ] = val_part

        print(
            f"{class_name:<15}: "
            f"train={len(train_part):<3} "
            f"val={len(val_part):<3}"
        )

    return (
        train_sources,
        val_sources
    )


# ============================================================
# READ IMAGE
# ============================================================

def read_image(path):

    try:

        # ----------------------------------------------------
        # Video
        # ----------------------------------------------------

        if path.suffix.lower() in VIDEO_EXTENSIONS:

            return None

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        image = cv2.imread(
            str(path)
        )

        return image

    except Exception:
        return None


# ============================================================
# IMAGE QUALITY CHECK
# ============================================================

def valid_frame(image):

    if image is None:
        return False

    if len(image.shape) != 3:
        return False

    height, width = image.shape[:2]

    if (
        width < MIN_IMAGE_WIDTH
        or height < MIN_IMAGE_HEIGHT
    ):
        return False

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    mean_value = float(
        gray.mean()
    )

    std_value = float(
        gray.std()
    )

    # Reject almost black frames.
    if mean_value < MIN_MEAN_BRIGHTNESS:
        return False

    # Reject almost completely blank frames.
    if std_value < MIN_STDDEV:
        return False

    return True


# ============================================================
# SAMPLE VIDEO FRAMES
# ============================================================

def extract_video_frames(
    path,
    frame_count
):

    frames = []

    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():
        return frames

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if total_frames <= 0:
        cap.release()
        return frames

    # --------------------------------------------------------
    # Uniform sampling
    #
    # Avoid using the same nearby frames repeatedly.
    # --------------------------------------------------------

    positions = np.linspace(
        0,
        max(total_frames - 1, 0),
        frame_count
    )

    positions = [
        int(x)
        for x in positions
    ]

    for position in positions:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            position
        )

        success, frame = cap.read()

        if not success:
            continue

        if not valid_frame(frame):
            continue

        frames.append(
            frame
        )

    cap.release()

    return frames


# ============================================================
# SUPPORT NUMPY
# ============================================================

try:
    import numpy as np
except ImportError:

    print(
        "\nERROR: numpy is required."
    )

    raise SystemExit(1)


# ============================================================
# SAVE IMAGE
# ============================================================

def save_frame(
    frame,
    output_path
):

    try:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        success = cv2.imwrite(
            str(output_path),
            frame
        )

        return success

    except Exception as e:

        print(
            "WARNING: Could not save:",
            output_path
        )

        print(e)

        return False


# ============================================================
# CREATE DATASET FROM SOURCES
# ============================================================

def create_dataset(
    train_sources,
    val_sources
):

    print("\n" + "=" * 70)
    print("CREATING VERSION 3 IMAGE DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Delete old V3 dataset
    # --------------------------------------------------------

    if OUTPUT_DATASET.exists():

        print(
            "\nRemoving previous V3 dataset..."
        )

        try:

            shutil.rmtree(
                OUTPUT_DATASET
            )

        except PermissionError:

            print(
                "\nERROR: Could not remove:"
            )

            print(
                OUTPUT_DATASET
            )

            print(
                "\nClose any program using this folder "
                "and run train.py again."
            )

            raise SystemExit(1)

    train_dir = (
        OUTPUT_DATASET
        / "train"
    )

    val_dir = (
        OUTPUT_DATASET
        / "val"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    statistics = {
        "train": defaultdict(int),
        "val": defaultdict(int),
    }

    # --------------------------------------------------------
    # Process one split
    # --------------------------------------------------------

    def process_split(
        source_dict,
        split_name
    ):

        print(
            f"\nProcessing {split_name.upper()}..."
        )

        for class_name in CLASSES:

            class_sources = (
                source_dict[class_name]
            )

            destination = (
                OUTPUT_DATASET
                / split_name
                / class_name
            )

            destination.mkdir(
                parents=True,
                exist_ok=True
            )

            print(
                f"\n{class_name}: "
                f"{len(class_sources)} sources"
            )

            for source_index, record in enumerate(
                class_sources
            ):

                path = record.path

                # ------------------------------------------------
                # Extract video frames
                # ------------------------------------------------

                if path.suffix.lower() in VIDEO_EXTENSIONS:

                    frames = extract_video_frames(
                        path,
                        FRAMES_PER_SOURCE
                    )

                # ------------------------------------------------
                # Handle image files
                # ------------------------------------------------

                else:

                    image = read_image(
                        path
                    )

                    if valid_frame(image):
                        frames = [image]
                    else:
                        frames = []

                # ------------------------------------------------
                # Save frames
                # ------------------------------------------------

                for frame_index, frame in enumerate(
                    frames
                ):

                    filename = (
                        f"{class_name}_"
                        f"{source_index:04d}_"
                        f"{frame_index:03d}.jpg"
                    )

                    output_path = (
                        destination
                        / filename
                    )

                    if save_frame(
                        frame,
                        output_path
                    ):

                        statistics[
                            split_name
                        ][class_name] += 1

                # ------------------------------------------------
                # Progress
                # ------------------------------------------------

                if (
                    source_index + 1
                ) % 10 == 0:

                    print(
                        f"  processed "
                        f"{source_index + 1}/"
                        f"{len(class_sources)}"
                    )

    process_split(
        train_sources,
        "train"
    )

    process_split(
        val_sources,
        "val"
    )

    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET CREATION COMPLETE")
    print("=" * 70)

    total_train = 0
    total_val = 0

    print("\nTRAIN:")

    for class_name in CLASSES:

        count = statistics[
            "train"
        ][class_name]

        total_train += count

        print(
            f"{class_name:<15}: {count}"
        )

    print(
        f"TOTAL TRAIN: {total_train}"
    )

    print("\nVALIDATION:")

    for class_name in CLASSES:

        count = statistics[
            "val"
        ][class_name]

        total_val += count

        print(
            f"{class_name:<15}: {count}"
        )

    print(
        f"TOTAL VAL: {total_val}"
    )

    return (
        total_train,
        total_val,
        statistics
    )


# ============================================================
# CHECK DATASET
# ============================================================

def check_dataset(
    statistics
):

    print("\n" + "=" * 70)
    print("DATASET BALANCE CHECK")
    print("=" * 70)

    for split in [
        "train",
        "val"
    ]:

        print(
            f"\n{split.upper()}:"
        )

        counts = []

        for class_name in CLASSES:

            count = statistics[
                split
            ][class_name]

            counts.append(
                count
            )

            print(
                f"{class_name:<15}: {count}"
            )

        if min(counts) == 0:

            print(
                "\nERROR: One or more classes contain zero images."
            )

            raise SystemExit(1)

    print(
        "\nSUCCESS: All 8 classes contain images."
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    print("\n" + "=" * 70)
    print("STARTING VERSION 3 MODEL TRAINING")
    print("=" * 70)

    print(
        "\nBase model:",
        MODEL_NAME
    )

    print(
        "Image size:",
        IMAGE_SIZE
    )

    print(
        "Epochs:",
        EPOCHS
    )

    print(
        "Batch:",
        BATCH_SIZE
    )

    print(
        "Frames/source:",
        FRAMES_PER_SOURCE
    )

    print(
        "Sources/class:",
        MAX_SOURCES_PER_CLASS
    )

    print(
        "\nLoading pretrained classification model..."
    )

    model = YOLO(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    results = model.train(

        data=str(
            OUTPUT_DATASET
        ),

        epochs=EPOCHS,

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        workers=WORKERS,

        patience=PATIENCE,

        pretrained=True,

        optimizer="AdamW",

        lr0=0.001,

        lrf=0.01,

        weight_decay=0.0005,

        dropout=0.15,

        # ----------------------------------------------------
        # Controlled augmentation
        # ----------------------------------------------------

        hsv_h=0.015,

        hsv_s=0.35,

        hsv_v=0.25,

        degrees=6.0,

        translate=0.06,

        scale=0.25,

        shear=2.0,

        perspective=0.0003,

        fliplr=0.5,

        flipud=0.0,

        mosaic=0.0,

        mixup=0.05,

        cutmix=0.05,

        auto_augment="randaugment",

        erasing=0.20,

        # ----------------------------------------------------
        # Training output
        # ----------------------------------------------------

        project=str(
            RUNS_DIR
        ),

        name="eight_class_activity_v3",

        exist_ok=True,

        verbose=True,

        plots=True,

        save=True,

        save_period=5,

        device=0,
    )

    return model, results


# ============================================================
# VALIDATE BEST MODEL
# ============================================================

def validate_model():

    print("\n" + "=" * 70)
    print("VALIDATING VERSION 3 MODEL")
    print("=" * 70)

    best_model_path = (
        RUNS_DIR
        / "eight_class_activity_v3"
        / "weights"
        / "best.pt"
    )

    if not best_model_path.exists():

        print(
            "\nERROR: best.pt was not found after training:"
        )

        print(
            best_model_path
        )

        raise SystemExit(1)

    print(
        "\nBest model:"
    )

    print(
        best_model_path
    )

    best_model = YOLO(
        str(best_model_path)
    )

    print(
        "\nRunning validation..."
    )

    metrics = best_model.val(
        data=str(
            OUTPUT_DATASET
        ),
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        verbose=True,
    )

    # --------------------------------------------------------
    # Copy best model to project root
    # --------------------------------------------------------

    shutil.copy2(
        best_model_path,
        MODEL_OUTPUT
    )

    print(
        "\nBest model copied to:"
    )

    print(
        MODEL_OUTPUT
    )

    # --------------------------------------------------------
    # Verify class mapping
    # --------------------------------------------------------

    names = best_model.names

    if isinstance(names, dict):

        model_classes = {
            int(k): str(v).lower()
            for k, v in names.items()
        }

    else:

        model_classes = {
            i: str(v).lower()
            for i, v in enumerate(names)
        }

    print(
        "\nMODEL CLASS MAPPING:"
    )

    for class_id in sorted(
        model_classes
    ):

        print(
            f"{class_id} -> "
            f"{model_classes[class_id]}"
        )

    # --------------------------------------------------------
    # Check classes
    # --------------------------------------------------------

    missing = [
        c
        for c in CLASSES
        if c not in model_classes.values()
    ]

    if missing:

        print(
            "\nERROR: Missing classes:"
        )

        for c in missing:
            print(
                " -",
                c
            )

        raise SystemExit(1)

    print(
        "\nSUCCESS: All 8 classes are present."
    )

    # --------------------------------------------------------
    # Model size
    # --------------------------------------------------------

    size = MODEL_OUTPUT.stat().st_size

    print(
        "\nModel size:",
        size,
        "bytes"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    try:

        top1 = float(
            metrics.top1
        )

        top5 = float(
            metrics.top5
        )

        print(
            "\nValidation Top-1:",
            f"{top1 * 100:.2f}%"
        )

        print(
            "Validation Top-5:",
            f"{top5 * 100:.2f}%"
        )

    except Exception:
        pass

    return best_model


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("VERSION 3 - 8 CLASS UNUSUAL ACTIVITY MODEL")
    print("=" * 70)

    print(
        "\nProject folder:"
    )

    print(
        BASE_DIR
    )

    print(
        "\nDataset folder:"
    )

    print(
        DATASETS_DIR
    )

    print(
        "\nClasses:"
    )

    for index, class_name in enumerate(
        CLASSES
    ):

        print(
            f"{index} -> {class_name}"
        )

    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------

    sources = discover_sources()

    # --------------------------------------------------------
    # Step 2
    # --------------------------------------------------------

    sources = remove_duplicate_paths(
        sources
    )

    # --------------------------------------------------------
    # Step 3
    # --------------------------------------------------------

    sources = remove_duplicate_files(
        sources
    )

    # --------------------------------------------------------
    # Step 4
    # --------------------------------------------------------

    balanced = balance_sources(
        sources
    )

    # --------------------------------------------------------
    # Step 5
    # --------------------------------------------------------

    (
        train_sources,
        val_sources
    ) = split_sources(
        balanced
    )

    # --------------------------------------------------------
    # Step 6
    # --------------------------------------------------------

    (
        total_train,
        total_val,
        statistics
    ) = create_dataset(
        train_sources,
        val_sources
    )

    # --------------------------------------------------------
    # Step 7
    # --------------------------------------------------------

    check_dataset(
        statistics
    )

    # --------------------------------------------------------
    # Step 8
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("READY FOR VERSION 3 TRAINING")
    print("=" * 70)

    print(
        "\nTraining images:",
        total_train
    )

    print(
        "Validation images:",
        total_val
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The model will train for the full 30 epochs."
    )

    print(
        "Early stopping is disabled."
    )

    print(
        "\nStarting training..."
    )

    # --------------------------------------------------------
    # Step 9
    # --------------------------------------------------------

    train_model()

    # --------------------------------------------------------
    # Step 10
    # --------------------------------------------------------

    validate_model()

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("VERSION 3 TRAINING COMPLETE")
    print("=" * 70)

    print(
        "\nFinal model:"
    )

    print(
        MODEL_OUTPUT
    )

    print(
        "\nNext step:"
    )

    print(
        "Run the 8-class model test before changing app.py."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()