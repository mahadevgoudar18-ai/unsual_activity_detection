from pathlib import Path
import torch

MODEL = Path("temporal_model.pth")

EXPECTED_CLASSES = [
    "accident",
    "falling",
    "fighting",
    "fire_smoke",
    "normal",
    "shooting",
    "theft",
    "vandalism",
]

print("=" * 70)
print("TEMPORAL MODEL DIAGNOSTIC")
print("=" * 70)

print("\nModel path:")
print(MODEL.resolve())

print("\nModel exists:")
print(MODEL.exists())

if not MODEL.exists():
    print("\nERROR: temporal_model.pth was not found.")
    raise SystemExit(1)

try:
    checkpoint = torch.load(
        str(MODEL),
        map_location="cpu",
        weights_only=False
    )
except Exception as error:
    print("\nERROR: Could not load checkpoint.")
    print(error)
    raise SystemExit(1)

print("\nCheckpoint type:")
print(type(checkpoint))

if isinstance(checkpoint, dict):

    print("\nCheckpoint keys:")
    for key in checkpoint.keys():
        print(" -", key)

    print("\nSaved classes:")
    saved_classes = checkpoint.get("classes")

    if saved_classes is None:
        print("No class list stored in checkpoint.")
    else:
        print(saved_classes)

        normalized = [
            str(x).strip().lower()
            for x in saved_classes
        ]

        if normalized == EXPECTED_CLASSES:
            print("\nCLASS ORDER: CORRECT")
        else:
            print("\nCLASS ORDER: WRONG")
            print("\nExpected:")
            print(EXPECTED_CLASSES)

            print("\nFound:")
            print(normalized)

    print("\nClip length:")
    print(
        checkpoint.get(
            "clip_length",
            "NOT STORED"
        )
    )

    print("\nImage size:")
    print(
        checkpoint.get(
            "image_size",
            "NOT STORED"
        )
    )

    print("\nState dictionary:")
    
    state = None

    if isinstance(
        checkpoint.get("model_state_dict"),
        dict
    ):
        state = checkpoint["model_state_dict"]
        print("Found: model_state_dict")

    elif isinstance(
        checkpoint.get("state_dict"),
        dict
    ):
        state = checkpoint["state_dict"]
        print("Found: state_dict")

    elif all(
        torch.is_tensor(v)
        for v in checkpoint.values()
    ):
        state = checkpoint
        print("Found: raw state dictionary")

    else:
        print("No usable state dictionary found.")

    if state is not None:

        print("\nNumber of parameters:")
        print(len(state))

        print("\nFirst 20 parameter names:")

        for key in list(state.keys())[:20]:
            print(key)

        print("\nFinal classifier parameters:")

        classifier_keys = [
            key
            for key in state.keys()
            if "classifier" in key or ".fc." in key
        ]

        for key in classifier_keys[-20:]:
            print(key)

else:
    print(
        "\nWARNING: Checkpoint is not a dictionary."
    )

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)