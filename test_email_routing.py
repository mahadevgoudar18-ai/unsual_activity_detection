
from pathlib import Path
from email_alert import send_activity_email

# Project folder
BASE_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = BASE_DIR / "evidence"


def test_email_routing():

    print("====================================")
    print("AI CAMERA - EMAIL ROUTING TEST")
    print("====================================")

    if not EVIDENCE_DIR.exists():
        print("ERROR: Evidence folder not found.")
        return

    # Find existing evidence images
    images = [
        file for file in EVIDENCE_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]

    if not images:
        print("ERROR: No evidence photos found.")
        return

    # Select latest evidence photo
    evidence_photo = max(
        images,
        key=lambda file: file.stat().st_mtime
    )

    print("Evidence photo:", evidence_photo.name)

    print("\nAvailable tests:")
    print("1. Accident -> Police + Medical")
    print("2. Fire/Smoke -> Fire")
    print("3. Fighting -> Police")
    print("4. Normal -> No email")

    choice = input("\nEnter test number (1-4): ").strip()

    activities = {
        "1": "accident",
        "2": "fire_smoke",
        "3": "fighting",
        "4": "normal"
    }

    if choice not in activities:
        print("Invalid choice.")
        return

    activity = activities[choice]

    print("\nSelected activity:", activity)
    print("This is a DEMO TEST, not a real emergency.")

    if activity == "normal":
        result = send_activity_email(
            activity="normal",
            confidence=95.0,
            risk=0.0,
            evidence_path=evidence_photo,
            location="TEST - Mobile Camera"
        )

        print("Normal activity processed.")
        print("Email sent:", result)
        return

    confirmation = input(
        "Send one DEMO alert to the configured "
        "recipient(s)? (yes/no): "
    ).strip().lower()

    if confirmation != "yes":
        print("Test cancelled.")
        return

    result = send_activity_email(
        activity=activity,
        confidence=94.0,
        risk=8.0,
        evidence_path=evidence_photo,
        location="TEST - Mobile Camera"
    )

    print("\n====================================")
    print("TEST RESULT")
    print("====================================")
    print("Activity:", activity)
    print("Email sent:", result)


if __name__ == "__main__":
    test_email_routing()
