
from pathlib import Path
from email_alert import send_activity_email

BASE_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = BASE_DIR / "evidence"

images = sorted(
    [
        p for p in EVIDENCE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ],
    key=lambda p: p.stat().st_mtime,
    reverse=True
) if EVIDENCE_DIR.exists() else []

if not images:
    print("No evidence image found in evidence folder.")
else:
    print("Using evidence photo:", images[0].name)

    result = send_activity_email(
        activity="accident",
        confidence=94.0,
        risk=8.0,
        evidence_path=images[0],
        location="Mobile Camera - TEST"
    )

    print("Email sent:", result)
