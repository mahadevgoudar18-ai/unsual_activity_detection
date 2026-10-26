
import base64
import mimetypes
from pathlib import Path
from datetime import datetime
from email.message import EmailMessage

from dotenv import dotenv_values
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# -----------------------------------
# PROJECT CONFIGURATION
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent
PRIVATE_DIR = Path(r"D:\Gmail_API_Private")
TOKEN_FILE = PRIVATE_DIR / "token.json"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.send"
]

# Read Gmail settings
config = dotenv_values(BASE_DIR / ".env")

GMAIL_SENDER = (
    config.get("GMAIL_SENDER") or ""
).strip()

RECIPIENTS = {
    "police": (
        config.get("POLICE_DEMO_EMAIL") or ""
    ).strip(),

    "medical": (
        config.get("MEDICAL_DEMO_EMAIL") or ""
    ).strip(),

    "fire": (
        config.get("FIRE_DEMO_EMAIL") or ""
    ).strip()
}

# -----------------------------------
# ACTIVITY EMAIL ROUTING
# -----------------------------------

ACTIVITY_ROUTING = {
    "accident": ["police", "medical"],
    "falling": ["police", "medical"],
    "fighting": ["police"],
    "theft": ["police"],
    "shooting": ["police"],
    "vandalism": ["police"],
    "fire_smoke": ["fire"],
    "normal": []
}


# -----------------------------------
# CONNECT TO GMAIL API
# -----------------------------------

def connect_gmail():

    if not TOKEN_FILE.is_file():
        raise FileNotFoundError(
            "Gmail token.json not found."
        )

    credentials = Credentials.from_authorized_user_file(
        str(TOKEN_FILE),
        SCOPES
    )

    if not credentials.valid:

        if credentials.expired and credentials.refresh_token:

            credentials.refresh(Request())

            TOKEN_FILE.write_text(
                credentials.to_json(),
                encoding="utf-8"
            )

        else:
            raise RuntimeError(
                "Gmail authorization expired. "
                "Run test_gmail_api.py again."
            )

    return build(
        "gmail",
        "v1",
        credentials=credentials
    )


# -----------------------------------
# SEND ACTIVITY EMAIL
# -----------------------------------

def send_activity_email(
    activity,
    confidence,
    risk,
    evidence_path,
    location="Mobile Camera"
):

    activity = str(activity).lower().strip()

    print("------------------------------------")
    print("ACTIVITY EMAIL ALERT")
    print("------------------------------------")
    print("Detected activity:", activity)

    # Check activity
    if activity not in ACTIVITY_ROUTING:
        print("Unknown activity. Email not sent.")
        return False

    # No email for normal activity
    if activity == "normal":
        print("Normal activity. No email required.")
        return False

    # Check sender
    if not GMAIL_SENDER:
        print("ERROR: GMAIL_SENDER missing.")
        return False

    # Check evidence image
    evidence_file = Path(evidence_path)

    if not evidence_file.is_file():
        print("ERROR: Evidence photo not found.")
        return False

    # Find recipients
    roles = ACTIVITY_ROUTING[activity]

    email_addresses = []

    for role in roles:
        address = RECIPIENTS.get(role, "")

        if not address:
            print("ERROR: Missing email for", role)
            return False

        if address not in email_addresses:
            email_addresses.append(address)

    if not email_addresses:
        print("No recipients configured.")
        return False

    try:
        service = connect_gmail()

        # Create email
        message = EmailMessage()

        message["From"] = GMAIL_SENDER
        message["To"] = ", ".join(email_addresses)

        message["Subject"] = (
            "[DEMO ALERT] AI Camera - "
            + activity.upper().replace("_", " ")
        )

        timestamp = datetime.now().strftime(
            "%d-%m-%Y %I:%M:%S %p"
        )

        message.set_content(
            "UNUSUAL ACTIVITY DETECTION SYSTEM\n"
            "----------------------------------\n\n"
            "DEMONSTRATION ALERT - HUMAN REVIEW REQUIRED\n"
            "This is an AI-generated prediction, "
            "not a verified emergency.\n\n"
            f"Detected Activity: {activity.upper()}\n"
            f"Confidence: {confidence}%\n"
            f"Risk Score: {risk}\n"
            f"Camera Location: {location}\n"
            f"Date and Time: {timestamp}\n\n"
            "An evidence photo is attached.\n\n"
            "Please review the evidence before "
            "taking any action.\n"
            "Do not treat this demo email as an "
            "emergency-services notification."
        )

        # Attach actual evidence photo
        mime_type, _ = mimetypes.guess_type(
            evidence_file.name
        )

        if not mime_type or not mime_type.startswith("image/"):
            print("ERROR: Invalid image format.")
            return False

        main_type, sub_type = mime_type.split("/", 1)

        with evidence_file.open("rb") as image_file:

            message.add_attachment(
                image_file.read(),
                maintype=main_type,
                subtype=sub_type,
                filename=evidence_file.name
            )

        # Encode message
        encoded_message = base64.urlsafe_b64encode(
            message.as_bytes()
        ).decode("utf-8")

        # Send email
        result = service.users().messages().send(
            userId="me",
            body={"raw": encoded_message}
        ).execute()

        print("SUCCESS: Demo alert email sent!")
        print("Activity:", activity)
        print("Recipients:", ", ".join(email_addresses))
        print("Evidence:", evidence_file.name)
        print("Message ID:", result.get("id"))

        return True

    except Exception as error:
        print("ERROR:", type(error).__name__)
        print(str(error))
        return False


if __name__ == "__main__":
    print("Gmail API alert module is ready.")
    print("Automatic camera alerts are not enabled yet.")
