
import base64
import mimetypes
from pathlib import Path
from email.message import EmailMessage

from dotenv import dotenv_values
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Gmail API permission
SCOPES = [
    "https://www.googleapis.com/auth/gmail.send"
]

# Private Google credentials
PRIVATE_DIR = Path(r"D:\Gmail_API_Private")
TOKEN_FILE = PRIVATE_DIR / "token.json"

# Project paths
BASE_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = BASE_DIR / "evidence"

# Gmail address from .env
config = dotenv_values(BASE_DIR / ".env")
GMAIL_ADDRESS = (config.get("GMAIL_SENDER") or "").strip()


def send_evidence_photo():
    print("====================================")
    print("GMAIL EVIDENCE PHOTO TEST")
    print("====================================")

    if not TOKEN_FILE.is_file():
        print("ERROR: token.json not found!")
        return

    if not GMAIL_ADDRESS:
        print("ERROR: GMAIL_SENDER missing in .env")
        return

    if not EVIDENCE_DIR.is_dir():
        print("ERROR: evidence folder not found!")
        return

    # Find the most recently modified evidence photo
    photos = [
        path for path in EVIDENCE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]

    if not photos:
        print("ERROR: No evidence photos found!")
        return

    latest_photo = max(
        photos,
        key=lambda path: path.stat().st_mtime
    )

    print("Evidence photo found:", latest_photo.name)

    try:
        # Load Google authorization
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
                print("ERROR: Authorization expired.")
                print("Run test_gmail_api.py again.")
                return

        # Connect to Gmail
        service = build(
            "gmail",
            "v1",
            credentials=credentials
        )

        print("Gmail API connected.")

        # Prepare email
        message = EmailMessage()
        message["From"] = GMAIL_ADDRESS
        message["To"] = GMAIL_ADDRESS
        message["Subject"] = (
            "AI Camera - Evidence Photo Test"
        )

        message.set_content(
            "This is a test email from my "
            "Unusual Activity Detection project.\n\n"
            "The attached image is an existing "
            "evidence photo from the project.\n\n"
            "TEST ONLY - No emergency was reported."
        )

        # Attach evidence photo
        mime_type, _ = mimetypes.guess_type(
            latest_photo.name
        )

        if not mime_type or not mime_type.startswith("image/"):
            print("ERROR: Unsupported image format.")
            return

        main_type, sub_type = mime_type.split("/", 1)

        with latest_photo.open("rb") as image_file:
            message.add_attachment(
                image_file.read(),
                maintype=main_type,
                subtype=sub_type,
                filename=latest_photo.name
            )

        # Encode email for Gmail API
        encoded_message = base64.urlsafe_b64encode(
            message.as_bytes()
        ).decode("utf-8")

        print("Sending email with evidence photo...")

        result = service.users().messages().send(
            userId="me",
            body={"raw": encoded_message}
        ).execute()

        print("====================================")
        print("SUCCESS: EMAIL WITH PHOTO SENT!")
        print("Photo:", latest_photo.name)
        print("Gmail message ID:", result.get("id"))
        print("Check your Gmail Inbox.")
        print("====================================")

    except Exception as error:
        print("ERROR:", type(error).__name__)
        print(str(error))


if __name__ == "__main__":
    send_evidence_photo()
