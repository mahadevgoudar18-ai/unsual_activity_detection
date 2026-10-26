
import base64
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

# Private Google authentication files
PRIVATE_DIR = Path(r"D:\Gmail_API_Private")
TOKEN_FILE = PRIVATE_DIR / "token.json"

# Read Gmail address from existing .env file
BASE_DIR = Path(__file__).resolve().parent
config = dotenv_values(BASE_DIR / ".env")
GMAIL_ADDRESS = (config.get("GMAIL_SENDER") or "").strip()


def send_test_email():

    print("====================================")
    print("GMAIL API EMAIL SENDING TEST")
    print("====================================")

    if not TOKEN_FILE.is_file():
        print("ERROR: token.json not found!")
        return

    if not GMAIL_ADDRESS:
        print("ERROR: GMAIL_SENDER missing in .env")
        return

    try:
        print("Loading Google authorization...")

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
                print("ERROR: Google authorization expired.")
                print("Run test_gmail_api.py again.")
                return

        print("Connecting to Gmail API...")

        service = build(
            "gmail",
            "v1",
            credentials=credentials
        )

        print("Gmail API connected.")
        print("Preparing test email...")

        # Send test email to the same Gmail account
        message = EmailMessage()

        message["To"] = GMAIL_ADDRESS
        message["From"] = GMAIL_ADDRESS
        message["Subject"] = (
            "Unusual Activity Detection - Gmail API Test"
        )

        message.set_content(
            "Hello!\n\n"
            "This is a test email from my "
            "Unusual Activity Detection project.\n\n"
            "Gmail API authentication and email sending "
            "are being tested.\n\n"
            "This is only a test.\n"
            "No unusual activity was detected.\n"
            "No emergency services were contacted."
        )

        # Encode email
        encoded_message = base64.urlsafe_b64encode(
            message.as_bytes()
        ).decode("utf-8")

        print("Sending test email...")

        result = service.users().messages().send(
            userId="me",
            body={
                "raw": encoded_message
            }
        ).execute()

        print("====================================")
        print("SUCCESS: TEST EMAIL SENT!")
        print("Gmail message ID:", result.get("id"))
        print("Check your Gmail Inbox or Sent folder.")
        print("====================================")

    except Exception as error:
        print("ERROR:", type(error).__name__)
        print(str(error))


if __name__ == "__main__":
    send_test_email()
