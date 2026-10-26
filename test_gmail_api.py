
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Gmail permission: send emails only
SCOPES = [
    "https://www.googleapis.com/auth/gmail.send"
]

# Private credentials folder on Local Disk D
PRIVATE_DIR = Path(r"D:\Gmail_API_Private")

CREDENTIALS_FILE = PRIVATE_DIR / "credentials.json"
TOKEN_FILE = PRIVATE_DIR / "token.json"


def test_gmail_api():
    print("===================================")
    print("GMAIL API CONNECTION TEST")
    print("===================================")

    if not CREDENTIALS_FILE.is_file():
        print("ERROR: credentials.json not found!")
        print("Expected location:", CREDENTIALS_FILE)
        return

    try:
        print("Opening Google sign-in...")

        flow = InstalledAppFlow.from_client_secrets_file(
            str(CREDENTIALS_FILE),
            SCOPES
        )

        credentials = flow.run_local_server(
            port=0,
            open_browser=True
        )

        # Save authorization for future use
        TOKEN_FILE.write_text(
            credentials.to_json(),
            encoding="utf-8"
        )

        service = build(
            "gmail",
            "v1",
            credentials=credentials
        )

        print("SUCCESS: Gmail API connected!")
        print("Authorization saved privately.")
        print("No email was sent.")

    except Exception as error:
        print("ERROR:", type(error).__name__)
        print(str(error))


if __name__ == "__main__":
    test_gmail_api()
