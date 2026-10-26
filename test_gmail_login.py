
import smtplib
import ssl
from pathlib import Path
from dotenv import dotenv_values

# Load Gmail configuration from .env
BASE_DIR = Path(__file__).resolve().parent
config = dotenv_values(BASE_DIR / ".env")

SENDER = (config.get("GMAIL_SENDER") or "").strip()
PASSWORD = (
    (config.get("GMAIL_APP_PASSWORD") or "")
    .replace(" ", "")
    .strip()
)


def test_gmail_plain():
    print("====================================")
    print("GMAIL AUTHENTICATION TEST")
    print("====================================")

    if not SENDER or len(PASSWORD) != 16:
        print("ERROR: Check Gmail settings in .env")
        return

    try:
        print("Connecting to Gmail...")

        with smtplib.SMTP(
            "smtp.gmail.com",
            587,
            timeout=30
        ) as server:

            server.ehlo()

            server.starttls(
                context=ssl.create_default_context()
            )

            server.ehlo()

            print("Secure connection established.")
            print("Testing PLAIN authentication...")

            server.user = SENDER
            server.password = PASSWORD

            server.auth(
                "PLAIN",
                server.auth_plain,
                initial_response_ok=True
            )

            print("SUCCESS: Gmail authentication completed!")
            print("No email was sent.")

    except smtplib.SMTPAuthenticationError as error:
        print("ERROR: Gmail rejected authentication.")
        print("SMTP response code:", error.smtp_code)

        response = (
            error.smtp_error.decode(
                "utf-8",
                errors="replace"
            )
            if isinstance(error.smtp_error, bytes)
            else str(error.smtp_error)
        )

        print("Gmail explanation:", response)

    except smtplib.SMTPServerDisconnected:
        print("ERROR: Gmail disconnected during authentication.")

    except Exception as error:
        print("ERROR:", type(error).__name__)
        print(str(error))

    print("====================================")
    print("TEST COMPLETED")
    print("====================================")


if __name__ == "__main__":
    test_gmail_plain()
