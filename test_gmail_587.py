
import os
import ssl
import smtplib
from pathlib import Path
from dotenv import load_dotenv

# Load Gmail configuration from .env
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

SENDER = os.getenv("GMAIL_SENDER", "").strip()
APP_PASSWORD = os.getenv(
    "GMAIL_APP_PASSWORD", ""
).replace(" ", "").strip()


def test_gmail_connection():

    print("=" * 45)
    print("GMAIL SMTP CONNECTION TEST")
    print("=" * 45)

    # Check configuration
    if not SENDER:
        print("ERROR: GMAIL_SENDER is missing in .env")
        return

    if not APP_PASSWORD:
        print("ERROR: GMAIL_APP_PASSWORD is missing in .env")
        return

    if len(APP_PASSWORD) != 16:
        print(
            "WARNING: App Password should contain "
            "16 characters without spaces."
        )
        return

    print("Gmail configuration found.")
    print("App Password format: OK")
    print("Connecting to Gmail on port 587...")

    try:
        context = ssl.create_default_context()

        with smtplib.SMTP(
            "smtp.gmail.com",
            587,
            timeout=30
        ) as server:

            # Connect to Gmail
            server.ehlo()

            # Start secure TLS connection
            server.starttls(context=context)
            server.ehlo()

            print("Secure connection established.")
            print("Attempting Gmail login...")

            try:
                server.login(SENDER, APP_PASSWORD)

                print()
                print("SUCCESS: Gmail login successful!")
                print("Your Gmail is ready for email alerts.")
                print("No email was sent.")

            except smtplib.SMTPAuthenticationError as error:
                print()
                print("ERROR: Gmail authentication failed.")
                print(
                    "Check your Gmail address and "
                    "16-character App Password."
                )
                print("SMTP response code:", error.smtp_code)

            except smtplib.SMTPServerDisconnected:
                print()
                print(
                    "ERROR: Gmail disconnected "
                    "during authentication."
                )
                print(
                    "Possible causes include account "
                    "restrictions or network interference."
                )
                print(
                    "Check Google Account security activity."
                )

    except smtplib.SMTPConnectError:
        print("ERROR: Could not connect to Gmail SMTP.")

    except smtplib.SMTPServerDisconnected:
        print(
            "ERROR: Gmail unexpectedly closed "
            "the SMTP connection."
        )

    except TimeoutError:
        print("ERROR: Gmail connection timed out.")

    except ssl.SSLError as error:
        print("ERROR: SSL connection failed.")
        print(type(error).__name__)

    except Exception as error:
        print("ERROR:", type(error).__name__)
        print(str(error))

    print()
    print("=" * 45)
    print("TEST COMPLETED")
    print("=" * 45)


if __name__ == "__main__":
    test_gmail_connection()
