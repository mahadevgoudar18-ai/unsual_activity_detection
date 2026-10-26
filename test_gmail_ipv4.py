
import socket
import smtplib
import ssl
from pathlib import Path
from dotenv import dotenv_values

BASE_DIR = Path(__file__).resolve().parent
config = dotenv_values(BASE_DIR / ".env")

sender = (config.get("GMAIL_SENDER") or "").strip()
password = (
    config.get("GMAIL_APP_PASSWORD") or ""
).replace(" ", "").strip()


def test_ipv4_gmail():
    if not sender or len(password) != 16:
        print("Check your Gmail configuration in .env")
        return

    original_getaddrinfo = socket.getaddrinfo

    def ipv4_only(host, port, *args, **kwargs):
        return original_getaddrinfo(
            host,
            port,
            socket.AF_INET,
            socket.SOCK_STREAM
        )

    try:
        print("Connecting to Gmail using IPv4...")

        socket.getaddrinfo = ipv4_only

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

            print("IPv4 secure connection established.")
            print("Attempting Gmail authentication...")

            server.login(sender, password)

            print("SUCCESS: Gmail authentication completed!")
            print("No email was sent.")

    except smtplib.SMTPAuthenticationError as error:
        print("Gmail rejected authentication.")
        print("SMTP response code:", error.smtp_code)

    except smtplib.SMTPServerDisconnected:
        print("Gmail disconnected during authentication.")

    except Exception as error:
        print("Error:", type(error).__name__)
        print(str(error))

    finally:
        socket.getaddrinfo = original_getaddrinfo


if __name__ == "__main__":
    test_ipv4_gmail()
