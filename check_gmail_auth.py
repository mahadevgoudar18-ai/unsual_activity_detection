
import smtplib
import ssl

try:
    print("Connecting to Gmail...")

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
        server.ehlo()
        server.starttls(context=ssl.create_default_context())
        server.ehlo()

        print("Secure connection: SUCCESS")
        print("Authentication methods:", server.esmtp_features.get("auth", "Not advertised"))
        print("SMTP server ready.")

except Exception as error:
    print("Error type:", type(error).__name__)
    print("Error:", str(error))
