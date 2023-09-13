import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr

PORT = 465  # For SSL
PASSWORD = os.getenv("G_P_APP_PWD")
USERNAME = os.getenv("G_P_ACCOUNT")
sender_email = formataddr(("OCHA CHD Data Science", USERNAME))
SERVER = os.getenv("G_P_SERVER")
mailing_list = ["tristan.downing@un.org"]

message = MIMEMultipart("alternative")
message["Subject"] = "Test"
message["From"] = sender_email
message["To"] = ", ".join(mailing_list)

text = """\
Test.
"""
html = """\
<html>
  <body>
    <p>
      Test.
    </p>
  </body>
</html>
"""

part1 = MIMEText(text, "plain")
part2 = MIMEText(html, "html")

message.attach(part1)
message.attach(part2)

# Create a secure SSL context
context = ssl.create_default_context()

with smtplib.SMTP_SSL(SERVER, PORT, context=context) as server:
    server.login(USERNAME, PASSWORD)
    server.sendmail(USERNAME, mailing_list, message.as_string())
