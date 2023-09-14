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
message["Subject"] = "Test: Fiji AA Monitoring"
message["From"] = sender_email
message["To"] = ", ".join(mailing_list)

text = """\
Test.
"""
html = """\
<html>
    <style type = "text/css">
        h1, h2, h3 {
          font-family: Arvo, "Helvetica Neue", Helvetica, Arial, sans-serif;
          font-weight: normal
        }
        p {
          font-family: "Source Sans Pro", "Helvetica Neue", Helvetica, Arial,
          sans-serif
        }
    </style>
    <body>
        <div style="font-size:10px;padding: 6px 10px 6px 10px;
        background-color:#CCCCCC;color:#FFFFFF">
            <p style="margin:0;">
                The global monitoring and alert system is managed by the OCHA
                Centre for Humanitarian Data in collaboration with the UN
                Central Emergency Response Fund in order to bring attention to
                changes in key indicators relevant to humanitarian response.
            </p>
        </div>
    </body>
</html>
"""

part1 = MIMEText(text, "plain")
part2 = MIMEText(html, "html")

message.attach(part1)
message.attach(part2)

context = ssl.create_default_context()

with smtplib.SMTP_SSL(SERVER, PORT, context=context) as server:
    server.login(USERNAME, PASSWORD)
    server.sendmail(USERNAME, mailing_list, message.as_string())
