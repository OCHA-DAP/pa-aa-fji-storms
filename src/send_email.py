import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr

from jinja2 import Environment, FileSystemLoader

trigger_type = "action"

PORT = 465  # For SSL
PASSWORD = os.getenv("G_P_APP_PWD")
USERNAME = os.getenv("G_P_ACCOUNT")
sender_email = formataddr(("OCHA Centre for Humanitarian Data", USERNAME))
SERVER = os.getenv("G_P_SERVER")
mailing_list = ["tristan.downing@un.org"]

environment = Environment(loader=FileSystemLoader("src/email/"))
if trigger_type == "action":
    template = environment.get_template("action.html")
else:
    template = environment.get_template("readiness.html")

message = MIMEMultipart("alternative")
message[
    "Subject"
] = f"Anticipatory action Fiji – {trigger_type.capitalize()} trigger reached"
message["From"] = sender_email
message["To"] = ", ".join(mailing_list)

html = template.render()
html_part = MIMEText(html, "html")
message.attach(html_part)

context = ssl.create_default_context()

with smtplib.SMTP_SSL(SERVER, PORT, context=context) as server:
    server.login(USERNAME, PASSWORD)
    server.sendmail(USERNAME, mailing_list, message.as_string())
