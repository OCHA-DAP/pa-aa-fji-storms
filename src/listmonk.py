import mimetypes
import os

import requests

BASE_URL = "https://listmonk-demo-afhcg8e2hde0fxca.eastus2-01.azurewebsites.net/api"  # noqa

DSCI_LISTMONK_API_USERNAME = os.getenv("DSCI_LISTMONK_API_USERNAME")
DSCI_LISTMONK_API_KEY = os.getenv("DSCI_LISTMONK_API_KEY")
AUTH = (DSCI_LISTMONK_API_USERNAME, DSCI_LISTMONK_API_KEY)

TRISTAN_ONLY_LIST_ID = 5
DSCI_LIST_ID = 6
CERF_INFO_LIST_ID = 14
CERF_TRIGGER_LIST_ID = 15
FRAMEWORK_INFO_LIST_ID = 11
FRAMEWORK_TRIGGER_LIST_ID = 10

TEST_LIST_IDS = [TRISTAN_ONLY_LIST_ID]
PROD_INFO_LIST_IDS = [FRAMEWORK_INFO_LIST_ID, CERF_INFO_LIST_ID, DSCI_LIST_ID]
PROD_TRIGGER_LIST_IDS = [
    FRAMEWORK_TRIGGER_LIST_ID,
    CERF_TRIGGER_LIST_ID,
    DSCI_LIST_ID,
]

BASE_CAMPAIGN_ID = 8


def create_campaign(
    name: str = "test_campaign",
    subject: str = "Test Subject",
    list_ids: list[int] = None,
    template_id: int = BASE_CAMPAIGN_ID,
    body: str = "",
    media: list[int] = None,
):
    if list_ids is None:
        list_ids = TEST_LIST_IDS
    create_payload = {
        "name": name,
        "subject": subject,
        "lists": list_ids,
        "template_id": template_id,
        "type": "regular",
        "content_type": "html",
        "body": body,
        "media": media or [],
    }

    r = requests.post(
        f"{BASE_URL}/campaigns",
        auth=AUTH,
        json=create_payload,
    )

    r.raise_for_status()
    campaign = r.json()["data"]
    campaign_id = campaign["id"]
    return campaign_id


def send_campaign(campaign_id: int):
    r = requests.put(
        f"{BASE_URL}/campaigns/{campaign_id}/status",
        auth=AUTH,
        json={"status": "running"},
    )
    r.raise_for_status()


def create_and_send_campaign(
    name: str = "test_campaign",
    subject: str = "Test Subject",
    list_ids: list[int] = None,
    template_id: int = BASE_CAMPAIGN_ID,
    body: str = "",
    media: list[int] = None,
):
    campaign_id = create_campaign(
        name=name,
        subject=subject,
        list_ids=list_ids,
        template_id=template_id,
        body=body,
        media=media,
    )
    send_campaign(campaign_id)
    return campaign_id


def upload_file(file_path):
    # Guess the MIME type based on the file extension
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # Fallback

    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, mime_type)}  # include MIME type
        r = requests.post(
            f"{BASE_URL}/media",
            auth=AUTH,
            files=files,
        )
    r.raise_for_status()
    return r.json()["data"]
