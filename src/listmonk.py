import os

import requests

BASE_URL = "https://listmonk-demo-afhcg8e2hde0fxca.eastus2-01.azurewebsites.net/api"  # noqa

DSCI_LISTMONK_API_USERNAME = os.getenv("DSCI_LISTMONK_API_USERNAME")
DSCI_LISTMONK_API_KEY = os.getenv("DSCI_LISTMONK_API_KEY")
AUTH = (DSCI_LISTMONK_API_USERNAME, DSCI_LISTMONK_API_KEY)

TRISTAN_ONLY_LIST_ID = 5
TEST_LIST_ID = TRISTAN_ONLY_LIST_ID

BASE_CAMPAIGN_ID = 8


def create_campaign(
    name: str = "test_campaign",
    subject: str = "Test Subject",
    list_ids: list[int] = None,
    template_id: int = BASE_CAMPAIGN_ID,
    body: str = "",
):
    if list_ids is None:
        list_ids = [TEST_LIST_ID]
    create_payload = {
        "name": name,
        "subject": subject,
        "lists": list_ids,
        "template_id": template_id,
        "type": "regular",
        "content_type": "html",
        "body": body,
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
):
    campaign_id = create_campaign(
        name=name,
        subject=subject,
        list_ids=list_ids,
        template_id=template_id,
        body=body,
    )
    send_campaign(campaign_id)
    return campaign_id
