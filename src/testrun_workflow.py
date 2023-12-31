import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

YASA = os.getenv("YASA")
OWNER = "OCHA-DAP"
REPO = "pa-aa-fji-storms"
WORKFLOW_ID = "check-trigger.yml"
GH_ACTIONS_TOKEN = os.getenv("GH_ACTIONS_TOKEN")


def testrun_workflow():
    """Sends POST to GitHub Actions REST API to manually trigger workflow.
    This simulates what is done in Power Automate.
    Returns
    -------

    """
    url = (
        f"https://api.github.com/repos/"
        f"{OWNER}/{REPO}/actions/workflows/{WORKFLOW_ID}/dispatches"
    )
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GH_ACTIONS_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    body = {"ref": "add-trigger", "inputs": {"csv": YASA}}
    response = requests.post(url=url, headers=headers, data=json.dumps(body))
    print(response.status_code)


if __name__ == "__main__":
    testrun_workflow()
