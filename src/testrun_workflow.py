import json
import os

import requests

OWNER = "OCHA-DAP"
REPO = "pa-aa-fji-storms"
WORKFLOW_ID = "check-trigger.yml"
GH_ACTIONS_TOKEN = os.getenv("GH_ACTIONS_TOKEN")


def testrun_workflow():
    """
    Sends POST to GitHub Actions REST API to manually trigger workflow.
    This simulates what is done in Power Automate.
    Returns
    -------

    """
    url = (
        f"https://api.github.com/repos/"
        f"{OWNER}/{REPO}/actions/workflows/{WORKFLOW_ID}/dispatches"
    )
    print(url)
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GH_ACTIONS_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    body = {"ref": "add-trigger"}
    response = requests.post(url=url, headers=headers, data=json.dumps(body))
    print(response.content)


if __name__ == "__main__":
    testrun_workflow()
