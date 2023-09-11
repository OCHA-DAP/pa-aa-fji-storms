import msal


def acquire_token_func():
    """
    Acquire token via MSAL
    """
    authority_url = "https://login.microsoftonline.com/unitednations"
    app = msal.ConfidentialClientApplication(
        authority=authority_url,
        client_id="{client_id}",
        client_credential="{client_secret}",
    )
    token = app.acquire_token_for_client(
        scopes=["https://graph.microsoft.com/.default"]
    )
    return token


def get_onedrive_file():
    # token = acquire_token_func()
    pass


def check_trigger():
    """
    Checks trigger, from GitHub Action
    Returns
    -------

    """
    print("checked trigger")


if __name__ == "__main__":
    check_trigger()
