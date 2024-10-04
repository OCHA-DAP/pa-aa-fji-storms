import base64

from src import blob


def load_simex_inject(inject_number: int):
    blob_name = (
        f"{blob.PROJECT_PREFIX}/simex/inject_forecast_{inject_number}.csv"
    )
    blob_data = blob.load_blob_data(blob_name)
    base64_string = base64.b64encode(blob_data).decode("utf-8")
    return base64_string
