from src import blob


def load_simex_inject(inject_number: int):
    blob_name = (
        f"{blob.PROJECT_PREFIX}/simex/inject_forecast_{inject_number}.csv"
    )
    df = blob.load_csv_from_blob(blob_name)
    return df
