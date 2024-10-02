import os
import re

import pandas as pd

from src import blob

TEST_LIST = os.getenv("TEST_LIST")
if TEST_LIST == "False":
    TEST_LIST = False
else:
    TEST_LIST = True

SIMEX_LIST = os.getenv("SIMEX_LIST")
if SIMEX_LIST == "True":
    SIMEX_LIST = True
else:
    SIMEX_LIST = False


def is_valid_email(email):
    # Define a regex pattern for validating an email
    email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

    # Use the re.match() method to check if the email matches the pattern
    if re.match(email_regex, email):
        return True
    else:
        return False


def get_distribution_list() -> pd.DataFrame:
    """Load distribution list from blob storage."""
    if TEST_LIST:
        blob_name = f"{blob.PROJECT_PREFIX}/email/test_distribution_list.csv"
    else:
        if SIMEX_LIST:
            blob_name = (
                f"{blob.PROJECT_PREFIX}/email/simex_distribution_list.csv"
            )
        else:
            blob_name = f"{blob.PROJECT_PREFIX}/email/distribution_list.csv"
    df = blob.load_csv_from_blob(blob_name)
    df["name"] = df["name"].fillna("").astype(str)
    return df
