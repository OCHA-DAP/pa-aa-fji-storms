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


def extract_email_groups(data):
    # Define column section names for clarity
    sections = {
        "trigger_to": ("TRIGGER_TO", "TRIGGER_CC"),
        "trigger_cc": ("TRIGGER_CC", "INFO_TO"),
        "info_to": ("INFO_TO", "INFO_CC"),
        "info_cc": ("INFO_CC", None),
    }

    # Initialize a dictionary to hold email lists for each section
    email_groups = {key: [] for key in sections.keys()}

    data = data.iloc[1:]  # Skip the first row (agency names)

    # Loop through each defined section
    for key, (start_col, end_col) in sections.items():
        # Slice the DataFrame between start_col and end_col
        if end_col:
            emails_section = data.loc[:, start_col:end_col].iloc[:, :-1]
        else:
            emails_section = data.loc[:, start_col:]

        # Flatten all emails in this section, remove NaNs, and add to the
        # corresponding list in email_groups
        emails_list = emails_section.values.flatten()
        email_groups[key] = [
            email.strip() for email in emails_list if pd.notna(email)
        ]

    return email_groups


def email_str_to_df(email_string):
    # Split the input string by "; " to get individual emails
    email_entries = email_string.split("; ")
    converted_emails = []

    for entry in email_entries:
        # Use regex to extract name and email address
        match = re.match(r"(.*) <(.*)>", entry)
        if match:
            name, email = match.groups()
            converted_emails.append([name, email])

    return pd.DataFrame(converted_emails, columns=["name", "email"])


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
