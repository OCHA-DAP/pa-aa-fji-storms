"""Record of the monitoring emails already sent, in the dev blob.

Each forecast still gets its emails (info, plus readiness/action when
triggered), but at most once per (forecast_id, email_type): a retried or
re-triggered run for the same forecast skips what already went out.

Test sends (TEST_EMAIL or SIMULATE_TRIGGER) use a separate record, so a
test can never mark a real forecast as sent. Under DRY_RUN nothing is sent,
so nothing is recorded.
"""

from datetime import datetime, timezone

import ocha_stratus as stratus
import pandas as pd
from azure.core.exceptions import ResourceNotFoundError

from src.blob import PROJECT_PREFIX

RECORD_BLOB = f"{PROJECT_PREFIX}/monitoring/email_record.csv"
TEST_RECORD_BLOB = f"{PROJECT_PREFIX}/monitoring/test_email_record.csv"
COLUMNS = ["forecast_id", "email_type", "campaign_id", "sent_at_utc"]


def record_blob(test: bool) -> str:
    return TEST_RECORD_BLOB if test else RECORD_BLOB


def load_record(test: bool) -> pd.DataFrame:
    try:
        df = stratus.load_csv_from_blob(record_blob(test), stage="dev")
    except ResourceNotFoundError:
        return pd.DataFrame(columns=COLUMNS)
    return df.reindex(columns=COLUMNS)


def already_sent(
    record: pd.DataFrame, forecast_id: str, email_type: str
) -> bool:
    return bool(
        (
            (record["forecast_id"] == forecast_id)
            & (record["email_type"] == email_type)
        ).any()
    )


def add_sent(
    record: pd.DataFrame,
    test: bool,
    forecast_id: str,
    email_type: str,
    campaign_id: int,
) -> pd.DataFrame:
    """Append one sent email and save the record straight away."""
    row = {
        "forecast_id": forecast_id,
        "email_type": email_type,
        "campaign_id": campaign_id,
        "sent_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
    }
    record = pd.concat([record, pd.DataFrame([row])], ignore_index=True)
    stratus.upload_csv_to_blob(record, record_blob(test), stage="dev")
    return record
