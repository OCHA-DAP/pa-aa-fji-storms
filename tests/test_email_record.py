"""Sent-email record: at most once per (forecast, email type)."""

import pandas as pd

from src import email_record


def test_already_sent_and_add(monkeypatch):
    saved = {}
    monkeypatch.setattr(
        email_record.stratus,
        "upload_csv_to_blob",
        lambda df, name, stage: saved.update(df=df, name=name, stage=stage),
    )
    record = pd.DataFrame(columns=email_record.COLUMNS)
    assert not email_record.already_sent(record, "f1", "forecast_info")

    record = email_record.add_sent(record, False, "f1", "forecast_info", 42)
    assert email_record.already_sent(record, "f1", "forecast_info")
    assert not email_record.already_sent(record, "f1", "readiness")
    assert not email_record.already_sent(record, "f2", "forecast_info")
    assert saved["name"] == email_record.RECORD_BLOB
    assert saved["stage"] == "dev"
    assert saved["df"]["campaign_id"].tolist() == [42]


def test_test_sends_use_their_own_record(monkeypatch):
    saved = {}
    monkeypatch.setattr(
        email_record.stratus,
        "upload_csv_to_blob",
        lambda df, name, stage: saved.update(name=name),
    )
    email_record.add_sent(
        pd.DataFrame(columns=email_record.COLUMNS), True, "f1", "action", 7
    )
    assert saved["name"] == email_record.TEST_RECORD_BLOB
    assert email_record.TEST_RECORD_BLOB != email_record.RECORD_BLOB


def test_missing_record_is_empty(monkeypatch):
    from azure.core.exceptions import ResourceNotFoundError

    def missing(*args, **kwargs):
        raise ResourceNotFoundError("nope")

    monkeypatch.setattr(email_record.stratus, "load_csv_from_blob", missing)
    record = email_record.load_record(False)
    assert list(record.columns) == email_record.COLUMNS
    assert record.empty
