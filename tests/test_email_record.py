"""Sent-email record: once per (forecast, CSV content, email type)."""

import pandas as pd

from src import email_record


def _capture(monkeypatch):
    saved = {}

    def upload(df, name, stage):
        saved.update(df=df.copy(), name=name, stage=stage)

    monkeypatch.setattr(email_record.stratus, "upload_csv_to_blob", upload)
    return saved


def test_already_sent_and_add(monkeypatch):
    saved = _capture(monkeypatch)
    record = pd.DataFrame(columns=email_record.COLUMNS)
    assert not email_record.already_sent(record, "f1", "abc", "forecast_info")

    record = email_record.add_sent(
        record, False, "f1", "abc", "forecast_info", 42
    )
    assert email_record.already_sent(record, "f1", "abc", "forecast_info")
    assert not email_record.already_sent(record, "f1", "abc", "readiness")
    assert not email_record.already_sent(record, "f2", "abc", "forecast_info")
    assert saved["name"] == email_record.RECORD_BLOB
    assert saved["stage"] == "dev"
    assert saved["df"]["campaign_id"].tolist() == ["42"]


def test_reexport_of_same_forecast_is_emailed_again(monkeypatch):
    _capture(monkeypatch)
    record = email_record.add_sent(
        pd.DataFrame(columns=email_record.COLUMNS),
        False,
        "f1",
        "abc",
        "forecast_info",
        1,
    )
    assert not email_record.already_sent(record, "f1", "def", "forecast_info")


def test_test_sends_use_their_own_record(monkeypatch):
    saved = _capture(monkeypatch)
    email_record.add_sent(
        pd.DataFrame(columns=email_record.COLUMNS), True, "f1", "", "action", 7
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


def test_loaded_record_round_trips(monkeypatch):
    csv = "forecast_id,content_sha12,email_type,campaign_id,sent_at_utc\n"
    csv += "f1,,forecast_info,3,t\n"
    monkeypatch.setattr(
        email_record.stratus,
        "load_csv_from_blob",
        lambda *a, **kw: pd.read_csv(pd.io.common.StringIO(csv), dtype=str),
    )
    record = email_record.load_record(False)
    assert email_record.already_sent(record, "f1", "", "forecast_info")
