"""Parameter checks in the Databricks task wrappers."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "databricks" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_ingest = _load("run_ingest")
run_monitor = _load("run_monitor")

BLOB = "pa-aa-fji-storms/raw/fms/inbox/abc123_forecast.csv"


def test_live_event_keeps_mode():
    assert (
        run_ingest.check_params(BLOB, "fms_forecast", "prod", "main") == "prod"
    )
    assert (
        run_ingest.check_params(BLOB, "fms_forecast", "dev", "main") == "dev"
    )


def test_test_event_always_dev():
    assert (
        run_ingest.check_params(BLOB, "fms_forecast_test", "prod", "main")
        == "dev"
    )


@pytest.mark.parametrize(
    "blob, event, mode, ref",
    [
        (
            "pa-aa-fji-storms/monitoring/email_record.csv",
            "fms_forecast",
            "dev",
            "main",
        ),
        (
            "pa-aa-fji-storms/raw/fms/inbox/../../x.csv",
            "fms_forecast",
            "dev",
            "main",
        ),
        (BLOB, "fms_forecast_live", "dev", "main"),
        (BLOB, "fms_forecast", "staging", "main"),
        (BLOB, "fms_forecast", "dev", "--upload-pack=x"),
        (BLOB, "fms_forecast", "dev", "a/../b"),
        ("", "fms_forecast", "dev", "main"),
    ],
)
def test_bad_params_rejected(blob, event, mode, ref):
    with pytest.raises(ValueError):
        run_ingest.check_params(blob, event, mode, ref)


@pytest.mark.parametrize(
    "value, expected",
    [("True", True), ("false", False), ("1", True), ("no", False)],
)
def test_env_flag(value, expected):
    assert run_monitor.env_flag("x", value) is expected


@pytest.mark.parametrize("value", ["", "maybe", "Tru"])
def test_env_flag_strict(value):
    with pytest.raises(ValueError):
        run_monitor.env_flag("x", value)


INGESTED = {
    "status": "ingested",
    "stage": "prod",
    "disturbance_id": "2526-11F",
    "track_type": "Official Forecast Track",
    "issued_time": "2026-04-07T00:00:00",
    "content_sha256": "ab" * 32,
}


def test_monitor_reads_db_after_ingest():
    args = run_monitor.monitor_input(INGESTED, BLOB)
    assert args[:2] == ["--from-db", "--disturbance-id"]
    assert "2526-11F" in args and "prod" in args
    assert args[-2:] == ["--content-sha", "ab" * 32]


@pytest.mark.parametrize("track_type", ["GP Forecast Track", ""])
def test_monitor_refuses_non_official_track(track_type):
    values = {**INGESTED, "track_type": track_type}
    with pytest.raises(ValueError, match="not monitored"):
        run_monitor.monitor_input(values, BLOB)


def test_monitor_quiet_only_for_skipped_file():
    assert run_monitor.monitor_input({"status": "skipped"}, BLOB) is None


@pytest.mark.parametrize("values", [{}, {"status": ""}, {"status": "weird"}])
def test_monitor_falls_back_to_staged_csv(values):
    assert run_monitor.monitor_input(values, BLOB) == ["--csv-blob", BLOB]


@pytest.mark.parametrize(
    "blob", ["", "pa-aa-fji-storms/monitoring/x.csv", BLOB + "/../../x"]
)
def test_fallback_refuses_blobs_outside_inbox(blob):
    with pytest.raises(ValueError):
        run_monitor.monitor_input({}, blob)
