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
