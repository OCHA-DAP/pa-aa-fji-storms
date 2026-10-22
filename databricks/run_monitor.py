"""Task 2 of the Fiji Cyclone Monitoring job: the trigger monitor.

Runs ``pipelines/monitor_forecast_trigger.py`` (trigger check, readiness /
action / info emails through Listmonk) on the forecast the ingest task
wrote, read back from the database (``--from-db``).

The task runs even when ingest fails (``run_if: ALL_DONE``): then it falls
back to the staged CSV itself, the way the GitHub Actions monitor works, so
a failed git clone, parser rejection or DB write never stops the alert. The
job still ends FAILED (the ingest task failed), so the failure email goes
out. Only a file the ingest recognised as not a forecast (``skipped``, e.g.
an Analysis Track) ends this task quietly. A forecast track that isn't the
Official one (FMS's archive also holds "GP Forecast Track" files) is not
monitored: the trigger is defined on the Official forecast, so the task
fails and the failure email reports it.

Positional parameters (databricks.yml):
    1 event                   fms_forecast | fms_forecast_test (test forces
                              TEST_EMAIL=True)
    2 test_email              True | False (from the poller)
    3 csv_blob                the staged CSV (fallback input)
    4 dry_run                 True | False (fixed at deploy)
    5 simulate_trigger        True | False (fixed at deploy)
    6 max_forecast_age_hours  no real-list emails for older forecasts

src/ and pipelines/ are copied off the workspace mount to local disk first
(imports from the wsfs clone are intermittently flaky). Never ``sys.exit``
here: the DBX exec context treats even ``SystemExit(0)`` as a failure.
"""

import os
import shutil
import subprocess
import sys
import tempfile

TEST_EVENT = "fms_forecast_test"
EVENTS = ("fms_forecast", TEST_EVENT)
# Must match pipelines/poll_fms_inbox.py and databricks/run_ingest.py
INBOX_PREFIX = "pa-aa-fji-storms/raw/fms/inbox/"
# Must match src/datasources/fms_db.py
OFFICIAL_TRACK = "Official Forecast Track"
LISTMONK_SECRETS = (
    "DSCI_LISTMONK_BASE_URL",
    "DSCI_LISTMONK_API_USERNAME",
    "DSCI_LISTMONK_API_KEY",
)


def _find_script_dir() -> str:
    """spark_python_task's exec context doesn't always define __file__."""
    try:
        return os.path.dirname(os.path.abspath(__file__))  # noqa: F821
    except NameError:
        pass
    if sys.argv and sys.argv[0]:
        return os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.getcwd()


def _arg(i: int) -> str:
    return sys.argv[i].strip() if len(sys.argv) > i else ""


def env_flag(name: str, value: str) -> bool:
    """Strict: anything but true/false (any case) is an error."""
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise ValueError(f"{name} must be True or False, got {value!r}")


def monitor_input(values: dict, csv_blob: str) -> list[str] | None:
    """The monitor's input arguments, from the ingest task's values.

    None means there is nothing to monitor (the ingest skipped a file that
    is not a forecast). Anything else the ingest didn't finish -> the
    staged CSV. A forecast track other than the Official one raises.
    """
    status = values.get("status", "")
    if status == "skipped":
        return None
    if status == "ingested":
        if values.get("track_type") != OFFICIAL_TRACK:
            raise ValueError(
                f"The ingest wrote a {values.get('track_type')!r}, not an "
                f"{OFFICIAL_TRACK!r}: not monitored."
            )
        return [
            "--from-db",
            "--disturbance-id", values["disturbance_id"],
            "--issued-time", values["issued_time"],
            "--track-type", values["track_type"],
            "--db-stage", values["stage"],
            "--content-sha", values.get("content_sha256", ""),
        ]  # fmt: skip
    if not csv_blob.startswith(INBOX_PREFIX) or ".." in csv_blob:
        raise ValueError(
            f"csv_blob must be under {INBOX_PREFIX}: {csv_blob!r}"
        )
    print(
        f"[run_monitor] ERROR: ingest status {status!r}; monitoring the "
        "staged CSV instead (the job will still be marked failed).",
        flush=True,
    )
    return ["--csv-blob", csv_blob]


def task_values() -> dict:
    from databricks.sdk.runtime import dbutils

    keys = (
        "status", "stage", "disturbance_id", "track_type", "issued_time",
        "content_sha256",
    )  # fmt: skip
    values = {}
    for key in keys:
        try:
            values[key] = dbutils.jobs.taskValues.get(
                taskKey="ingest", key=key, default="", debugValue=""
            )
        except Exception as exc:  # noqa: BLE001 — ingest may have failed
            print(f"[run_monitor] no task value {key!r} ({exc})")
            values[key] = ""
    return values


def main() -> None:
    from databricks.sdk.runtime import dbutils

    event = _arg(1)
    if event not in EVENTS:
        raise ValueError(f"event must be one of {EVENTS}, got {event!r}")
    test_email = env_flag("test_email", _arg(2))
    csv_blob = _arg(3)
    dry_run = env_flag("dry_run", _arg(4))
    simulate = env_flag("simulate_trigger", _arg(5))
    max_age_hours = _arg(6)
    if event == TEST_EVENT:
        test_email = True

    input_args = monitor_input(task_values(), csv_blob)
    if input_args is None:
        print("[run_monitor] ingest skipped a non-forecast file; done.")
        return

    for key in LISTMONK_SECRETS:
        try:
            os.environ[key] = dbutils.secrets.get("dsci", key)
        except Exception as exc:  # noqa: BLE001
            if not dry_run:
                raise RuntimeError(f"dsci/{key} unavailable: {exc}") from None
            print(f"[run_monitor] WARNING: dsci/{key} unavailable ({exc})")

    repo_root = os.path.abspath(os.path.join(_find_script_dir(), ".."))
    local_root = os.path.join(
        "/local_disk0"
        if os.path.isdir("/local_disk0")
        else tempfile.gettempdir(),
        "fji_monitor",
    )
    shutil.rmtree(local_root, ignore_errors=True)
    for sub in ("src", "pipelines"):
        shutil.copytree(
            os.path.join(repo_root, sub),
            os.path.join(local_root, sub),
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    env = dict(os.environ)
    env.update(
        {
            "TEST_EMAIL": str(test_email),
            "DRY_RUN": str(dry_run),
            "SIMULATE_TRIGGER": str(simulate),
            "MAX_FORECAST_AGE_HOURS": max_age_hours,
            "DISABLE_TQDM": "True",
            "PYTHONPATH": local_root + os.pathsep + env.get("PYTHONPATH", ""),
            "PYTHONUNBUFFERED": "1",
            "MPLCONFIGDIR": "/tmp/mplconfig",
        }
    )
    cmd = [
        sys.executable,
        os.path.join(local_root, "pipelines", "monitor_forecast_trigger.py"),
        *input_args,
    ]
    print(
        f"[run_monitor] event={event} TEST_EMAIL={test_email} "
        f"DRY_RUN={dry_run} SIMULATE_TRIGGER={simulate} "
        f"MAX_FORECAST_AGE_HOURS={max_age_hours} args={' '.join(input_args)}",
        flush=True,
    )
    rc = subprocess.run(cmd, cwd=local_root, env=env, check=False).returncode
    if rc != 0:
        raise RuntimeError(
            f"monitor_forecast_trigger.py exited with code {rc}"
        )
    print("[run_monitor] OK")


if __name__ == "__main__":
    main()
