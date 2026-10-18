"""Task 2 of the Fiji Cyclone Monitoring job: the trigger monitor.

Reads the forecast key the ingest task set as task values and runs
``pipelines/monitor_forecast_trigger.py --from-db`` on it: trigger check,
readiness/action/info emails through Listmonk. Only Official Forecast
Tracks are monitored; anything else the ingest wrote (or skipped) ends the
task quietly.

Positional parameters (databricks.yml):
    1 event             fms_forecast | fms_forecast_test (test forces
                        TEST_EMAIL=True)
    2 test_email        True | False
    3 dry_run           True | False
    4 simulate_trigger  True | False

src/ and pipelines/ are copied off the workspace mount to local disk first
(imports from the wsfs clone are intermittently flaky). Never ``sys.exit``
here: the DBX exec context treats even ``SystemExit(0)`` as a failure.
"""

import os
import shutil
import subprocess
import sys
import tempfile

OFFICIAL_TRACK = "Official Forecast Track"
TEST_EVENT = "fms_forecast_test"
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


def main() -> None:
    from databricks.sdk.runtime import dbutils

    event = _arg(1)
    test_email = env_flag("test_email", _arg(2))
    dry_run = env_flag("dry_run", _arg(3))
    simulate = env_flag("simulate_trigger", _arg(4))
    if event == TEST_EVENT:
        test_email = True

    def ingested(key: str) -> str:
        return dbutils.jobs.taskValues.get(
            taskKey="ingest", key=key, default="", debugValue=""
        )

    status = ingested("status")
    track_type = ingested("track_type")
    if status != "ingested":
        print(f"[run_monitor] ingest status {status!r}; nothing to monitor.")
        return
    if track_type != OFFICIAL_TRACK:
        print(f"[run_monitor] {track_type!r} is not monitored; done.")
        return

    for key in LISTMONK_SECRETS:
        try:
            os.environ[key] = dbutils.secrets.get("dsci", key)
        except Exception as exc:  # noqa: BLE001
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
            "DISABLE_TQDM": "True",
            "PYTHONPATH": local_root + os.pathsep + env.get("PYTHONPATH", ""),
            "PYTHONUNBUFFERED": "1",
            "MPLCONFIGDIR": "/tmp/mplconfig",
        }
    )
    cmd = [
        sys.executable,
        os.path.join(local_root, "pipelines", "monitor_forecast_trigger.py"),
        "--from-db",
        "--disturbance-id",
        ingested("disturbance_id"),
        "--issued-time",
        ingested("issued_time"),
        "--track-type",
        track_type,
        "--db-stage",
        ingested("stage"),
    ]
    print(
        f"[run_monitor] TEST_EMAIL={test_email} DRY_RUN={dry_run} "
        f"SIMULATE_TRIGGER={simulate} cmd={' '.join(cmd[1:])}",
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
