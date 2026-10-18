"""Task 1 of the Fiji Cyclone Monitoring job: ingest one FMS forecast CSV.

The FMS inbox poller stages the CSV in the dev blob and starts the job.
This task fetches ds-storms-pipeline at ``storms_ref`` and runs its FMS
ingest (``python -m src.pipelines.fms ingest``), which archives the raw CSV
to the ``storm`` container and writes storms.fms_forecasts / fms_tracks_geo.
The forecast's key goes to the monitor task as task values.

Positional parameters (databricks.yml):
    1 csv_blob    dev-blob name the poller staged (inbox prefix only)
    2 event       fms_forecast | fms_forecast_test
    3 mode        dev | prod: database + raw archive. Test events: always dev
    4 storms_ref  ds-storms-pipeline branch or tag

Never ``sys.exit`` here: the DBX exec context treats even ``SystemExit(0)``
as a failed task. Failures raise.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

STORMS_REPO = "https://github.com/OCHA-DAP/ds-storms-pipeline.git"
INBOX_PREFIX = "pa-aa-fji-storms/raw/fms/inbox/"
EVENTS = ("fms_forecast", "fms_forecast_test")
TEST_EVENT = "fms_forecast_test"
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


def _arg(i: int) -> str:
    return sys.argv[i].strip() if len(sys.argv) > i else ""


def check_params(csv_blob: str, event: str, mode: str, storms_ref: str) -> str:
    """Validate the run's parameters; return the database stage to use."""
    if event not in EVENTS:
        raise ValueError(f"event must be one of {EVENTS}, got {event!r}")
    if not csv_blob.startswith(INBOX_PREFIX) or ".." in csv_blob:
        raise ValueError(
            f"csv_blob must be under {INBOX_PREFIX}: {csv_blob!r}"
        )
    if mode not in ("dev", "prod"):
        raise ValueError(f"mode must be dev or prod, got {mode!r}")
    if not _REF_RE.match(storms_ref) or ".." in storms_ref:
        raise ValueError(f"Bad storms_ref {storms_ref!r}")
    # A test forecast never lands in the prod tables, whatever mode says.
    return "dev" if event == TEST_EVENT else mode


def set_task_values(values: dict) -> None:
    from databricks.sdk.runtime import dbutils

    for key, value in values.items():
        dbutils.jobs.taskValues.set(key=key, value=value)


def main() -> None:
    csv_blob, event, mode, storms_ref = (_arg(i) for i in range(1, 5))
    stage = check_params(csv_blob, event, mode, storms_ref)
    print(
        f"[run_ingest] event={event} mode={mode} -> stage={stage} "
        f"storms_ref={storms_ref} csv_blob={csv_blob}",
        flush=True,
    )

    work = os.path.join(
        "/local_disk0"
        if os.path.isdir("/local_disk0")
        else tempfile.gettempdir(),
        "fji_ingest",
    )
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    repo = os.path.join(work, "ds-storms-pipeline")
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--depth",
            "1",
            "--branch",
            storms_ref,
            STORMS_REPO,
            repo,
        ],
        check=True,
    )
    sha = subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"[run_ingest] ds-storms-pipeline {storms_ref} at {sha}", flush=True)

    out_json = os.path.join(work, "ingest.json")
    env = dict(os.environ)
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        sys.executable,
        "-m",
        "src.pipelines.fms",
        "ingest",
        "--csv-blob",
        csv_blob,
        "--csv-blob-stage",
        "dev",
        "--mode",
        stage,
        "--out-json",
        out_json,
    ]
    rc = subprocess.run(cmd, cwd=repo, env=env, check=False).returncode
    if rc != 0:
        raise RuntimeError(f"FMS ingest exited with code {rc}")

    with open(out_json) as f:
        result = json.load(f)
    values = {
        "status": result["status"],
        "stage": stage,
        "event": event,
        "storms_sha": sha,
        "disturbance_id": result.get("disturbance_id", ""),
        "track_type": result.get("track_type", ""),
        "issued_time": result.get("issued_time", ""),
        "content_sha256": result.get("content_sha256", ""),
    }
    set_task_values(values)
    print(f"[run_ingest] {json.dumps(values)}", flush=True)


if __name__ == "__main__":
    main()
