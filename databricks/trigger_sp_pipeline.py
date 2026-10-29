"""Task 3 of the Fiji Cyclone Monitoring job: start the South Pacific pipeline.

After a live FMS forecast has been ingested, start ds-storms-pipeline's
"South Pacific FMS Pipeline" job, which computes the forecast's wind swaths
and exposure for every Pacific island country and then starts the South
Pacific storm alert. Fire-and-forget: ``run_now`` returns once the run is
queued, and nothing here can fail the Fiji job, whose own alert has already
gone out by then.

Positional parameters (databricks.yml):
    1 job_id  the South Pacific pipeline's job id, or "" to do nothing
    2 event   fms_forecast | fms_forecast_test (test events write the dev
              database, which the prod South Pacific pipeline does not read)

Never ``sys.exit`` or raise here: the DBX exec context treats even
``SystemExit(0)`` as a failed task. Falling off the end is success.
"""

import sys


def _arg(i, default=""):
    return sys.argv[i].strip() if len(sys.argv) > i else default


def main() -> None:
    job_id, event = _arg(1), _arg(2)
    if not job_id:
        print("trigger_sp_pipeline: no job id set; nothing to start.", flush=True)
        return
    if event != "fms_forecast":
        print(f"trigger_sp_pipeline: event {event!r} is not live; not starting.", flush=True)
        return
    try:
        from databricks.sdk import WorkspaceClient

        run = WorkspaceClient().jobs.run_now(job_id=int(job_id))
        print(f"trigger_sp_pipeline: started job {job_id}, run {run.run_id}", flush=True)
    except Exception as exc:  # noqa: BLE001 — never fail the Fiji job
        print(f"trigger_sp_pipeline: could not start job {job_id}: {exc!r}", flush=True)


main()
