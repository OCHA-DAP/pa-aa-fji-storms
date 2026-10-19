# Fiji Anticipatory Action: storms

[![Generic badge](https://img.shields.io/badge/STATUS-ENDORSED-%231EBFB3)](https://shields.io/)

## Background information

Work began on the Fiji Tropical Cyclone AA Pilot in June 2023.

The tropical cyclone (TC) trigger has been developed with Fiji Meteorological
Services (FMS, operators of RSMC Nadi) in
consultation with the National Disaster Management Office (NDMO).

Over August 28-29 2023,
a [workshop](https://www.linkedin.com/feed/update/urn:li:activity:7103241608472514560/)
was held
by OCHA in Suva to build understanding of the framework and finalize the trigger
mechanism.

The proposed trigger has two stages, based on FMS's forecasts:

- Readiness: unofficial internal 120-hr forecast
- Action: official 72-hr forecast

The proposed trigger threshold is a TC that forecast to either:

- Be at Category 4 or greater while within 250 km of any point in Fiji, _or_
- Be at Category 3 or greater while making landfall in Fiji

The anticipatory action framework was formally endorsed by the Emergency Relief Coordinator on Nov 13, 2023.

That was the 2023 trigger. The 2025–2027 framework (endorsed 17 December
2025) replaced it with a population-exposure trigger: at least 5,000 people
in the area forecast to get winds of 64 knots or more, within 120 hours
(readiness) or 72 hours (action) of the forecast. Its third trigger, on
observed IMERG rainfall, is not automated.

## Monitoring

Every FMS forecast email goes through this chain (live since 18 October
2026). There is no seasonal switch: it runs whenever FMS sends a forecast.

```text
RSMC Nadi (FMS) forecaster
  |  email, forecast CSV attached
  v
ocha-datascience@un.org  --forwarded-->  the maintainer's un.org mailbox
  |  Power Automate flow fji_trigger_monitoring (see below)
  v
humdata.org (Zoho) relay inbox, subject "[fji-relay] fms_forecast"
  |  .github/workflows/poll_fms_inbox.yml, every 10 minutes (IMAP)
  v
FMS_TARGETS picks the monitor:
  gha  "Check Trigger" (.github/workflows/run_monitor_forecast_trigger.yml,
       repository_dispatch; runs the code on branch initial-2025-trigger)
  dbx  Databricks job "Fiji Cyclone Monitoring" (databricks.yml; runs main)
  |  pipelines/monitor_forecast_trigger.py: >= 5,000 people exposed to
  |  >= 64-knot winds within 120 h (readiness) or 72 h (action)?
  v
Listmonk: an information email for every forecast, plus readiness and
action emails when the trigger is met
```

Check Trigger checks out `initial-2025-trigger`, so a change to the GitHub
Actions monitor only takes effect once it is on that branch. Never delete
it: GitHub Pages is also served from its `docs/` folder.

### The Power Automate flow

`fji_trigger_monitoring` is a personal flow on the maintainer's un.org
mailbox (default Power Automate environment). It is not in this repo, and
it stops working if that account or mailbox goes away: whoever takes over
monitoring needs their own copy. The internal KB has the addresses and the
flow's details.

- **Starts on:** "When a new email arrives (V3)". It carries on only if the
  sender's address ends with `met.gov.fj` or the subject is exactly
  `fji_test`, and the email has a CSV attached.
- **For each CSV:** saves it to OneDrive `fji_fms_forecasts/` (the archive
  of every forecast received), replies to the sender to acknowledge it,
  then sends it as an attachment to the relay inbox with the subject
  `[fji-relay] fms_forecast` (`[fji-relay] fms_forecast_test` for a
  `fji_test` email), using the Outlook "Send an email (V2)" action.
- **Why the relay:** until October 2026 the flow posted the CSV straight to
  Check Trigger. That needs the HTTP connector, which is premium (the
  licence lapsed), and the GitHub connector is blocked by the UN data
  policy. The standard Outlook connector is allowed, so the flow emails the
  CSV on and the poller below picks it up.
- **Long-term fix:** read the team mailbox directly (Microsoft Graph
  `Mail.Read` on it, which needs OCHA IT). The flow and the relay inbox
  could then go.

### FMS relay inbox poller

The Power Automate flow can no longer call GitHub directly (the HTTP
connector is premium and the GitHub connector is blocked by the UN data
policy). Instead:

1. FMS emails the forecast CSV to ocha-datascience@un.org, which forwards it
   to a personal un.org mailbox.
2. The `fji_trigger_monitoring` flow there (standard Outlook connector only)
   sends a new email to a humdata.org (Zoho) relay inbox, with the CSV
   attached and the subject `[fji-relay] fms_forecast`
   (`[fji-relay] fms_forecast_test` for `fji_test` emails).
3. `.github/workflows/poll_fms_inbox.yml` runs every 10 minutes. It reads the
   relay inbox over IMAP (read-only, nothing is flagged or moved) and sends
   each new CSV to Check Trigger as a `repository_dispatch`. A CSV is
   identified by the sha256 of its bytes plus the event type, so the same
   file relayed twice goes out once. Handled CSVs are recorded in
   `pa-aa-fji-storms/monitoring/fms_inbox_processed.csv` on the dev blob,
   and each Check Trigger run is titled `FMS <csv short hash> <event>`; the
   poller checks recent run titles before sending, so a dispatch whose
   response was lost is not sent twice.

A relay email is accepted only if:

- it has one `From` header with one address, parsed without defects, and
  that address is in `FMS_RELAY_SENDERS`;
- the subject is exactly one of the two above;
- the topmost `Authentication-Results` header comes from `FMS_AUTHSERV_ID`
  and has `dmarc=pass` with `header.from` equal to the sender's domain
  (un.org publishes DMARC `p=quarantine`, so forged un.org senders fail).

If the first real relay email is skipped because the authserv-id differs,
the run log shows the id that was seen: set `FMS_AUTHSERV_ID` to it.
`REQUIRE_DMARC_PASS=False` is refused outside dry runs.

Repo variables: `FMS_IMAP_HOST` (default `imappro.zoho.com`),
`FMS_IMAP_FOLDER` (default `INBOX`), `FMS_AUTHSERV_ID` (default
`mx.zohomail.com`), `REQUIRE_DMARC_PASS` (default `True`) and
`POLL_DRY_RUN`. Repo secrets: `FMS_IMAP_USER`, `FMS_IMAP_PASSWORD` (a Zoho
app-specific password), `FMS_RELAY_SENDERS` (comma-separated addresses) and,
optionally, `FMS_RELAY_TOKEN` (if set, the email body must contain it). The
inbox address and the senders are secrets because the repo is public and
GitHub prints the step's environment in the log.

We recommend a Zoho filter that moves `[fji-relay]` mail into its own folder
(e.g. `fji-relay`), with `FMS_IMAP_FOLDER` set to that folder.

Going live:

1. Run the workflow by hand with `dry_run: True` and check the log lists
   the expected relay emails (and no authserv-id warnings).
2. Run it once with `dry_run: False`, `seed_state: True`. This records every
   CSV already in the 3-day window as handled, so switching on doesn't send
   old forecasts.
3. Set the repo variables `POLL_ENABLED` to `True` (scheduled runs are
   skipped until then) and `POLL_DRY_RUN` to `False` (until then scheduled
   runs only log what they would dispatch).

### Where the CSVs go: `FMS_TARGETS`

The repo variable `FMS_TARGETS` picks the monitoring each new CSV is sent to.
Exactly one live target is always set, so switching over, or back, is one
`gh variable set`:

| `FMS_TARGETS` | Live monitoring | Also |
|---|---|---|
| `gha` (default, or unset) | Check Trigger (GitHub Actions) | — |
| `gha,dbx-shadow` | Check Trigger | the Databricks job, all emails to the test list |
| `dbx` | the Databricks job (Fiji Cyclone Monitoring) | — |

For Databricks the poller stages the CSV in the dev blob
(`pa-aa-fji-storms/raw/fms/inbox/`) and starts the job through the Jobs API as
the `fji-fms-relay` service principal, passing `test_email=False` only for a
live `fms_forecast` on `dbx`. It needs the repo variables `DATABRICKS_HOST`,
`DATABRICKS_CLIENT_ID` (the principal's application id) and `FMS_DBX_JOB_ID`,
and the secret `DATABRICKS_CLIENT_SECRET` (an OAuth secret of the principal,
generated with the `jobs` scope only; the poller requests that scope).
The state file records the target of each send; a CSV that either live target
handled counts as handled for both, so switching never re-sends recent
forecasts. A failed shadow run only logs a warning. If `dbx-shadow` is set
but Databricks isn't configured, the shadow is dropped and Check Trigger
runs as usual.

A send is recorded once GitHub or Databricks accepts it, not when the run
finishes. Failed Databricks runs email the job's failure address; when
switching back from `dbx` to `gha` after one, re-send that forecast by hand
(Check Trigger's `workflow_dispatch` with the base64 CSV), because the
poller counts it as handled.

### Databricks job

"Fiji Cyclone Monitoring" is defined in `databricks.yml` (its header has the
deploy and run commands). It has no schedule: the poller starts one run per
new CSV. Two tasks run on one cluster:

1. `ingest` (`databricks/run_ingest.py`) runs ds-storms-pipeline's FMS
   ingest, which archives the raw CSV and writes the forecast to
   `storms.fms_forecasts` and `storms.fms_tracks_geo`. The deploy variable
   `ingest_mode` picks the dev or prod database.
2. `monitor` (`databricks/run_monitor.py`) runs
   `pipelines/monitor_forecast_trigger.py --from-db` on that forecast. If
   the ingest failed, it runs on the staged CSV instead, so the emails
   still go out (the run is still marked failed).

Each email is sent at most once per forecast CSV:
`pa-aa-fji-storms/monitoring/email_record.csv` on the dev blob records what
went out (`test_email_record.csv` for test sends). A forecast more than 48
hours old (`max_forecast_age_hours`) is not emailed to the real lists.

### Testing end to end

Email the maintainer's un.org mailbox with the subject exactly `fji_test`
and a real FMS forecast CSV attached. The flow relays it as
`fms_forecast_test`, and the poller picks it up within 10 minutes (or run
"Poll FMS inbox" by hand with `dry_run: False`). All emails then go to the
test list only (Listmonk list 5), with `[TEST]` in the subject. With
`FMS_TARGETS=gha,dbx-shadow` you get two, one from each monitor. The
poller sends each CSV once, so use a CSV it hasn't seen yet.

### When something breaks

Follow the chain: the flow's run history in Power Automate (and the CSV in
OneDrive `fji_fms_forecasts/`), the relay inbox, the "Poll FMS inbox" run
log (it says why it skipped an email), the Check Trigger run titled
`FMS <csv short hash> <event>` or the Databricks job run, then the Listmonk
campaign.

- **Poll runs fail at the IMAP login:** the Zoho app password in
  `FMS_IMAP_PASSWORD` was revoked or changed.
- **Databricks runs don't start (`DbxError` in the poll log):**
  `DATABRICKS_CLIENT_SECRET`, an OAuth secret of the `fji-fms-relay`
  service principal with the `jobs` scope only, has expired. The current
  one expires on 18 October 2028; generate a new one before then.
- **To send a forecast by hand:** run Check Trigger
  (`workflow_dispatch`) with the base64 of the CSV, or the Databricks job
  with `databricks bundle run` (see `databricks.yml`).

## Overview of analysis

Key analysis in `analysis/`:

- `01_returnperiod.md`: Calculates return period of TCs by strength and distance to Fiji
- `02_historicaltriggers.md`: Checks historical forecasts against trigger
- `03_forecastplots.md`: Creates interactive plots of historical forecasts for simulation exercises
- `04_distances.md`: Calculates distance of historical forecasts and actual tracks to administrative divisions

## Data description

Datasets:

- FMS historical best tracks (private)
  - analysis in this repo is based on file received from FMS, but the
    historical tracks are also publicly available
    on [IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive)
- FMS historical official 72-hr forecasts (private)
  - for Yasa, Harold, and Evan
- NDMO impact data (private)
  - housing destroyed / damaged
  - geolocated infrastructure damage
- Desinventar impact data (public)
- ECMWF track hindcasts (public)

## Directory structure

The code in this repository is organized as follows:

```shell

├── analysis      # Main repository of analytical work for the AA pilot
├── docs          # .Rmd files or other relevant documentation
├── exploration   # Experimental work not intended to be replicated
├── src           # Code to run any relevant data acquisition/processing pipelines
|
├── .gitignore
├── README.md
└── requirements.txt

```

## Reproducing this analysis

Create a directory where you would like the data to be stored,
and point to it using an environment variable called
`AA_DATA_DIR`.

Next create a new virtual environment and install the requirements with:

```shell
pip install -r requirements.txt
```

Finally, install any code in `src` using the command:

```shell
pip install -e .
```

If you would like to instead receive the processed data from our team, please
[contact us](mailto:centrehumdata@un.org).

## Development

All code is formatted according to black and flake8 guidelines.
The repo is set-up to use pre-commit.
Before you start developing in this repository, you will need to run

```shell
pre-commit install
```

The `markdownlint` hook will require
[Ruby](https://www.ruby-lang.org/en/documentation/installation/)
to be installed on your computer.

You can run all hooks against all your files using

```shell
pre-commit run --all-files
```

It is also **strongly** recommended to use `jupytext`
to convert all Jupyter notebooks (`.ipynb`) to Markdown files (`.md`)
before committing them into version control. This will make for
cleaner diffs (and thus easier code reviews) and will ensure that cell outputs
aren't
committed to the repo (which might be problematic if working with sensitive
data).
