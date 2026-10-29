"""Poll the FMS relay inbox and hand each new CSV to the monitoring.

RSMC Nadi (FMS) emails its official forecast track CSV to
ocha-datascience@un.org, which forwards it to a personal un.org mailbox. A
Power Automate flow there (standard Outlook connector only) sends a NEW email,
with the CSV attached and the subject "[fji-relay] fms_forecast" (or
"[fji-relay] fms_forecast_test" for test emails), to a relay inbox on
humdata.org (Zoho). This script reads that inbox over IMAP (read-only: no
message is ever flagged or moved) and, for every CSV not yet handled, sends a
`repository_dispatch` to this repo. The `Check Trigger` workflow
(`run_monitor_forecast_trigger.yml`) picks it up and runs the monitoring;
`fms_forecast_test` forces TEST_EMAIL=True there.

A message is only accepted when all of these hold:
- it has exactly one From header, with exactly one address and no parse
  defects, and that address is in FMS_RELAY_SENDERS (a repo secret: it names
  a person);
- the subject is exactly "[fji-relay] fms_forecast" or
  "[fji-relay] fms_forecast_test" (the suffix is the event type);
- the TOPMOST Authentication-Results header (the one the receiving MX
  prepends) comes from FMS_AUTHSERV_ID (default mx.zohomail.com) and has a
  `dmarc=pass` result whose header.from is the sender's own domain; lower
  headers are ignored because a sender can forge them (REQUIRE_DMARC_PASS,
  default True, and only False in dry runs);
- if FMS_RELAY_TOKEN is set, the body contains it.

A CSV is identified by (sha256 of its bytes, event type), so the same file
relayed twice is only sent once. Handled CSVs are recorded in a CSV on the dev
blob, written after each successful dispatch. Each dispatch also names its
Check Trigger run after the CSV's short hash; before sending, recent runs are
checked so that a dispatch GitHub accepted but whose response was lost is not
sent again. SEED_STATE=True records every CSV currently in the window as
handled without dispatching (run it once before going live). The repo is
public, so the log only ever shows filenames, short hashes, event types and
counts.

Where each CSV goes is set by FMS_TARGETS (a repo variable):
- ``gha`` (default): repository_dispatch to Check Trigger, as above;
- ``gha,dbx-shadow``: also run the Databricks job (Fiji Cyclone
  Monitoring) on it, with every email forced to the test list;
- ``dbx``: the Databricks job is the live monitoring; no dispatch.
Exactly one live target (``gha`` or ``dbx``) is always set, so switching
over or back is one variable change. For the Databricks job the CSV is
staged in the dev blob and the run started through the Jobs API as the
fji-fms-relay service principal (DATABRICKS_CLIENT_ID / _SECRET), with an
idempotency token so a lost response can't start a second run. State is
kept per target; a CSV either live target handled counts as handled for
both, so switching never re-sends recent forecasts. A failed shadow run
only logs a warning.

Vanuatu (ds-aa-vut-cyclones) rides on the same relay. VUT_MONITORING (a repo
variable: off, the default | test | live) adds the Databricks job "Vanuatu
Cyclone Monitoring" (VUT_DBX_JOB_ID) as one more target, ``vut``:
- every FMS CSV also goes to it (the Vanuatu readiness trigger is on the
  RSMC Nadi forecast), after the Fiji targets;
- relay emails with the subject "[vut-relay] vmhs_forecast" (or
  "[vut-relay] vmhs_forecast_test"), which a second Power Automate flow sends
  for VMHS (Vanuatu Met) forecast CSVs, go to it and nowhere else.
``test`` sends every Vanuatu run to the test list; ``live`` sends real
forecasts to the real lists. The ``vut`` target keeps its own state rows: a
CSV handled for Fiji is still sent to Vanuatu once, and the reverse. A
Vanuatu problem never stops or delays the Fiji path: a misconfiguration is
logged and fails the run only after the Fiji targets were handled, and a
failed hand-off counts as a failure only in ``live``.
"""

import base64
import datetime as dt
import email
import hashlib
import imaplib
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from email import policy
from email.message import EmailMessage
from typing import Callable

import ocha_stratus as stratus
import pandas as pd
from azure.core.exceptions import ResourceNotFoundError

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("poll_fms_inbox")

STATE_BLOB = "pa-aa-fji-storms/monitoring/fms_inbox_processed.csv"
STATE_COLS = [
    "content_sha256",
    "event_type",
    "message_id",
    "filename",
    "dispatched_at_utc",
    "target",
]
CHECK_TRIGGER_WORKFLOW = "run_monitor_forecast_trigger.yml"
# Check Trigger's run-name for a dispatch is "FMS <short hash> <event type>"
RUN_TITLE_PREFIX = "FMS"

DEFAULT_AUTHSERV_ID = "mx.zohomail.com"
RELAY_SUBJECT_PREFIX = "[fji-relay] "
EVENT_FORECAST = "fms_forecast"
EVENT_TEST = "fms_forecast_test"
EVENTS = (EVENT_FORECAST, EVENT_TEST)
VUT_RELAY_SUBJECT_PREFIX = "[vut-relay] "
EVENT_VMHS = "vmhs_forecast"
EVENT_VMHS_TEST = "vmhs_forecast_test"
VUT_EVENTS = (EVENT_VMHS, EVENT_VMHS_TEST)
RELAY_EVENTS = {
    RELAY_SUBJECT_PREFIX: EVENTS,
    VUT_RELAY_SUBJECT_PREFIX: VUT_EVENTS,
}

TARGET_GHA = "gha"
TARGET_DBX = "dbx"
TARGET_DBX_SHADOW = "dbx-shadow"
LIVE_TARGETS = (TARGET_GHA, TARGET_DBX)
DBX_TARGETS = (TARGET_DBX, TARGET_DBX_SHADOW)
TARGET_VUT = "vut"
VUT_OFF, VUT_TEST, VUT_LIVE = "off", "test", "live"
VUT_MODES = (VUT_OFF, VUT_TEST, VUT_LIVE)
# The OAuth secret of fji-fms-relay is generated with only this scope: the
# token can call the Jobs API and nothing else.
DBX_OAUTH_SCOPE = "jobs"
# Must match databricks/run_ingest.py, which refuses anything else
INBOX_PREFIX = "pa-aa-fji-storms/raw/fms/inbox/"
# Must match ds-aa-vut-cyclones databricks/run_monitor.py
VUT_INBOX_PREFIX = "ds-aa-vut-cyclones/raw/vmhs/inbox/"

_QUOTED_WITH_SEMICOLON = re.compile(r'"(?:[^"\\]|\\.)*;(?:[^"\\]|\\.)*"')
_DMARC_PASS = re.compile(r"dmarc=pass\b", re.IGNORECASE)

# imaplib wants English month abbreviations regardless of locale
_MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def load_boolean_env(name: str, default: bool) -> bool:
    """Strict boolean env var: only 'True' or 'False' (or unset)."""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{name} must be 'True' or 'False', got {value!r}")


def parse_targets(raw: str | None) -> tuple[str, ...]:
    """FMS_TARGETS -> targets, live one first. Default: ("gha",)."""
    names = [t.strip() for t in (raw or TARGET_GHA).split(",") if t.strip()]
    allowed = {TARGET_GHA, TARGET_DBX, TARGET_DBX_SHADOW}
    unknown = set(names) - allowed
    if unknown:
        raise ValueError(f"Unknown FMS_TARGETS {sorted(unknown)}")
    live = [t for t in names if t in LIVE_TARGETS]
    if len(live) != 1:
        raise ValueError(
            f"FMS_TARGETS needs exactly one of {LIVE_TARGETS}, got {names}"
        )
    if TARGET_DBX in names and TARGET_DBX_SHADOW in names:
        raise ValueError("dbx and dbx-shadow together make no sense")
    shadow = [t for t in names if t == TARGET_DBX_SHADOW]
    return tuple(live + shadow[:1])


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def run_title(content_sha256: str, event_type: str) -> str:
    return f"{RUN_TITLE_PREFIX} {content_sha256[:12]} {event_type}"


def message_key(msg: EmailMessage) -> str:
    """Message-ID, or a stable hash of Date/From/Subject if it's missing."""
    message_id = (msg.get("Message-ID") or "").strip()
    if message_id:
        return message_id
    fallback = "|".join(
        str(msg.get(h) or "") for h in ("Date", "From", "Subject")
    )
    key = "sha256:" + hashlib.sha256(fallback.encode("utf-8")).hexdigest()
    logger.warning(
        "Message without Message-ID; using fallback key %s", short_hash(key)
    )
    return key


def relay_senders() -> set[str]:
    """Allowed relay senders from FMS_RELAY_SENDERS (comma-separated)."""
    raw = os.getenv("FMS_RELAY_SENDERS") or ""
    senders = {x.strip().lower() for x in raw.split(",") if x.strip()}
    if not senders:
        raise ValueError("FMS_RELAY_SENDERS is empty; nothing would pass")
    return senders


def sender_address(msg: EmailMessage) -> str | None:
    """The single From address, or None if From is missing or ambiguous.

    Rejects several From headers, several addresses, and any header the
    parser had to repair: e.g. the obsolete `From: a@un.org <b@evil.com>`
    reads as a@un.org here while the receiving server judges DMARC on
    evil.com.
    """
    headers = msg.get_all("From") or []
    if len(headers) != 1:
        return None
    header = headers[0]
    if getattr(header, "defects", None):
        return None
    addresses = getattr(header, "addresses", ())
    if len(addresses) != 1:
        return None
    return addresses[0].addr_spec.lower()


def _clean_token(value: str) -> str:
    """Attacker-controlled header text, made safe to log."""
    return re.sub(r"[^A-Za-z0-9.\-]", "", value)[:64]


def topmost_authserv_id(msg: EmailMessage) -> str | None:
    """authserv-id of the topmost Authentication-Results header, if any."""
    headers = msg.get_all("Authentication-Results") or []
    if not headers:
        return None
    first = str(headers[0]).split(";", 1)[0].strip()
    # RFC 8601 allows a version number after the authserv-id
    return first.split()[0].lower() if first else ""


def dmarc_passed_for(msg: EmailMessage, domain: str) -> bool:
    """True if the topmost Authentication-Results header has a dmarc=pass
    result for `domain` (its header.from).

    Only the topmost header counts: the receiving MX prepends its own, and a
    sender can add any number of forged ones further down. The caller checks
    that its authserv-id is the expected server. ARC-Authentication-Results
    is a different header and is never looked at.
    """
    headers = msg.get_all("Authentication-Results") or []
    if not headers:
        return False
    # A quoted string (e.g. a quoted MAIL FROM local part echoed into
    # smtp.mailfrom=) could smuggle in a fake "; dmarc=pass ..." clause, so
    # drop quoted strings that contain a ';' before splitting.
    value = _QUOTED_WITH_SEMICOLON.sub('""', str(headers[0]))
    for clause in value.split(";")[1:]:
        tokens = clause.split()
        # "dmarc=pass" may be followed directly by a "(comment)"
        if not tokens or not _DMARC_PASS.match(tokens[0]):
            continue
        for token in tokens[1:]:
            name, _, value = token.partition("=")
            if name.lower() == "header.from":
                if value.strip("<>\"'").lower() == domain.lower():
                    return True
    return False


def body_contains(msg: EmailMessage, token: str) -> bool:
    for part in msg.walk():
        if part.get_content_type() not in ("text/plain", "text/html"):
            continue
        if part.get_filename():
            continue
        try:
            content = part.get_content()
        except (KeyError, LookupError):
            continue
        if token in content:
            return True
    return False


def classify(
    msg: EmailMessage,
    senders: set[str],
    require_dmarc: bool = True,
    authserv_id: str = DEFAULT_AUTHSERV_ID,
    token: str = "",
) -> str | None:
    """Event type for a relay message, or None if it should be ignored."""
    address = sender_address(msg)
    if address is None or address not in senders:
        return None
    subject = str(msg.get("Subject") or "").strip().lower()
    for prefix, events in RELAY_EVENTS.items():
        if subject.startswith(prefix):
            event_type = subject[len(prefix) :].strip()
            break
    else:
        return None
    if event_type not in events:
        return None
    if require_dmarc:
        seen = topmost_authserv_id(msg)
        if seen != authserv_id.lower():
            logger.warning(
                "Relay message %s skipped: topmost Authentication-Results "
                "is from %r, expected %r (set FMS_AUTHSERV_ID if this is "
                "the relay inbox's own server)",
                short_hash(message_key(msg)),
                _clean_token(seen or ""),
                authserv_id.lower(),
            )
            return None
        if not dmarc_passed_for(msg, address.rsplit("@", 1)[1]):
            logger.warning(
                "Relay message %s skipped: no dmarc=pass for the sender",
                short_hash(message_key(msg)),
            )
            return None
    if token and not body_contains(msg, token):
        logger.warning(
            "Relay message %s skipped: relay token missing",
            short_hash(message_key(msg)),
        )
        return None
    return event_type


def csv_attachments(msg: EmailMessage) -> list[tuple[str, bytes]]:
    """(filename, raw bytes) for every attachment ending in .csv (any case)."""
    out = []
    for part in msg.walk():
        if part.is_multipart():
            continue
        filename = part.get_filename()
        if not filename or not filename.lower().endswith(".csv"):
            continue
        data = part.get_payload(decode=True)
        if data:
            out.append((filename, data))
    return out


def imap_since(today: dt.date, lookback_days: int) -> str:
    since = today - dt.timedelta(days=lookback_days)
    return f"{since.day:02d}-{_MONTHS[since.month - 1]}-{since.year}"


def search_criteria(since: str, senders: set[str]) -> list[str]:
    """IMAP SEARCH keys: from an allowed sender, received since `since`.

    The server filters on sender so nothing else in the inbox is
    downloaded. The subject is NOT searched on the server (how a server
    tokenises SUBJECT is not guaranteed); `classify` checks it exactly.
    """
    from_keys = [["FROM", f'"{s}"'] for s in sorted(senders)]
    sender_key = from_keys[0]
    for key in from_keys[1:]:
        sender_key = ["OR", *sender_key, *key]
    return ["SINCE", since, *sender_key]


def fetch_messages(
    host: str,
    user: str,
    password: str,
    folder: str,
    since: str,
    senders: set[str],
) -> list[EmailMessage]:
    """Fetch candidate relay messages without flagging them."""
    messages = []
    with imaplib.IMAP4_SSL(host, 993) as imap:
        try:
            imap.login(user, password)
        except imaplib.IMAP4.error:
            # never echo the server's reply: it can name the account
            raise RuntimeError("IMAP login failed") from None
        status, _ = imap.select(folder, readonly=True)
        if status != "OK":
            raise RuntimeError("Could not select the IMAP folder")
        status, data = imap.uid(
            "SEARCH", None, *search_criteria(since, senders)
        )
        if status != "OK":
            raise RuntimeError("IMAP search failed")
        for uid in data[0].split():
            status, parts = imap.uid("FETCH", uid, "(BODY.PEEK[])")
            if status != "OK":
                raise RuntimeError("IMAP fetch failed")
            raws = [p[1] for p in parts if isinstance(p, tuple)]
            if not raws:
                # expunged between SEARCH and FETCH
                continue
            messages.append(
                email.message_from_bytes(raws[0], policy=policy.default)
            )
    return messages


def _github_request(
    url: str, token: str, method: str = "GET", body: dict | None = None
) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )


def dispatch(repo: str, token: str, event_type: str, payload: dict) -> int:
    """Send a repository_dispatch; returns the HTTP status (204 = sent).

    A network error returns -1 (failed, retried next tick). If GitHub did
    accept it, the run-title check in the next tick stops a second send.
    """
    request = _github_request(
        f"https://api.github.com/repos/{repo}/dispatches",
        token,
        method="POST",
        body={"event_type": event_type, "client_payload": payload},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, OSError) as e:
        logger.error("Dispatch request failed: %s", type(e).__name__)
        return -1


def recent_run_titles(repo: str, token: str, since: dt.date) -> set[str]:
    """Titles of Check Trigger runs started by repository_dispatch."""
    query = urllib.parse.urlencode(
        {
            "event": "repository_dispatch",
            "created": f">={since.isoformat()}",
            "per_page": 100,
        }
    )
    request = _github_request(
        f"https://api.github.com/repos/{repo}/actions/workflows/"
        f"{CHECK_TRIGGER_WORKFLOW}/runs?{query}",
        token,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        runs = json.load(response).get("workflow_runs", [])
    return {r.get("display_title") or "" for r in runs}


def staged_blob_name(
    content_sha256: str, filename: str, prefix: str = INBOX_PREFIX
) -> str:
    """Dev-blob name for a staged CSV: no '/' and no '..' (the job's ingest
    task refuses names with either outside the inbox prefix)."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    safe = re.sub(r"\.{2,}", ".", safe)[-120:]
    return f"{prefix}{content_sha256[:12]}_{safe}"


def test_email_for(target: str, event_type: str) -> str:
    """The Databricks job's test_email: real lists only for a live
    fms_forecast; shadow runs and test events always go to the test list."""
    live = target == TARGET_DBX and event_type == EVENT_FORECAST
    return "False" if live else "True"


def vut_test_email_for(event_type: str, live: bool) -> str:
    """The Vanuatu job's test_email: real lists only for a real forecast
    (FMS or VMHS) once VUT_MONITORING is live."""
    real = event_type in (EVENT_FORECAST, EVENT_VMHS)
    return "False" if live and real else "True"


_TOKEN_CODES = {
    TARGET_DBX: "l",
    TARGET_DBX_SHADOW: "s",
    TARGET_VUT: "v",
    EVENT_FORECAST: "f",
    EVENT_TEST: "t",
    EVENT_VMHS: "m",
    EVENT_VMHS_TEST: "n",
}


def idempotency_token(
    content_sha256: str, target: str, event_type: str
) -> str:
    """Same CSV + target + event -> same token; Databricks allows 64 chars."""
    codes = _TOKEN_CODES[target] + _TOKEN_CODES[event_type]
    job = "vut" if target == TARGET_VUT else "fji"
    return f"{job}-{content_sha256[:48]}-{codes}"


class DbxError(RuntimeError):
    """A Databricks API call failed (message carries no response body)."""


class DbxClient:
    """The two Databricks REST calls the hand-off needs, OAuth M2M."""

    def __init__(self, host: str, client_id: str, secret: str, job_id: int):
        self.host = host.rstrip("/")
        if not self.host.startswith("https://"):
            self.host = "https://" + self.host
        self.client_id = client_id
        self.secret = secret
        self.job_id = job_id
        self._token = None
        self._token_expiry = 0.0

    def _access_token(self) -> str:
        if self._token and time.monotonic() < self._token_expiry:
            return self._token
        basic = base64.b64encode(
            f"{self.client_id}:{self.secret}".encode("utf-8")
        ).decode("ascii")
        request = urllib.request.Request(
            f"{self.host}/oidc/v1/token",
            data=urllib.parse.urlencode(
                {"grant_type": "client_credentials", "scope": DBX_OAUTH_SCOPE}
            ).encode("ascii"),
            method="POST",
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.load(response)
        except urllib.error.HTTPError as e:
            raise DbxError(f"token request: HTTP {e.code}") from None
        except (urllib.error.URLError, OSError) as e:
            raise DbxError(f"token request: {type(e).__name__}") from None
        self._token = body["access_token"]
        self._token_expiry = (
            time.monotonic() + int(body.get("expires_in", 300)) - 60
        )
        return self._token

    def run_now(self, job_parameters: dict, token: str) -> int:
        """Start the job; returns its run id."""
        request = urllib.request.Request(
            f"{self.host}/api/2.2/jobs/run-now",
            data=json.dumps(
                {
                    "job_id": self.job_id,
                    "idempotency_token": token,
                    "job_parameters": job_parameters,
                }
            ).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self._access_token()}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return int(json.load(response)["run_id"])
        except urllib.error.HTTPError as e:
            raise DbxError(f"run-now: HTTP {e.code}") from None
        except (urllib.error.URLError, OSError) as e:
            raise DbxError(f"run-now: {type(e).__name__}") from None


def stage_csv(data: bytes, blob_name: str) -> None:
    try:
        stratus.upload_blob_data(
            data, blob_name, stage="dev", content_type="text/csv"
        )
    except Exception as e:
        # the SDK's message can carry the request URL (and its SAS)
        raise DbxError(f"staging the CSV: {type(e).__name__}") from None


def make_handoff(
    client: DbxClient,
    stage_fn: Callable[[bytes, str], None] = stage_csv,
) -> Callable[[str, str, str, bytes, str], bool]:
    """Stage a CSV and start the Databricks job on it; True if started."""

    def handoff(
        target: str, event_type: str, filename: str, data: bytes, sha: str
    ) -> bool:
        blob = staged_blob_name(sha, filename)
        try:
            stage_fn(data, blob)
            run_id = client.run_now(
                {
                    "csv_blob": blob,
                    "event": event_type,
                    "test_email": test_email_for(target, event_type),
                },
                idempotency_token(sha, target, event_type),
            )
        except DbxError as e:
            logger.error("Databricks hand-off failed: %s", e)
            return False
        except Exception as e:  # noqa: BLE001 — e.g. an unexpected reply
            logger.error("Databricks hand-off failed: %s", type(e).__name__)
            return False
        logger.info("Databricks run %s started", run_id)
        return True

    return handoff


def make_vut_handoff(
    client: DbxClient,
    live: bool,
    stage_fn: Callable[[bytes, str], None] = stage_csv,
) -> Callable[[str, str, str, bytes, str], bool]:
    """Like make_handoff, for the Vanuatu job. FMS CSVs are staged where
    the Fiji job stages them; VMHS CSVs under the Vanuatu inbox."""

    def handoff(
        target: str, event_type: str, filename: str, data: bytes, sha: str
    ) -> bool:
        prefix = VUT_INBOX_PREFIX if event_type in VUT_EVENTS else INBOX_PREFIX
        blob = staged_blob_name(sha, filename, prefix)
        try:
            stage_fn(data, blob)
            run_id = client.run_now(
                {
                    "csv_blob": blob,
                    "event": event_type,
                    "test_email": vut_test_email_for(event_type, live),
                },
                idempotency_token(sha, target, event_type),
            )
        except DbxError as e:
            logger.error("Vanuatu hand-off failed: %s", e)
            return False
        except Exception as e:  # noqa: BLE001 — e.g. an unexpected reply
            logger.error("Vanuatu hand-off failed: %s", type(e).__name__)
            return False
        logger.info("Vanuatu Databricks run %s started", run_id)
        return True

    return handoff


def dbx_client_from_env(
    job_id_var: str = "FMS_DBX_JOB_ID",
) -> DbxClient | None:
    """A client from the DATABRICKS_* / job id env, or None."""
    names = (
        "DATABRICKS_HOST",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
        job_id_var,
    )
    values = {n: os.getenv(n) or "" for n in names}
    missing = [n for n, v in values.items() if not v]
    if missing:
        logger.warning("Databricks hand-off not configured: %s unset", missing)
        return None
    try:
        job_id = int(values[job_id_var])
    except ValueError:
        logger.warning(
            "Databricks hand-off not configured: bad %s", job_id_var
        )
        return None
    return DbxClient(
        values["DATABRICKS_HOST"],
        values["DATABRICKS_CLIENT_ID"],
        values["DATABRICKS_CLIENT_SECRET"],
        job_id,
    )


def vut_mode_from_env() -> tuple[str, bool]:
    """(VUT_MONITORING, misconfigured). A bad value means off + an error:
    it must never raise, or the Fiji path would stop with it."""
    mode = (os.getenv("VUT_MONITORING") or VUT_OFF).strip().lower()
    if mode not in VUT_MODES:
        logger.error(
            "VUT_MONITORING must be one of %s; Vanuatu is off this run",
            VUT_MODES,
        )
        return VUT_OFF, True
    return mode, False


def load_state() -> pd.DataFrame:
    try:
        df = stratus.load_csv_from_blob(STATE_BLOB, stage="dev", dtype=str)
    except ResourceNotFoundError:
        logger.info("No state file yet; starting empty.")
        return pd.DataFrame(columns=STATE_COLS)
    except Exception as e:
        # the SDK's message can carry the request URL (and its SAS)
        raise RuntimeError(
            f"Could not load state: {type(e).__name__}"
        ) from None
    return df.reindex(columns=STATE_COLS)


def save_state(df: pd.DataFrame) -> None:
    try:
        stratus.upload_csv_to_blob(df, STATE_BLOB, stage="dev")
    except Exception as e:
        raise RuntimeError(
            f"Could not save state: {type(e).__name__}"
        ) from None


def run(
    messages: list[EmailMessage],
    state: pd.DataFrame,
    dry_run: bool,
    dispatch_fn: Callable[[str, dict], int],
    save_fn: Callable[[pd.DataFrame], None],
    classify_fn: Callable[[EmailMessage], str | None],
    run_titles: set[str] | None = None,
    seed: bool = False,
    now_fn: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    targets: tuple[str, ...] = (TARGET_GHA,),
    handoff_fn: Callable[[str, str, str, bytes, str], bool] | None = None,
    vut: str = VUT_OFF,
) -> tuple[int, int]:
    """Send every unhandled CSV to each target. Returns (n_sent, n_failed).

    n_failed counts live targets only: a failed shadow run is a warning,
    and so is a failed Vanuatu run unless ``vut`` is live. FMS CSVs go to
    ``targets`` and then, unless ``vut`` is off, to the Vanuatu job; VMHS
    CSVs only ever go to the Vanuatu job. State is saved after each
    successful send. If saving fails we stop straight away: carrying on
    would risk the next tick re-dispatching (and re-emailing) everything
    sent since the last good save.
    """
    # Rows from before targets existed were all Check Trigger dispatches.
    row_targets = state["target"].fillna(TARGET_GHA).replace("", TARGET_GHA)
    done = {
        "live": set(),
        "shadow": set(),
        "vut": set(),
    }

    def role_of(target: str) -> str:
        if target == TARGET_VUT:
            return "vut"
        return "shadow" if target == TARGET_DBX_SHADOW else "live"

    for sha, event_type, target in zip(
        state["content_sha256"], state["event_type"], row_targets
    ):
        done[role_of(target)].add((sha, event_type))
    run_titles = run_titles or set()
    n_dispatched = n_failed = 0
    vut_targets = () if vut == VUT_OFF else (TARGET_VUT,)

    def record(
        sha: str, event_type: str, key: str, filename: str, target: str
    ) -> None:
        nonlocal state
        row = {
            "content_sha256": sha,
            "event_type": event_type,
            "message_id": key,
            "filename": filename,
            "dispatched_at_utc": now_fn().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "target": target,
        }
        state = pd.concat([state, pd.DataFrame([row])], ignore_index=True)
        done[role_of(target)].add((sha, event_type))
        save_fn(state)

    def send(
        target: str, event_type: str, key: str, filename: str, data: bytes
    ) -> None:
        nonlocal n_dispatched, n_failed
        sha = hashlib.sha256(data).hexdigest()
        if (sha, event_type) in done[role_of(target)]:
            return
        label = f"{filename} ({event_type}, csv {sha[:12]})"
        if target != TARGET_GHA:
            label = f"{label} to {target}"
        if dry_run:
            action = "seed" if seed else "dispatch"
            logger.info("DRY RUN: would %s %s", action, label)
            return
        if seed:
            logger.info("Seeding %s as handled", label)
            record(sha, event_type, key, filename, target)
            return
        if target == TARGET_GHA:
            if run_title(sha, event_type) in run_titles:
                logger.info("Already dispatched (run found): %s", label)
                record(sha, event_type, key, filename, target)
                return
            payload = {
                "csv": base64.b64encode(data).decode("ascii"),
                "filename": filename,
                "content_sha": sha[:12],
            }
            status = dispatch_fn(event_type, payload)
            if status != 204:
                logger.error("Dispatch of %s failed: HTTP %s", label, status)
                n_failed += 1
                return
        else:
            ok = handoff_fn is not None and handoff_fn(
                target, event_type, filename, data, sha
            )
            if not ok:
                if target == TARGET_DBX_SHADOW:
                    logger.warning("Shadow run of %s not started", label)
                elif target == TARGET_VUT and vut != VUT_LIVE:
                    logger.warning("Vanuatu test run of %s not started", label)
                else:
                    logger.error("Hand-off of %s failed", label)
                    n_failed += 1
                return
        logger.info("Dispatched %s", label)
        n_dispatched += 1
        record(sha, event_type, key, filename, target)

    accepted = []
    for msg in messages:
        event_type = classify_fn(msg)
        if event_type is None:
            continue
        key = message_key(msg)
        attachments = csv_attachments(msg)
        if not attachments:
            logger.info("Message %s has no CSV attachment", short_hash(key))
        if event_type in VUT_EVENTS and not vut_targets:
            logger.warning(
                "Message %s is a %s but VUT_MONITORING is off; not sent",
                short_hash(key),
                event_type,
            )
            continue
        accepted.append((event_type, key, attachments))

    # Every Fiji target for every CSV first, then the Vanuatu job: a slow
    # or failing Vanuatu hand-off can never hold up a Fiji one.
    for fiji_pass in (True, False):
        for event_type, key, attachments in accepted:
            if fiji_pass:
                pass_targets = () if event_type in VUT_EVENTS else targets
            else:
                pass_targets = vut_targets
            for filename, data in attachments:
                for target in pass_targets:
                    send(target, event_type, key, filename, data)
    return n_dispatched, n_failed


def main() -> int:
    dry_run = load_boolean_env("DRY_RUN", True)
    seed = load_boolean_env("SEED_STATE", False)
    require_dmarc = load_boolean_env("REQUIRE_DMARC_PASS", True)
    if not require_dmarc and not dry_run:
        raise ValueError(
            "REQUIRE_DMARC_PASS=False is only allowed in dry runs"
        )
    lookback_days = int(os.getenv("LOOKBACK_DAYS", "3"))
    targets = parse_targets(os.getenv("FMS_TARGETS"))
    handoff_fn = None
    if any(t in DBX_TARGETS for t in targets):
        client = dbx_client_from_env()
        if client is not None:
            handoff_fn = make_handoff(client)
        elif TARGET_DBX in targets:
            raise ValueError(
                "FMS_TARGETS has dbx but Databricks is unset or invalid"
            )
        else:
            # never let a shadow misconfiguration touch the live path
            targets = tuple(t for t in targets if t != TARGET_DBX_SHADOW)
    # Vanuatu: a problem here is logged and fails the run at the end, after
    # the Fiji targets were handled; it never raises.
    vut, vut_misconfigured = vut_mode_from_env()
    vut_handoff = None
    if vut != VUT_OFF:
        vut_client = dbx_client_from_env("VUT_DBX_JOB_ID")
        if vut_client is None:
            logger.error(
                "VUT_MONITORING=%s but the Vanuatu job is not configured; "
                "Vanuatu is off this run",
                vut,
            )
            vut, vut_misconfigured = VUT_OFF, True
        else:
            vut_handoff = make_vut_handoff(vut_client, live=vut == VUT_LIVE)
    if vut_handoff is not None:
        fji_handoff = handoff_fn

        def handoff_fn(target, event_type, filename, data, sha):
            fn = vut_handoff if target == TARGET_VUT else fji_handoff
            return fn is not None and fn(
                target, event_type, filename, data, sha
            )

    senders = relay_senders()
    authserv_id = os.getenv("FMS_AUTHSERV_ID") or DEFAULT_AUTHSERV_ID
    relay_token = os.getenv("FMS_RELAY_TOKEN") or ""
    today = dt.datetime.now(dt.UTC).date()
    since = imap_since(today, lookback_days)
    logger.info(
        f"Polling FMS relay inbox: {dry_run=}, {seed=}, {require_dmarc=}, "
        f"{lookback_days=}, n_senders={len(senders)}, {authserv_id=}, "
        f"relay_token_set={bool(relay_token)}, {targets=}, {vut=}"
    )

    messages = fetch_messages(
        host=os.environ["FMS_IMAP_HOST"],
        user=os.environ["FMS_IMAP_USER"],
        password=os.environ["FMS_IMAP_PASSWORD"],
        folder=os.getenv("FMS_IMAP_FOLDER") or "INBOX",
        since=since,
        senders=senders,
    )
    logger.info("Fetched %d message(s) from allowed senders", len(messages))

    repo = os.environ["GITHUB_REPOSITORY"]
    gh_token = os.environ["GITHUB_TOKEN"]
    run_titles = set()
    if messages and not dry_run and not seed and TARGET_GHA in targets:
        try:
            run_titles = recent_run_titles(
                repo, gh_token, today - dt.timedelta(days=lookback_days + 1)
            )
        except (urllib.error.URLError, OSError, ValueError) as e:
            # a missed alert is worse than a rare duplicate: carry on
            logger.warning(
                "Could not list recent Check Trigger runs (%s); "
                "relying on the state file only",
                type(e).__name__,
            )
    n_dispatched, n_failed = run(
        messages,
        load_state(),
        dry_run=dry_run,
        dispatch_fn=lambda event, payload: dispatch(
            repo, gh_token, event, payload
        ),
        save_fn=save_state,
        classify_fn=lambda msg: classify(
            msg, senders, require_dmarc, authserv_id, relay_token
        ),
        run_titles=run_titles,
        seed=seed,
        targets=targets,
        handoff_fn=handoff_fn,
        vut=vut,
    )
    logger.info("Dispatched %d, failed %d", n_dispatched, n_failed)
    return 1 if n_failed or vut_misconfigured else 0


if __name__ == "__main__":
    sys.exit(main())
