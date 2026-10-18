"""Poll the FMS relay inbox and dispatch each new CSV to Check Trigger.

RSMC Nadi (FMS) emails its official forecast track CSV to
ocha-datascience@un.org, which forwards it to a personal un.org mailbox. A
Power Automate flow there (standard Outlook connector only) sends a NEW email,
with the CSV attached and the subject "[fji-relay] fms_forecast" (or
"[fji-relay] fms_forecast_test" for test emails), to a relay inbox on
humdata.org (Zoho). This script reads that inbox over IMAP (read-only: no
message is ever flagged or moved) and, for every CSV attachment not yet
handled, sends a `repository_dispatch` to this repo. The `Check Trigger`
workflow (`run_monitor_forecast_trigger.yml`) picks it up and runs the
monitoring; `fms_forecast_test` forces TEST_EMAIL=True there.

A message is only accepted when all of these hold:
- the sender is in FMS_RELAY_SENDERS (a repo secret: it names a person);
- the subject is exactly "[fji-relay] fms_forecast" or
  "[fji-relay] fms_forecast_test" (the suffix is the event type);
- the TOPMOST Authentication-Results header (the one the receiving MX
  prepends) comes from FMS_AUTHSERV_ID (default mx.zohomail.com) and says
  dmarc=pass; lower headers are ignored because a sender can forge them
  (REQUIRE_DMARC_PASS, default True);
- if FMS_RELAY_TOKEN is set, the body contains it.

Handled (message_id, filename) pairs are recorded in a CSV on the dev blob,
written after each successful dispatch. The repo is public, so the log only
ever shows filenames, hashed message ids, event types and counts.
"""

import base64
import datetime as dt
import email
import hashlib
import imaplib
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from email import policy
from email.message import EmailMessage
from email.utils import parseaddr
from typing import Callable

import ocha_stratus as stratus
import pandas as pd
from azure.core.exceptions import ResourceNotFoundError

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("poll_fms_inbox")

STATE_BLOB = "pa-aa-fji-storms/monitoring/fms_inbox_processed.csv"
STATE_COLS = ["message_id", "filename", "event_type", "dispatched_at_utc"]

DEFAULT_AUTHSERV_ID = "mx.zohomail.com"
RELAY_SUBJECT_PREFIX = "[fji-relay] "
EVENT_FORECAST = "fms_forecast"
EVENT_TEST = "fms_forecast_test"
EVENTS = (EVENT_FORECAST, EVENT_TEST)

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


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


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


def topmost_authserv_id(msg: EmailMessage) -> str | None:
    """authserv-id of the topmost Authentication-Results header, if any."""
    headers = msg.get_all("Authentication-Results") or []
    if not headers:
        return None
    first = str(headers[0]).split(";", 1)[0].strip()
    # RFC 8601 allows a version number after the authserv-id
    return first.split()[0].lower() if first else ""


def dmarc_passed(msg: EmailMessage) -> bool:
    """True if the topmost Authentication-Results header says dmarc=pass.

    Only the topmost header counts: the receiving MX prepends its own, and a
    sender can add any number of forged ones further down. The caller checks
    that its authserv-id is the expected server. ARC-Authentication-Results
    is a different header and is never looked at.
    """
    headers = msg.get_all("Authentication-Results") or []
    return bool(headers) and "dmarc=pass" in str(headers[0]).lower()


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
    _, address = parseaddr(str(msg.get("From") or ""))
    if address.lower() not in senders:
        return None
    subject = str(msg.get("Subject") or "").strip().lower()
    if not subject.startswith(RELAY_SUBJECT_PREFIX):
        return None
    event_type = subject[len(RELAY_SUBJECT_PREFIX) :].strip()
    if event_type not in EVENTS:
        return None
    if require_dmarc:
        seen = topmost_authserv_id(msg)
        if seen != authserv_id.lower():
            logger.warning(
                "Relay message %s skipped: topmost Authentication-Results "
                "is from %r, expected %r (set FMS_AUTHSERV_ID if this is "
                "the relay inbox's own server)",
                short_hash(message_key(msg)),
                seen,
                authserv_id.lower(),
            )
            return None
        if not dmarc_passed(msg):
            logger.warning(
                "Relay message %s skipped: no dmarc=pass",
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
    """IMAP SEARCH keys: relay subject, an allowed sender, received since.

    The server does the filtering so that nothing else in the inbox is ever
    downloaded; `classify` re-checks every message that comes back.
    """
    from_keys = [["FROM", f'"{s}"'] for s in sorted(senders)]
    sender_key = from_keys[0]
    for key in from_keys[1:]:
        sender_key = ["OR", *sender_key, *key]
    subject = RELAY_SUBJECT_PREFIX.strip()
    return ["SINCE", since, "SUBJECT", f'"{subject}"', *sender_key]


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
        imap.login(user, password)
        status, _ = imap.select(folder, readonly=True)
        if status != "OK":
            raise RuntimeError(f"Could not select IMAP folder {folder!r}")
        status, data = imap.search(None, *search_criteria(since, senders))
        if status != "OK":
            raise RuntimeError("IMAP search failed")
        for num in data[0].split():
            status, parts = imap.fetch(num, "(BODY.PEEK[])")
            if status != "OK":
                raise RuntimeError("IMAP fetch failed")
            raw = next(p[1] for p in parts if isinstance(p, tuple))
            messages.append(
                email.message_from_bytes(raw, policy=policy.default)
            )
    return messages


def dispatch(repo: str, token: str, event_type: str, payload: dict) -> int:
    """Send a repository_dispatch; returns the HTTP status (204 = sent)."""
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/dispatches",
        data=json.dumps(
            {"event_type": event_type, "client_payload": payload}
        ).encode("utf-8"),
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as e:
        return e.code


def load_state() -> pd.DataFrame:
    try:
        return stratus.load_csv_from_blob(STATE_BLOB, stage="dev", dtype=str)
    except ResourceNotFoundError:
        logger.info("No state file yet; starting empty.")
        return pd.DataFrame(columns=STATE_COLS)


def save_state(df: pd.DataFrame) -> None:
    stratus.upload_csv_to_blob(df, STATE_BLOB, stage="dev")


def run(
    messages: list[EmailMessage],
    state: pd.DataFrame,
    dry_run: bool,
    dispatch_fn: Callable[[str, dict], int],
    save_fn: Callable[[pd.DataFrame], None],
    classify_fn: Callable[[EmailMessage], str | None],
    now_fn: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
) -> tuple[int, int]:
    """Dispatch every unhandled CSV. Returns (n_dispatched, n_failed).

    State is saved after each successful dispatch. If saving fails we stop
    straight away: carrying on would risk the next tick re-dispatching (and
    re-emailing) everything sent since the last good save.
    """
    done = set(zip(state["message_id"], state["filename"]))
    n_dispatched = n_failed = 0
    for msg in messages:
        event_type = classify_fn(msg)
        if event_type is None:
            continue
        key = message_key(msg)
        attachments = csv_attachments(msg)
        if not attachments:
            logger.info("Message %s has no CSV attachment", short_hash(key))
        for filename, data in attachments:
            if (key, filename) in done:
                continue
            label = f"{filename} ({event_type}, msg {short_hash(key)})"
            if dry_run:
                logger.info("DRY RUN: would dispatch %s", label)
                continue
            payload = {
                "csv": base64.b64encode(data).decode("ascii"),
                "filename": filename,
                "message_id": key,
            }
            status = dispatch_fn(event_type, payload)
            if status != 204:
                logger.error("Dispatch of %s failed: HTTP %s", label, status)
                n_failed += 1
                continue
            logger.info("Dispatched %s", label)
            n_dispatched += 1
            row = {
                "message_id": key,
                "filename": filename,
                "event_type": event_type,
                "dispatched_at_utc": now_fn().strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            state = pd.concat([state, pd.DataFrame([row])], ignore_index=True)
            done.add((key, filename))
            save_fn(state)
    return n_dispatched, n_failed


def main() -> int:
    dry_run = load_boolean_env("DRY_RUN", True)
    require_dmarc = load_boolean_env("REQUIRE_DMARC_PASS", True)
    lookback_days = int(os.getenv("LOOKBACK_DAYS", "3"))
    senders = relay_senders()
    authserv_id = os.getenv("FMS_AUTHSERV_ID") or DEFAULT_AUTHSERV_ID
    relay_token = os.getenv("FMS_RELAY_TOKEN") or ""
    since = imap_since(dt.datetime.now(dt.UTC).date(), lookback_days)
    logger.info(
        f"Polling FMS relay inbox: {dry_run=}, {require_dmarc=}, "
        f"{lookback_days=}, n_senders={len(senders)}, {authserv_id=}, "
        f"relay_token_set={bool(relay_token)}"
    )

    messages = fetch_messages(
        host=os.environ["FMS_IMAP_HOST"],
        user=os.environ["FMS_IMAP_USER"],
        password=os.environ["FMS_IMAP_PASSWORD"],
        folder=os.getenv("FMS_IMAP_FOLDER") or "INBOX",
        since=since,
        senders=senders,
    )
    logger.info(
        "Fetched %d candidate message(s) since %s", len(messages), since
    )

    repo = os.environ["GITHUB_REPOSITORY"]
    gh_token = os.environ["GITHUB_TOKEN"]
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
    )
    logger.info("Dispatched %d, failed %d", n_dispatched, n_failed)
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
