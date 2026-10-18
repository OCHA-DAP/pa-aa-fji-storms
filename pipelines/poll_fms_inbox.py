"""Poll the FMS relay inbox and dispatch each new CSV to Check Trigger.

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
]
CHECK_TRIGGER_WORKFLOW = "run_monitor_forecast_trigger.yml"
# Check Trigger's run-name for a dispatch is "FMS <short hash> <event type>"
RUN_TITLE_PREFIX = "FMS"

DEFAULT_AUTHSERV_ID = "mx.zohomail.com"
RELAY_SUBJECT_PREFIX = "[fji-relay] "
EVENT_FORECAST = "fms_forecast"
EVENT_TEST = "fms_forecast_test"
EVENTS = (EVENT_FORECAST, EVENT_TEST)

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
) -> tuple[int, int]:
    """Dispatch every unhandled CSV. Returns (n_dispatched, n_failed).

    State is saved after each successful dispatch. If saving fails we stop
    straight away: carrying on would risk the next tick re-dispatching (and
    re-emailing) everything sent since the last good save.
    """
    done = set(zip(state["content_sha256"], state["event_type"]))
    run_titles = run_titles or set()
    n_dispatched = n_failed = 0

    def record(sha: str, event_type: str, key: str, filename: str) -> None:
        nonlocal state
        row = {
            "content_sha256": sha,
            "event_type": event_type,
            "message_id": key,
            "filename": filename,
            "dispatched_at_utc": now_fn().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        state = pd.concat([state, pd.DataFrame([row])], ignore_index=True)
        done.add((sha, event_type))
        save_fn(state)

    for msg in messages:
        event_type = classify_fn(msg)
        if event_type is None:
            continue
        key = message_key(msg)
        attachments = csv_attachments(msg)
        if not attachments:
            logger.info("Message %s has no CSV attachment", short_hash(key))
        for filename, data in attachments:
            sha = hashlib.sha256(data).hexdigest()
            if (sha, event_type) in done:
                continue
            label = f"{filename} ({event_type}, csv {sha[:12]})"
            if dry_run:
                action = "seed" if seed else "dispatch"
                logger.info("DRY RUN: would %s %s", action, label)
                continue
            if seed:
                logger.info("Seeding %s as handled", label)
                record(sha, event_type, key, filename)
                continue
            if run_title(sha, event_type) in run_titles:
                logger.info("Already dispatched (run found): %s", label)
                record(sha, event_type, key, filename)
                continue
            payload = {
                "csv": base64.b64encode(data).decode("ascii"),
                "filename": filename,
                "content_sha": sha[:12],
            }
            status = dispatch_fn(event_type, payload)
            if status != 204:
                logger.error("Dispatch of %s failed: HTTP %s", label, status)
                n_failed += 1
                continue
            logger.info("Dispatched %s", label)
            n_dispatched += 1
            record(sha, event_type, key, filename)
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
    senders = relay_senders()
    authserv_id = os.getenv("FMS_AUTHSERV_ID") or DEFAULT_AUTHSERV_ID
    relay_token = os.getenv("FMS_RELAY_TOKEN") or ""
    today = dt.datetime.now(dt.UTC).date()
    since = imap_since(today, lookback_days)
    logger.info(
        f"Polling FMS relay inbox: {dry_run=}, {seed=}, {require_dmarc=}, "
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
    logger.info("Fetched %d message(s) from allowed senders", len(messages))

    repo = os.environ["GITHUB_REPOSITORY"]
    gh_token = os.environ["GITHUB_TOKEN"]
    run_titles = set()
    if messages and not dry_run and not seed:
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
    )
    logger.info("Dispatched %d, failed %d", n_dispatched, n_failed)
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
