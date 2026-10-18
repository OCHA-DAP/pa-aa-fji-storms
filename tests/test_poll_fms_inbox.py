import base64
import datetime as dt
import importlib.util
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "poll_fms_inbox",
    Path(__file__).resolve().parents[1] / "pipelines" / "poll_fms_inbox.py",
)
poll = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(poll)

REAL_CSV = (
    Path.home()
    / "Library/CloudStorage/OneDrive-UnitedNations(2)/fji_fms_forecasts"
    / "20260407T000000Z_Official_Forecast_TrackVAIANU.csv"
)
FAKE_CSV = (
    b"# trackType=Official Forecast Track,baseTime=2026-04-07T00:00:00Z\n"
)
NOW = dt.datetime(2026, 10, 18, 12, 0, tzinfo=dt.UTC)
SENDERS = {"relay.sender@un.org"}
ZOHO_AR = (
    "mx.zohomail.com; dkim=pass; spf=pass "
    "smtp.mailfrom=relay.sender@un.org; dmarc=pass header.from=un.org"
)


def make_msg(
    sender="Relay Sender <Relay.Sender@un.org>",
    subject="[fji-relay] fms_forecast",
    attachments=(("forecast.CSV", FAKE_CSV),),
    message_id="<abc123@un.org>",
    auth_results=(ZOHO_AR,),
    arc_auth_results=(),
    body="FMS forecast relayed.",
):
    msg = EmailMessage()
    for header in auth_results:
        msg["Authentication-Results"] = header
    for header in arc_auth_results:
        msg["ARC-Authentication-Results"] = header
    msg["From"] = sender
    msg["To"] = "relay@humdata.org"
    msg["Subject"] = subject
    msg["Date"] = "Sun, 18 Oct 2026 06:00:00 +0000"
    if message_id:
        msg["Message-ID"] = message_id
    msg.set_content(body)
    for name, data in attachments:
        maintype, subtype = (
            ("text", "csv")
            if name.lower().endswith(".csv")
            else ("application", "pdf")
        )
        msg.add_attachment(
            data, maintype=maintype, subtype=subtype, filename=name
        )
    return msg


def classify(msg, **kw):
    kw.setdefault("senders", SENDERS)
    return poll.classify(msg, **kw)


def empty_state():
    return pd.DataFrame(columns=poll.STATE_COLS)


class Recorder:
    """Fake dispatch + save functions that record what they were given."""

    def __init__(self, statuses=None):
        self.statuses = list(statuses or [])
        self.dispatched = []
        self.saved = []

    def dispatch(self, event_type, payload):
        self.dispatched.append((event_type, payload))
        return self.statuses.pop(0) if self.statuses else 204

    def save(self, df):
        self.saved.append(df.copy())


def run(messages, state=None, dry_run=False, rec=None, **classify_kw):
    rec = rec or Recorder()
    result = poll.run(
        messages,
        empty_state() if state is None else state,
        dry_run=dry_run,
        dispatch_fn=rec.dispatch,
        save_fn=rec.save,
        classify_fn=lambda msg: classify(msg, **classify_kw),
        now_fn=lambda: NOW,
    )
    return result, rec


# --- gate -----------------------------------------------------------------


def test_allowed_sender_subject_and_dmarc_pass_is_accepted():
    assert classify(make_msg()) == poll.EVENT_FORECAST
    assert (
        classify(make_msg(subject=" [FJI-RELAY] fms_forecast_test "))
        == poll.EVENT_TEST
    )


def test_sender_not_in_allowlist_is_skipped():
    assert classify(make_msg(sender="forecaster@met.gov.fj")) is None
    assert classify(make_msg(sender="relay.sender@un.org.evil.com")) is None


def test_allowlist_comes_from_env(monkeypatch):
    monkeypatch.setenv("FMS_RELAY_SENDERS", " A@un.org , b@un.org ")
    assert poll.relay_senders() == {"a@un.org", "b@un.org"}
    monkeypatch.delenv("FMS_RELAY_SENDERS")
    with pytest.raises(ValueError):
        poll.relay_senders()


@pytest.mark.parametrize(
    "subject",
    [
        "fms_forecast",
        "[fji-relay] fms_forecast extra",
        "[fji-relay] something_else",
        "Fwd: [fji-relay] fms_forecast",
        "fji_test",
    ],
)
def test_wrong_subject_is_skipped(subject):
    assert classify(make_msg(subject=subject)) is None


def test_missing_dmarc_pass_is_skipped():
    assert classify(make_msg(auth_results=())) is None
    failed = ZOHO_AR.replace("dmarc=pass", "dmarc=fail")
    assert classify(make_msg(auth_results=(failed,))) is None


def test_arc_authentication_results_do_not_count():
    msg = make_msg(auth_results=(), arc_auth_results=("i=1; " + ZOHO_AR,))
    assert classify(msg) is None


def test_dmarc_check_can_be_disabled():
    assert (
        classify(make_msg(auth_results=()), require_dmarc=False)
        == poll.EVENT_FORECAST
    )


def test_forged_lower_header_under_topmost_fail_is_skipped():
    topmost_fail = ZOHO_AR.replace("dmarc=pass", "dmarc=fail")
    forged = "mx.zohomail.com; dmarc=pass header.from=un.org"
    msg = make_msg(auth_results=(topmost_fail, forged))
    assert classify(msg) is None


def test_topmost_header_from_wrong_server_is_skipped(caplog):
    forged = "attacker.example; dmarc=pass header.from=un.org"
    msg = make_msg(auth_results=(forged, ZOHO_AR))
    assert classify(msg) is None
    assert "attacker.example" in caplog.text  # logged so the id can be fixed


def test_authserv_id_match_is_case_insensitive_and_allows_version():
    assert (
        classify(make_msg(), authserv_id="MX.ZOHOMAIL.COM")
        == poll.EVENT_FORECAST
    )
    versioned = ZOHO_AR.replace("mx.zohomail.com;", "MX.ZohoMail.com 1;")
    assert classify(make_msg(auth_results=(versioned,))) == poll.EVENT_FORECAST
    other = ZOHO_AR.replace("mx.zohomail.com", "mx.example.org")
    msg = make_msg(auth_results=(other,))
    assert classify(msg) is None
    assert classify(msg, authserv_id="mx.example.org") == poll.EVENT_FORECAST


def test_relay_token_required_when_set():
    assert classify(make_msg(), token="s3cret-tok") is None
    assert (
        classify(make_msg(body="relay s3cret-tok here"), token="s3cret-tok")
        == poll.EVENT_FORECAST
    )


# --- dispatch + state ------------------------------------------------------


def test_dispatches_accepted_and_skips_others():
    messages = [
        make_msg(message_id="<fc@un.org>"),
        make_msg(
            subject="[fji-relay] fms_forecast_test",
            attachments=(("test.csv", FAKE_CSV),),
            message_id="<test@un.org>",
        ),
        make_msg(sender="spam@example.com", message_id="<spam@x>"),
        make_msg(auth_results=(), message_id="<nodmarc@un.org>"),
        make_msg(
            attachments=(("bulletin.pdf", b"%PDF"),),
            message_id="<pdf@un.org>",
        ),
    ]
    (n_ok, n_fail), rec = run(messages)
    assert (n_ok, n_fail) == (2, 0)
    assert [(e, p["filename"]) for e, p in rec.dispatched] == [
        (poll.EVENT_FORECAST, "forecast.CSV"),
        (poll.EVENT_TEST, "test.csv"),
    ]
    final = rec.saved[-1]
    assert list(final.columns) == poll.STATE_COLS
    assert set(final["message_id"]) == {"<fc@un.org>", "<test@un.org>"}
    assert (final["dispatched_at_utc"] == "2026-10-18T12:00:00Z").all()


def test_already_processed_pair_is_skipped():
    state = pd.DataFrame(
        [["<abc123@un.org>", "forecast.CSV", "fms_forecast", "x"]],
        columns=poll.STATE_COLS,
    )
    (n_ok, n_fail), rec = run([make_msg()], state=state)
    assert (n_ok, n_fail) == (0, 0)
    assert rec.dispatched == [] and rec.saved == []


def test_second_csv_in_same_message_still_dispatched():
    state = pd.DataFrame(
        [["<abc123@un.org>", "a.csv", "fms_forecast", "x"]],
        columns=poll.STATE_COLS,
    )
    msg = make_msg(attachments=(("a.csv", FAKE_CSV), ("b.csv", FAKE_CSV)))
    (n_ok, _), rec = run([msg], state=state)
    assert n_ok == 1
    assert rec.dispatched[0][1]["filename"] == "b.csv"


def test_state_saved_only_after_204_and_failure_counted():
    messages = [
        make_msg(message_id="<one@un.org>"),
        make_msg(message_id="<two@un.org>"),
    ]
    rec = Recorder(statuses=[500, 204])
    (n_ok, n_fail), rec = run(messages, rec=rec)
    assert (n_ok, n_fail) == (1, 1)
    assert len(rec.saved) == 1
    assert list(rec.saved[0]["message_id"]) == ["<two@un.org>"]


def test_save_failure_stops_further_dispatches():
    messages = [
        make_msg(message_id="<one@un.org>"),
        make_msg(message_id="<two@un.org>"),
    ]
    rec = Recorder()

    def broken_save(df):
        raise OSError("blob down")

    with pytest.raises(OSError):
        poll.run(
            messages,
            empty_state(),
            dry_run=False,
            dispatch_fn=rec.dispatch,
            save_fn=broken_save,
            classify_fn=lambda msg: classify(msg),
            now_fn=lambda: NOW,
        )
    assert len(rec.dispatched) == 1


def test_dry_run_dispatches_and_writes_nothing():
    (n_ok, n_fail), rec = run([make_msg()], dry_run=True)
    assert (n_ok, n_fail) == (0, 0)
    assert rec.dispatched == [] and rec.saved == []


def test_missing_message_id_uses_stable_fallback_key():
    key = poll.message_key(make_msg(message_id=None))
    assert key.startswith("sha256:")
    assert key == poll.message_key(make_msg(message_id=None))


def test_payload_round_trips_through_check_trigger_decoding():
    data = REAL_CSV.read_bytes() if REAL_CSV.exists() else FAKE_CSV * 200
    (n_ok, _), rec = run([make_msg(attachments=(("f.csv", data),))])
    assert n_ok == 1
    b64 = rec.dispatched[0][1]["csv"]
    assert "\n" not in b64
    # Check Trigger's decode_b64_string appends "==" before decoding
    decoded = base64.b64decode(b64.encode("ascii") + b"==").decode("ascii")
    assert decoded.encode("ascii") == data


def test_real_fms_csv_survives_rfc822_round_trip():
    if not REAL_CSV.exists():
        pytest.skip("real FMS CSV not available on this machine")
    msg = make_msg(attachments=((REAL_CSV.name, REAL_CSV.read_bytes()),))
    parsed = poll.email.message_from_bytes(
        msg.as_bytes(), policy=poll.policy.default
    )
    assert classify(parsed) == poll.EVENT_FORECAST
    [(name, data)] = poll.csv_attachments(parsed)
    assert name == REAL_CSV.name
    assert data == REAL_CSV.read_bytes()


# --- helpers + main --------------------------------------------------------


def test_imap_since_uses_english_months():
    assert poll.imap_since(dt.date(2026, 10, 18), 3) == "15-Oct-2026"
    assert poll.imap_since(dt.date(2026, 1, 2), 3) == "30-Dec-2025"


def test_load_boolean_env_is_strict(monkeypatch):
    monkeypatch.setenv("DRY_RUN", "true")
    with pytest.raises(ValueError):
        poll.load_boolean_env("DRY_RUN", True)
    monkeypatch.setenv("DRY_RUN", "False")
    assert poll.load_boolean_env("DRY_RUN", True) is False
    monkeypatch.delenv("DRY_RUN")
    assert poll.load_boolean_env("DRY_RUN", True) is True


def _main_env(monkeypatch):
    monkeypatch.setenv("DRY_RUN", "False")
    for var in ("FMS_IMAP_HOST", "FMS_IMAP_USER", "FMS_IMAP_PASSWORD"):
        monkeypatch.setenv(var, "x")
    monkeypatch.setenv("GITHUB_REPOSITORY", "OCHA-DAP/pa-aa-fji-storms")
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("FMS_RELAY_SENDERS", "relay.sender@un.org")
    for var in ("FMS_RELAY_TOKEN", "FMS_AUTHSERV_ID", "REQUIRE_DMARC_PASS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(poll, "load_state", empty_state)
    monkeypatch.setattr(poll, "save_state", lambda df: None)


def test_main_exits_nonzero_on_failed_dispatch(monkeypatch):
    _main_env(monkeypatch)
    monkeypatch.setattr(poll, "fetch_messages", lambda **kw: [make_msg()])
    monkeypatch.setattr(poll, "dispatch", lambda *a: 422)
    assert poll.main() == 1
    monkeypatch.setattr(poll, "dispatch", lambda *a: 204)
    assert poll.main() == 0


def test_main_exits_zero_with_nothing_to_do(monkeypatch):
    _main_env(monkeypatch)
    monkeypatch.setattr(poll, "fetch_messages", lambda **kw: [])

    def no_dispatch(*a):
        raise AssertionError("should not dispatch")

    monkeypatch.setattr(poll, "dispatch", no_dispatch)
    assert poll.main() == 0


def test_main_rejects_bad_require_dmarc_value(monkeypatch):
    _main_env(monkeypatch)
    monkeypatch.setenv("REQUIRE_DMARC_PASS", "yes")
    with pytest.raises(ValueError):
        poll.main()


def test_main_uses_relay_token_not_github_token(monkeypatch):
    _main_env(monkeypatch)
    monkeypatch.setenv("FMS_RELAY_TOKEN", "relay-tok")
    sent = []
    monkeypatch.setattr(
        poll,
        "fetch_messages",
        lambda **kw: [
            make_msg(body="has relay-tok", message_id="<ok@un.org>"),
            make_msg(body="has x only", message_id="<no@un.org>"),
        ],
    )
    monkeypatch.setattr(
        poll, "dispatch", lambda repo, tok, event, p: sent.append(tok) or 204
    )
    assert poll.main() == 0
    assert sent == ["x"]  # one dispatch, authenticated with GITHUB_TOKEN
