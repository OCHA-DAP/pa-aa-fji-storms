import base64
import datetime as dt
import hashlib
import importlib.util
import urllib.error
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
FAKE_SHA = hashlib.sha256(FAKE_CSV).hexdigest()
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


def old_state(rows):
    """State as written before targets existed (5 columns), as load_state
    returns it: reindexed, so target is NaN."""
    return pd.DataFrame(rows, columns=poll.STATE_COLS[:5]).reindex(
        columns=poll.STATE_COLS
    )


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


def run(
    messages,
    state=None,
    dry_run=False,
    rec=None,
    run_titles=None,
    seed=False,
    **classify_kw,
):
    rec = rec or Recorder()
    result = poll.run(
        messages,
        empty_state() if state is None else state,
        dry_run=dry_run,
        dispatch_fn=rec.dispatch,
        save_fn=rec.save,
        classify_fn=lambda msg: classify(msg, **classify_kw),
        run_titles=run_titles,
        seed=seed,
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


def test_already_processed_csv_is_skipped():
    state = old_state(
        [[FAKE_SHA, "fms_forecast", "<other@un.org>", "other.csv", "x"]]
    )
    (n_ok, n_fail), rec = run([make_msg()], state=state)
    assert (n_ok, n_fail) == (0, 0)
    assert rec.dispatched == [] and rec.saved == []


def test_second_csv_in_same_message_still_dispatched():
    state = old_state(
        [[FAKE_SHA, "fms_forecast", "<abc123@un.org>", "a.csv", "x"]]
    )
    msg = make_msg(
        attachments=(("a.csv", FAKE_CSV), ("b.csv", FAKE_CSV + b"1,2\n"))
    )
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


def test_search_criteria_filters_sender_on_server_not_subject():
    one = poll.search_criteria("15-Oct-2026", {"a@un.org"})
    assert one == ["SINCE", "15-Oct-2026", "FROM", '"a@un.org"']
    two = poll.search_criteria("15-Oct-2026", {"b@un.org", "a@un.org"})
    assert two[2:] == ["OR", "FROM", '"a@un.org"', "FROM", '"b@un.org"']


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
    for var in ("SEED_STATE", "LOOKBACK_DAYS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(poll, "load_state", empty_state)
    monkeypatch.setattr(poll, "save_state", lambda df: None)
    monkeypatch.setattr(poll, "recent_run_titles", lambda *a: set())


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


# --- review fixes: spoofing, duplicates, going live -------------------------


def _parse(raw: bytes):
    return poll.email.message_from_bytes(raw, policy=poll.policy.default)


def _raw(from_lines, ar=ZOHO_AR):
    head = "".join(f"From: {f}\r\n" for f in from_lines)
    return (
        f"Authentication-Results: {ar}\r\n{head}"
        "To: relay@humdata.org\r\nSubject: [fji-relay] fms_forecast\r\n"
        "Message-ID: <raw@un.org>\r\n\r\nbody\r\n"
    ).encode()


def test_obsolete_from_syntax_spoof_is_skipped():
    # policy.default reads this as relay.sender@un.org, Zoho judges DMARC on
    # the angle-addr domain
    ar = "mx.zohomail.com; dmarc=pass header.from=evil.com"
    msg = _parse(_raw(["relay.sender@un.org <attacker@evil.com>"], ar=ar))
    assert poll.sender_address(msg) is None
    assert classify(msg) is None


def test_multiple_from_headers_or_addresses_are_skipped():
    two_headers = _parse(_raw(["relay.sender@un.org", "x@evil.com"]))
    assert classify(two_headers) is None
    two_addrs = _parse(_raw(["relay.sender@un.org, x@evil.com"]))
    assert classify(two_addrs) is None
    good = _parse(_raw(["Relay Sender <relay.sender@un.org>"]))
    assert classify(good) == poll.EVENT_FORECAST


def test_dmarc_pass_for_another_domain_is_skipped():
    ar = "mx.zohomail.com; dmarc=pass header.from=evil.com"
    assert classify(make_msg(auth_results=(ar,))) is None


def test_dmarc_pass_substring_elsewhere_is_skipped():
    ar = (
        "mx.zohomail.com; spf=pass smtp.mailfrom=dmarc=pass@evil.com; "
        "dmarc=fail header.from=un.org"
    )
    assert classify(make_msg(auth_results=(ar,))) is None


def test_dmarc_pass_with_comment_and_quotes_is_accepted():
    ar = (
        "mx.zohomail.com; dmarc=pass (p=quarantine dis=none) "
        'header.from="un.org"'
    )
    assert classify(make_msg(auth_results=(ar,))) == poll.EVENT_FORECAST


def test_same_csv_relayed_twice_is_sent_once_per_event_type():
    messages = [
        make_msg(message_id="<first@un.org>"),
        make_msg(message_id="<resend@un.org>"),
        make_msg(
            subject="[fji-relay] fms_forecast_test",
            message_id="<test@un.org>",
        ),
    ]
    (n_ok, _), rec = run(messages)
    assert n_ok == 2
    assert [e for e, _ in rec.dispatched] == [
        poll.EVENT_FORECAST,
        poll.EVENT_TEST,
    ]
    assert rec.dispatched[0][1]["content_sha"] == FAKE_SHA[:12]


def test_csv_with_existing_check_trigger_run_is_recorded_not_resent():
    titles = {poll.run_title(FAKE_SHA, poll.EVENT_FORECAST)}
    (n_ok, n_fail), rec = run([make_msg()], run_titles=titles)
    assert (n_ok, n_fail) == (0, 0)
    assert rec.dispatched == []
    assert list(rec.saved[-1]["content_sha256"]) == [FAKE_SHA]


def test_network_error_counts_as_failed(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr(poll.urllib.request, "urlopen", boom)
    assert poll.dispatch("o/r", "t", poll.EVENT_FORECAST, {}) == -1
    monkeypatch.setattr(
        poll.urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(TimeoutError()),
    )
    assert poll.dispatch("o/r", "t", poll.EVENT_FORECAST, {}) == -1


def test_seed_records_without_dispatching():
    (n_ok, n_fail), rec = run([make_msg()], seed=True)
    assert (n_ok, n_fail) == (0, 0)
    assert rec.dispatched == []
    assert list(rec.saved[-1]["content_sha256"]) == [FAKE_SHA]
    (_, _), rec = run([make_msg()], seed=True, dry_run=True)
    assert rec.saved == []


def test_main_refuses_dmarc_off_outside_dry_run(monkeypatch):
    _main_env(monkeypatch)
    monkeypatch.setenv("REQUIRE_DMARC_PASS", "False")
    with pytest.raises(ValueError):
        poll.main()
    monkeypatch.setenv("DRY_RUN", "True")
    monkeypatch.setattr(poll, "fetch_messages", lambda **kw: [])
    assert poll.main() == 0


def test_check_trigger_run_name_matches_run_title():
    wf = (
        Path(__file__).resolve().parents[1]
        / ".github/workflows/run_monitor_forecast_trigger.yml"
    ).read_text()
    assert (
        "format('FMS {0} {1}', github.event.client_payload.content_sha" in wf
    )
    assert poll.run_title("ab" * 32, "x") == "FMS " + "ab" * 6 + " x"


def test_quoted_mailfrom_cannot_inject_a_dmarc_clause():
    ar = (
        'mx.zohomail.com; spf=pass smtp.mailfrom="; dmarc=pass '
        'header.from=un.org "@evil.com; dmarc=fail header.from=un.org'
    )
    assert classify(make_msg(auth_results=(ar,))) is None


def test_dmarc_pass_followed_directly_by_comment_is_accepted():
    ar = "mx.zohomail.com; dmarc=pass(p=quarantine) header.from=un.org"
    assert classify(make_msg(auth_results=(ar,))) == poll.EVENT_FORECAST
    ar = "mx.zohomail.com; dmarc=passed header.from=un.org"
    assert classify(make_msg(auth_results=(ar,))) is None


# --- Databricks hand-off (FMS_TARGETS) ------------------------------------


class Handoff:
    """Fake Databricks hand-off: records calls, answers from `results`."""

    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = []

    def __call__(self, target, event_type, filename, data, sha):
        self.calls.append((target, event_type, filename, sha))
        return self.results.pop(0) if self.results else True


def run_targets(messages, targets, state=None, handoff=None, rec=None, **kw):
    rec = rec or Recorder()
    handoff = handoff if handoff is not None else Handoff()
    result = poll.run(
        messages,
        empty_state() if state is None else state,
        dry_run=False,
        dispatch_fn=rec.dispatch,
        save_fn=rec.save,
        classify_fn=lambda msg: classify(msg),
        now_fn=lambda: NOW,
        targets=targets,
        handoff_fn=handoff,
        **kw,
    )
    return result, rec, handoff


def state_with(target):
    return pd.DataFrame(
        [[FAKE_SHA, "fms_forecast", "<x@un.org>", "x.csv", "t", target]],
        columns=poll.STATE_COLS,
    )


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, ("gha",)),
        ("", ("gha",)),
        ("gha", ("gha",)),
        ("gha,dbx-shadow", ("gha", "dbx-shadow")),
        ("dbx-shadow, gha", ("gha", "dbx-shadow")),
        ("dbx", ("dbx",)),
    ],
)
def test_parse_targets(raw, expected):
    assert poll.parse_targets(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["dbx-shadow", "gha,dbx", "dbx,dbx-shadow", "gha,lambda", "nothing"],
)
def test_parse_targets_needs_exactly_one_live(raw):
    with pytest.raises(ValueError):
        poll.parse_targets(raw)


def test_shadow_runs_alongside_live_dispatch():
    (n_ok, n_fail), rec, handoff = run_targets(
        [make_msg()], ("gha", "dbx-shadow")
    )
    assert (n_ok, n_fail) == (2, 0)
    assert [e for e, _ in rec.dispatched] == ["fms_forecast"]
    assert handoff.calls == [
        ("dbx-shadow", "fms_forecast", "forecast.CSV", FAKE_SHA)
    ]
    assert list(rec.saved[-1]["target"]) == ["gha", "dbx-shadow"]


def test_failed_shadow_is_not_a_failure_and_is_retried():
    (n_ok, n_fail), rec, _ = run_targets(
        [make_msg()], ("gha", "dbx-shadow"), handoff=Handoff([False])
    )
    assert (n_ok, n_fail) == (1, 0)
    assert list(rec.saved[-1]["target"]) == ["gha"]
    # next tick: gha is done, the shadow run is tried again
    (n_ok, n_fail), _, handoff = run_targets(
        [make_msg()], ("gha", "dbx-shadow"), state=rec.saved[-1]
    )
    assert (n_ok, n_fail) == (1, 0)
    assert [c[0] for c in handoff.calls] == ["dbx-shadow"]


def test_failed_live_dbx_counts_as_failure():
    (n_ok, n_fail), rec, _ = run_targets(
        [make_msg()], ("dbx",), handoff=Handoff([False])
    )
    assert (n_ok, n_fail) == (0, 1)
    assert rec.saved == []


def test_flip_to_dbx_does_not_resend_what_gha_sent():
    for state in (
        state_with("gha"),
        old_state([[FAKE_SHA, "fms_forecast", "<x@un.org>", "x.csv", "t"]]),
    ):
        (n_ok, n_fail), rec, handoff = run_targets(
            [make_msg()], ("dbx",), state=state
        )
        assert (n_ok, n_fail) == (0, 0)
        assert handoff.calls == [] and rec.dispatched == []


def test_rollback_to_gha_does_not_resend_what_dbx_sent():
    (n_ok, _), rec, _ = run_targets(
        [make_msg()], ("gha",), state=state_with("dbx")
    )
    assert n_ok == 0 and rec.dispatched == []


def test_shadow_run_does_not_count_as_live():
    (n_ok, _), _, handoff = run_targets(
        [make_msg()], ("dbx",), state=state_with("dbx-shadow")
    )
    assert n_ok == 1
    assert [c[0] for c in handoff.calls] == ["dbx"]


def test_seed_records_every_target():
    (n_ok, _), rec, handoff = run_targets(
        [make_msg()], ("gha", "dbx-shadow"), seed=True
    )
    assert handoff.calls == [] and rec.dispatched == []
    assert list(rec.saved[-1]["target"]) == ["gha", "dbx-shadow"]


@pytest.mark.parametrize(
    "target, event_type, expected",
    [
        ("dbx", "fms_forecast", "False"),
        ("dbx", "fms_forecast_test", "True"),
        ("dbx-shadow", "fms_forecast", "True"),
        ("dbx-shadow", "fms_forecast_test", "True"),
    ],
)
def test_test_email_for(target, event_type, expected):
    assert poll.test_email_for(target, event_type) == expected


def test_idempotency_token_is_stable_and_short():
    a = poll.idempotency_token(FAKE_SHA, "dbx-shadow", "fms_forecast_test")
    assert a == poll.idempotency_token(
        FAKE_SHA, "dbx-shadow", "fms_forecast_test"
    )
    assert len(a) <= 64
    assert a != poll.idempotency_token(FAKE_SHA, "dbx", "fms_forecast_test")
    assert a != poll.idempotency_token(FAKE_SHA, "dbx-shadow", "fms_forecast")


def test_staged_blob_name():
    name = poll.staged_blob_name(FAKE_SHA, "Rae [ray] 2025/../x...csv")
    assert name == (
        poll.INBOX_PREFIX + FAKE_SHA[:12] + "_Rae__ray__2025_._x.csv"
    )
    assert ".." not in name


class FakeClient:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def run_now(self, job_parameters, token):
        self.calls.append((job_parameters, token))
        if self.fail:
            raise poll.DbxError("run-now: HTTP 403")
        return 123


def test_make_handoff_stages_then_runs():
    staged = []
    client = FakeClient()
    handoff = poll.make_handoff(
        client, stage_fn=lambda data, blob: staged.append((data, blob))
    )
    assert handoff("dbx", "fms_forecast", "f.csv", FAKE_CSV, FAKE_SHA)
    blob = poll.staged_blob_name(FAKE_SHA, "f.csv")
    assert staged == [(FAKE_CSV, blob)]
    params, token = client.calls[0]
    assert params == {
        "csv_blob": blob,
        "event": "fms_forecast",
        "test_email": "False",
    }
    assert token == poll.idempotency_token(FAKE_SHA, "dbx", "fms_forecast")


def test_make_handoff_failure_returns_false():
    handoff = poll.make_handoff(
        FakeClient(fail=True), stage_fn=lambda data, blob: None
    )
    assert not handoff("dbx", "fms_forecast", "f.csv", FAKE_CSV, FAKE_SHA)

    def broken_stage(data, blob):
        raise poll.DbxError("staging the CSV: HttpResponseError")

    handoff = poll.make_handoff(FakeClient(), stage_fn=broken_stage)
    assert not handoff("dbx", "fms_forecast", "f.csv", FAKE_CSV, FAKE_SHA)


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return poll.json.dumps(self.body).encode()


def test_dbx_client_requests(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        if request.full_url.endswith("/oidc/v1/token"):
            return FakeResponse({"access_token": "tok", "expires_in": 3600})
        return FakeResponse({"run_id": 77})

    monkeypatch.setattr(poll.urllib.request, "urlopen", fake_urlopen)
    client = poll.DbxClient("adb-1.azuredatabricks.net", "app-id", "s3cret", 9)
    assert client.run_now({"csv_blob": "b"}, "tok-1") == 77
    assert client.run_now({"csv_blob": "c"}, "tok-2") == 77
    token_requests = [r for r in requests if r.full_url.endswith("/token")]
    assert len(token_requests) == 1  # cached
    run = requests[-1]
    assert (
        run.full_url
        == "https://adb-1.azuredatabricks.net/api/2.2/jobs/run-now"
    )
    assert run.headers["Authorization"] == "Bearer tok"
    assert poll.json.loads(run.data) == {
        "job_id": 9,
        "idempotency_token": "tok-2",
        "job_parameters": {"csv_blob": "c"},
    }


def test_dbx_client_errors_carry_no_body(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url, 401, "Unauthorized", {}, None
        )

    monkeypatch.setattr(poll.urllib.request, "urlopen", fake_urlopen)
    client = poll.DbxClient("https://h", "app-id", "s3cret", 9)
    with pytest.raises(poll.DbxError) as e:
        client.run_now({}, "t")
    assert str(e.value) == "token request: HTTP 401"
    assert "s3cret" not in str(e.value)


def test_main_live_dbx_needs_databricks_env(monkeypatch):
    _main_env(monkeypatch)
    monkeypatch.setenv("FMS_TARGETS", "dbx")
    for var in (
        "DATABRICKS_HOST",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
        "FMS_DBX_JOB_ID",
    ):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError):
        poll.main()


def test_main_unconfigured_shadow_leaves_live_path_alone(monkeypatch):
    _main_env(monkeypatch)
    monkeypatch.setenv("FMS_TARGETS", "gha,dbx-shadow")
    for var in (
        "DATABRICKS_HOST",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
        "FMS_DBX_JOB_ID",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(poll, "fetch_messages", lambda **kw: [make_msg()])
    sent = []
    monkeypatch.setattr(poll, "dispatch", lambda *a: sent.append(a) or 204)
    assert poll.main() == 0
    assert len(sent) == 1


def test_main_default_targets_unchanged(monkeypatch):
    _main_env(monkeypatch)
    monkeypatch.delenv("FMS_TARGETS", raising=False)
    monkeypatch.setattr(poll, "fetch_messages", lambda **kw: [make_msg()])

    def no_client():
        raise AssertionError("no Databricks client by default")

    monkeypatch.setattr(poll, "dbx_client_from_env", no_client)
    sent = []
    monkeypatch.setattr(poll, "dispatch", lambda *a: sent.append(a) or 204)
    assert poll.main() == 0
    assert len(sent) == 1


def test_make_handoff_unexpected_error_returns_false():
    class Broken:
        def run_now(self, job_parameters, token):
            raise KeyError("run_id")

    handoff = poll.make_handoff(Broken(), stage_fn=lambda data, blob: None)
    assert not handoff(
        "dbx-shadow", "fms_forecast", "f.csv", FAKE_CSV, FAKE_SHA
    )


def test_bad_job_id_drops_shadow_but_not_live(monkeypatch):
    _main_env(monkeypatch)
    monkeypatch.setenv("DATABRICKS_HOST", "https://h")
    monkeypatch.setenv("DATABRICKS_CLIENT_ID", "id")
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "s")
    monkeypatch.setenv("FMS_DBX_JOB_ID", "not-a-number")
    monkeypatch.setenv("FMS_TARGETS", "gha,dbx-shadow")
    monkeypatch.setattr(poll, "fetch_messages", lambda **kw: [make_msg()])
    sent = []
    monkeypatch.setattr(poll, "dispatch", lambda *a: sent.append(a) or 204)
    assert poll.main() == 0
    assert len(sent) == 1
    monkeypatch.setenv("FMS_TARGETS", "dbx")
    with pytest.raises(ValueError):
        poll.main()
