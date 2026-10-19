"""Controlled-fault local storage tests; no production reliability claims."""
import os
import sqlite3

import pytest

try:
    from agent_pilots.state_service import LocalStateService
except ImportError:
    LocalStateService = None


def config(**changes):
    return {"template": "reservation", "expected_fields": {"item": "book", "quantity": 2},
            "outcome": "full", "apply_delay": 0, "cache_lag": 0, "journal_lag": 0,
            "journal_mode": "lagged", "idempotent": False, **changes}


@pytest.fixture
def service(tmp_path):
    assert LocalStateService is not None, "the local HTTP/SQLite state service is not implemented"
    with LocalStateService(tmp_path / "state.sqlite") as instance:
        yield instance


@pytest.mark.parametrize("outcome,counts", [("none", (0, 0, 0)), ("full", (1, 1, 0)), ("partial", (1, 0, 1))])
def test_start_uses_real_sqlite_writes_at_tick_zero(service, outcome, counts):
    public = service.start_episode(config(outcome=outcome))
    audit = service.audit(public["episode_id"], drain=False)
    assert audit["tick"] == 0
    assert (audit["record_count"], audit["complete_count"], audit["partial_count"]) == counts
    assert audit["business_success"] is (outcome == "full")


def test_service_runs_in_a_different_process_and_persists_rows(tmp_path):
    assert LocalStateService is not None, "the local HTTP/SQLite state service is not implemented"
    path = tmp_path / "durable.sqlite"
    with LocalStateService(path) as instance:
        public = instance.start_episode(config())
        audit = instance.audit(public["episode_id"], drain=False)
        assert audit["events"]
        assert all(event["writer_pid"] != os.getpid() for event in audit["events"])
        assert path.exists()
        with sqlite3.connect(path) as con:
            assert con.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 1
    with LocalStateService(path) as reopened:
        assert reopened.audit(public["episode_id"], drain=False)["rows"] == audit["rows"]


def test_cache_and_lagged_journal_are_distinct_durable_visibility_tables(service):
    public = service.start_episode(config(cache_lag=2, journal_lag=4))
    eid = public["episode_id"]
    assert service.action(eid, "cache") == {"state": "absent"}  # t1
    cache = service.action(eid, "cache")  # t2
    assert set(cache) == {"state", "record_id"} and cache["state"] == "present"
    assert service.action(eid, "journal") == {"state": "unknown"}  # t3
    journal = service.action(eid, "journal")  # t4
    assert journal == {"state": "complete", "record_id": cache["record_id"]}
    with sqlite3.connect(service.db_path) as con:
        assert con.execute("SELECT record_id FROM cache_entries WHERE episode_id=?", (eid,)).fetchone()[0] == cache["record_id"]
        assert con.execute("SELECT state FROM operation_journal WHERE episode_id=?", (eid,)).fetchone()[0] == "complete"


def test_delayed_write_and_journal_lag_use_apply_tick_not_request_tick(service):
    eid = service.start_episode(config(apply_delay=2, journal_lag=2))["episode_id"]
    assert service.action(eid, "fields") == {"records": []}  # t1
    assert len(service.action(eid, "fields")["records"]) == 1  # t2
    assert service.action(eid, "journal") == {"state": "unknown"}  # t3
    assert service.action(eid, "journal")["state"] == "complete"  # t4


@pytest.mark.parametrize("outcome,first,final", [("none", "not_found", "not_found"), ("full", "pending", "complete"), ("partial", "pending", "partial")])
def test_atomic_journal_registers_before_delayed_apply(service, outcome, first, final):
    eid = service.start_episode(config(outcome=outcome, apply_delay=3, journal_mode="atomic"))["episode_id"]
    assert service.action(eid, "journal") == {"state": first}
    service.action(eid, "wait")
    result = service.action(eid, "journal")
    assert result["state"] == final
    if final in ("complete", "partial"):
        assert "record_id" in result


def test_unavailable_journal_never_exposes_hidden_outcome(service):
    eid = service.start_episode(config(journal_mode="unavailable"))["episode_id"]
    assert service.action(eid, "journal") == {"state": "unavailable"}


def test_nonidempotent_retry_collides_with_future_original_write_on_drain(service):
    eid = service.start_episode(config(apply_delay=5))["episode_id"]
    retry = service.action(eid, "retry")
    assert retry["state"] == "complete"
    assert service.audit(eid, drain=False)["record_count"] == 1
    final = service.audit(eid, drain=True)
    assert final["record_count"] == final["complete_count"] == 2
    assert final["duplicate"] is True and final["business_success"] is False
    assert len({row["record_id"] for row in final["rows"]}) == 2
    assert service.audit(eid)["rows"] == final["rows"]


@pytest.mark.parametrize("original_outcome", ["full", "partial"])
def test_idempotent_retry_reuses_one_row_even_when_original_arrives_later(service, original_outcome):
    eid = service.start_episode(config(outcome=original_outcome, apply_delay=5, idempotent=True))["episode_id"]
    retry = service.action(eid, "retry")
    again = service.action(eid, "retry")
    final = service.audit(eid)
    assert retry == again
    assert final["business_success"] is True and final["record_count"] == 1
    assert final["rows"] == [{"record_id": retry["record_id"], "fields": config()["expected_fields"]}]


def test_repair_requires_observed_id_and_completes_same_partial_row(service):
    eid = service.start_episode(config(outcome="partial", cache_lag=10, journal_lag=10))["episode_id"]
    unobserved = service.audit(eid, drain=False)["rows"][0]["record_id"]
    with pytest.raises(ValueError, match="observed"):
        service.action(eid, "repair", unobserved)
    visible = service.action(eid, "fields")["records"]
    assert visible[0]["fields"] == {"item": "book"}
    assert service.action(eid, "repair", visible[0]["record_id"]) == {"state": "complete", "record_id": unobserved}
    final = service.audit(eid)
    assert final["business_success"] is True and final["record_count"] == 1
    assert service.action(eid, "journal")["state"] == "complete"


def test_repair_rejects_missing_unknown_and_other_episode_ids(service):
    one = service.start_episode(config())
    two = service.start_episode(config())
    rid = service.action(one["episode_id"], "fields")["records"][0]["record_id"]
    for invalid in (None, "nonexistent", rid):
        with pytest.raises(ValueError):
            service.action(two["episode_id"], "repair", invalid)


def test_start_and_action_responses_use_only_contract_public_fields(service):
    hidden = config(outcome="partial", apply_delay=2, cache_lag=1, journal_lag=1)
    started = service.start_episode(hidden)
    assert set(started) == {"episode_id", "operation_id", "template", "expected_fields", "journal_mode", "idempotent", "status"}
    assert started["status"] == "timeout"
    assert started["template"] == hidden["template"]
    assert started["expected_fields"] == hidden["expected_fields"]
    assert "partial" not in started["episode_id"] and "partial" not in started["operation_id"]
    eid = started["episode_id"]
    assert service.action(eid, "wait") == {"state": "waited"}
    fields = service.action(eid, "fields")
    assert set(fields) == {"records"}
    assert set(fields["records"][0]) == {"record_id", "fields"}
    assert set(service.action(eid, "cache")) <= {"state", "record_id"}
    assert set(service.action(eid, "journal")) <= {"state", "record_id"}


def test_invalid_action_and_config_fail_without_fabricating_success(service):
    eid = service.start_episode(config())["episode_id"]
    with pytest.raises(ValueError):
        service.action(eid, "accept")
    with pytest.raises(ValueError):
        service.start_episode(config(apply_delay=-1))
    with pytest.raises(ValueError):
        service.start_episode(config(expected_fields={"one": 1}))


def test_pending_write_survives_process_restart_and_wall_time_does_not_advance_ticks(tmp_path):
    path = tmp_path / "pending.sqlite"
    with LocalStateService(path) as first:
        eid = first.start_episode(config(apply_delay=3, cache_lag=1))["episode_id"]
        assert first.audit(eid, drain=False)["tick"] == 0
        assert first.audit(eid, drain=False)["record_count"] == 0
    with LocalStateService(path) as restarted:
        assert restarted.audit(eid, drain=False)["tick"] == 0
        assert restarted.action(eid, "fields") == {"records": []}
        assert restarted.action(eid, "fields") == {"records": []}
        assert restarted.action(eid, "journal") == {"state": "complete", "record_id": restarted.audit(eid, drain=False)["rows"][0]["record_id"]}
        assert restarted.action(eid, "cache")["state"] == "present"


def test_observing_cache_or_journal_id_is_sufficient_for_same_row_repair(service):
    for observation in ("cache", "journal"):
        eid = service.start_episode(config(outcome="partial", journal_mode="atomic"))["episode_id"]
        record_id = service.action(eid, observation)["record_id"]
        assert service.action(eid, "repair", record_id) == {"state": "complete", "record_id": record_id}
        assert service.audit(eid)["business_success"] is True


def test_idempotent_retry_completes_existing_partial_without_creating_second_row(service):
    eid = service.start_episode(config(outcome="partial", idempotent=True))["episode_id"]
    original = service.action(eid, "fields")["records"][0]
    retry = service.action(eid, "retry")
    assert retry["record_id"] == original["record_id"]
    assert service.audit(eid)["business_success"] is True


def test_rejected_repair_rolls_back_tick_and_due_write_transaction(service):
    eid = service.start_episode(config(apply_delay=1))["episode_id"]
    with pytest.raises(ValueError, match="observed"):
        service.action(eid, "repair", "unknown")
    audit = service.audit(eid, drain=False)
    assert audit["tick"] == 0 and audit["record_count"] == 0
    assert len(service.action(eid, "fields")["records"]) == 1


def test_none_outcome_can_be_retried_and_closed_service_rejects_requests(tmp_path):
    instance = LocalStateService(tmp_path / "close.sqlite")
    eid = instance.start_episode(config(outcome="none", journal_mode="atomic"))["episode_id"]
    assert instance.action(eid, "retry")["state"] == "complete"
    assert instance.audit(eid)["business_success"] is True
    instance.close()
    instance.close()
    with pytest.raises(RuntimeError, match="closed"):
        instance.action(eid, "wait")
