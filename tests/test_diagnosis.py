"""Behavioral checks for costs, evidence and non-idempotent retry safety."""
import importlib
import pytest

try:
    diagnosis = importlib.import_module('agent_pilots.diagnosis')
except ModuleNotFoundError:
    diagnosis = None


def model(journal='lagged', cost='status-cheap', regime='nominal'):
    assert diagnosis is not None, 'state-diagnosis controller is not implemented'
    return diagnosis.DiagnosticModel(regime, journal, cost)


def test_empty_primary_read_does_not_make_inflight_retry_safe():
    m = model()
    after = m.update(m.initial(), 'fields', 'absent')
    assert after.tick == 1
    assert 0 < m.not_accepted_probability(after) < .99
    assert all(action != 'retry' for _, action in m.terminals(after))


def test_cache_presence_is_not_a_completion_certificate():
    m = model()
    after = m.update(m.initial(), 'cache', 'present')
    assert after.known_record and not after.certified
    assert all(action != 'accept' for _, action in m.terminals(after))
    assert any(action == 'repair' for _, action in m.terminals(after))


def test_partial_and_complete_status_require_different_terminal_actions():
    m = model('atomic')
    partial = m.update(m.initial(), 'journal', 'partial')
    complete = m.update(m.initial(), 'journal', 'complete')
    assert m.best_terminal(partial)[1] == 'repair'
    assert m.best_terminal(complete) == (0., 'accept')


def test_atomic_not_found_allows_safe_retry():
    m = model('atomic')
    after = m.update(m.initial(), 'journal', 'not_found')
    assert m.not_accepted_probability(after) == pytest.approx(1)
    assert m.best_terminal(after)[1] == 'retry'


def test_exact_planning_is_no_more_costly_than_deadline_fallback():
    m = model('unavailable')
    value, action = m.optimal(m.initial())
    assert value <= m.deadline_value(m.initial()) + 1e-10
    assert action in ('cache', 'fields', 'wait', 'retry', 'abstain')
    metrics = diagnosis.expected_metrics(m, 'optimal')
    assert metrics['success'] >= .99
    assert metrics['duplicate'] <= .01
    assert metrics['max_ticks'] <= diagnosis.HORIZON


def test_model_rejects_impossible_public_history_instead_of_renormalizing():
    m = model('unavailable')
    with pytest.raises(ValueError, match='impossible'):
        m.update(m.initial(), 'journal', 'complete')


def test_primary_response_requires_all_expected_fields():
    m = model()
    assert m.canonical('fields', {'records': [{'record_id': 'r', 'fields': {'x': 1}}]}, {'x': 1, 'y': 2}) == 'partial'
    assert m.canonical('fields', {'records': [{'record_id': 'r', 'fields': {'x': 1, 'y': 2}}]}, {'x': 1, 'y': 2}) == 'complete'
