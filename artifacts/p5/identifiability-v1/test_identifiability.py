"""Contract tests for the finite-sequence P5 identifiability probe."""
import copy
from fractions import Fraction

import pytest

try:
    from artifacts.p5.probe_identifiability import (
        ACTIONS, branch_weights, normalize_ids, public_episode, sequence_grid,
    )
except ImportError:
    ACTIONS = None


def test_sequence_grid_contains_every_ordered_length_five_readonly_history():
    assert ACTIONS == ("cache", "journal", "fields", "wait")
    rows = sequence_grid()
    assert len(rows) == 1024
    assert len(set(rows)) == 1024
    assert rows[0] == ("cache",) * 5
    assert rows[-1] == ("wait",) * 5


def test_frozen_branch_weight_is_same_and_exact_across_worlds():
    assert ACTIONS is not None, "the identifiability probe is not implemented"
    assert branch_weights("A") == branch_weights("B") == {"short": Fraction(4,5), "ambiguous": Fraction(1,5)}


def test_normalization_replaces_uuids_without_dropping_response_fields():
    assert ACTIONS is not None, "the identifiability probe is not implemented"
    row = {"episode_id": "a" * 32, "operation_id": "b" * 32,
           "responses": [{"state": "present", "record_id": "c" * 32,
                          "fields": {"item": "book", "quantity": 2}}]}
    original_keys = copy.deepcopy(row)
    normalized = normalize_ids(row)
    assert set(normalized) == set(original_keys)
    assert normalized["episode_id"] == "<episode-0>"
    assert normalized["operation_id"] == "<operation-0>"
    assert normalized["responses"][0] == {"state": "present", "record_id": "<record-0>",
                                          "fields": {"item": "book", "quantity": 2}}
    assert normalize_ids(row) == normalized


def test_public_episode_comparison_keeps_every_non_uuid_difference():
    assert ACTIONS is not None, "the identifiability probe is not implemented"
    a = {"public": {"episode_id": "a" * 32, "responses": [{"state": "absent"}]}}
    b = {"public": {"episode_id": "b" * 32, "responses": [{"state": "present"}]}}
    assert public_episode(a) != public_episode(b)
    b["public"]["responses"][0]["state"] = "absent"
    assert public_episode(a) == public_episode(b)


def test_normalization_preserves_order_and_record_identity():
    assert ACTIONS is not None, "the identifiability probe is not implemented"
    row=[{"record_id":"d"*32},{"record_id":"e"*32},{"record_id":"d"*32}]
    assert normalize_ids(row)==[{"record_id":"<record-0>"},{"record_id":"<record-1>"},{"record_id":"<record-0>"}]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
