"""P5 post hoc explanatory probe on the real loopback HTTP/SQLite harness.

Exhaustive finite controlled histories, not natural samples or a new learning
algorithm. Only opaque ID values are normalized; all public fields and order
remain. Audit results are hidden terminal judges, never part of public history.
Run once under an external hard timeout; existing output is never overwritten.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

from agent_pilots.state_service import LocalStateService

ACTIONS = ("cache", "journal", "fields", "wait")
FIELDS = {"item": "book", "quantity": 2, "delivery": "pickup"}


def sequence_grid():
    return list(itertools.product(ACTIONS, repeat=5))


def branch_weights(world):
    if world not in ("A", "B"):
        raise ValueError("unknown world")
    return {"short": Fraction(4, 5), "ambiguous": Fraction(1, 5)}


def normalize_ids(value):
    maps = {"episode_id": {}, "operation_id": {}, "record_id": {}}
    prefixes = {"episode_id": "episode", "operation_id": "operation", "record_id": "record"}

    def visit(item, field=None, business=False):
        if isinstance(item, dict):
            return {key: visit(val, key, business or key == "fields" or key == "expected_fields") for key, val in item.items()}
        if isinstance(item, list):
            return [visit(val, field, business) for val in item]
        if field in maps and item is not None and not business:
            if not isinstance(item, str):
                raise ValueError("opaque ID must be a string")
            mapping = maps[field]
            if item not in mapping:
                mapping[item] = f"<{prefixes[field]}-{len(mapping)}>"
            return mapping[item]
        return item

    return visit(value)


def public_episode(record):
    return normalize_ids(record["public"])


def scenario(world, branch, *, atomic=False, idempotent=False):
    branch_weights(world)
    if branch not in ("short", "ambiguous"):
        raise ValueError("unknown branch")
    return {"template": "reservation", "expected_fields": dict(FIELDS),
            "outcome": "none" if world == "A" and branch == "ambiguous" else "full",
            "apply_delay": 2 if branch == "short" else 9,
            "cache_lag": 0, "journal_lag": 0,
            "journal_mode": "atomic" if atomic else "lagged", "idempotent": idempotent}


class MeasuredService:
    def __init__(self, service):
        self.service = service
        self.calls = []

    def request(self, endpoint, *args, **kwargs):
        start = time.perf_counter_ns()
        ok = False
        try:
            result = getattr(self.service, endpoint)(*args, **kwargs)
            ok = True
            return result
        finally:
            self.calls.append({"endpoint": endpoint, "ms": (time.perf_counter_ns() - start) / 1e6, "ok": ok})

    def episode(self, world, branch, actions, *, atomic=False, idempotent=False,
                terminal="observation_window_end"):
        config = scenario(world, branch, atomic=atomic, idempotent=idempotent)
        started = self.request("start_episode", config)
        steps = []
        for tick, action in enumerate(actions, 1):
            response = self.request("action", started["episode_id"], action)
            steps.append({"tick": tick, "action": action, "response": response})
        audit = self.request("audit", started["episode_id"], drain=True)
        row = {"world": world, "branch": branch, "config": config,
               "public": {"start": started, "steps": steps}, "terminal_audit": audit,
               "controller_terminal": terminal,
               "controller_success": terminal == "accept" and audit["business_success"]}
        row["normalized_public"] = public_episode(row)
        return row

    def summary(self):
        by_endpoint = {}
        for endpoint in sorted({r["endpoint"] for r in self.calls}):
            values = [r["ms"] for r in self.calls if r["endpoint"] == endpoint]
            by_endpoint[endpoint] = {"count": len(values), "total_ms": sum(values),
                                     "mean_ms": sum(values) / len(values), "max_ms": max(values)}
        return {"requests": len(self.calls), "failed_requests": sum(not r["ok"] for r in self.calls),
                "total_measured_ms": sum(r["ms"] for r in self.calls), "by_endpoint": by_endpoint,
                "scope": "client start_episode/action/audit wall time including HTTP and SQLite; excludes startup/close; not production latency"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare_pair(left, right):
    a, b = public_episode(left), public_episode(right)
    initial = a["start"] == b["start"]
    steps = [x == y for x, y in zip(a["steps"], b["steps"], strict=True)]
    return {"initial_equal": initial, "step_equal": steps,
            "full_public_equal": a == b,
            "different_step_indices": [i + 1 for i, same in enumerate(steps) if not same]}


def run(output, protocol):
    output, protocol = Path(output), Path(protocol)
    output.mkdir(parents=True, exist_ok=False)
    began = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    project = Path(__file__).resolve().parents[2]
    source_paths = [Path(__file__).resolve(), project / "src/agent_pilots/state_service.py",
                    project / "tests/test_identifiability.py", protocol.resolve()]
    frozen = {"schema": "p5-identifiability-v1", "started_utc": began,
              "interpretation": "post-P5 explanatory validation, controlled exhaustive histories, no new contribution claim",
              "sequence_count": 1024, "readonly_steps": 5, "worlds": ["A", "B"],
              "branches": ["short", "ambiguous"], "main_episode_count": 4096,
              "branch_weights": {w: {b: {"numerator": p.numerator, "denominator": p.denominator}
                                     for b, p in branch_weights(w).items()} for w in ("A", "B")},
              "fixed_fields": FIELDS, "source_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
              "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
              "normalization": "first appearance per ID namespace, preserve all non-ID fields, lists and event order; business fields never normalized",
              "command": [sys.executable, *sys.argv], "hard_timeout_seconds": 300,
              "no_natural_sampling": True, "gpu_used": False, "paid_api_calls": 0}
    (output / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
    for source in source_paths:
        shutil.copyfile(source, output / source.name)
    differences = []
    pair_count = 0
    actual_episodes = 0
    main_rows = 0
    mechanism = []
    try:
        with LocalStateService(output / "service.sqlite") as raw_service:
            service = MeasuredService(raw_service)
            with gzip.open(output / "trajectories.jsonl.gz", "wt") as trajectories, (output / "comparisons.jsonl").open("w") as comparisons:
                for index, sequence in enumerate(sequence_grid()):
                    for branch in ("short", "ambiguous"):
                        pair = []
                        for world in ("A", "B"):
                            row = service.episode(world, branch, sequence)
                            row.update(sequence_index=index, sequence=list(sequence), role="enumeration")
                            trajectories.write(json.dumps(row, separators=(",", ":")) + "\n")
                            pair.append(row)
                            actual_episodes += 1
                            main_rows += 1
                        comparison = {"sequence_index": index, "sequence": list(sequence), "branch": branch,
                                      **compare_pair(*pair)}
                        comparisons.write(json.dumps(comparison) + "\n")
                        pair_count += 1
                        if not comparison["full_public_equal"]:
                            differences.append(comparison)
                    if (index + 1) % 128 == 0:
                        print(json.dumps({"event": "enumeration_progress", "sequences": index + 1,
                                          "episodes": actual_episodes, "elapsed_seconds": time.monotonic() - clock}), flush=True)

                def extra(world, branch, actions, role, **kwargs):
                    nonlocal actual_episodes
                    row = service.episode(world, branch, actions, **kwargs)
                    row["role"] = role
                    trajectories.write(json.dumps(row, separators=(",", ":")) + "\n")
                    actual_episodes += 1
                    mechanism.append(row)
                    return row

                fields5 = ["fields"] * 5
                retry = {}
                for branch in ("short", "ambiguous"):
                    retry[branch] = {w: extra(w, branch, fields5 + ["retry"], "nonidempotent_retry", terminal="accept") for w in ("A", "B")}
                abstain = {w: extra(w, "ambiguous", fields5, "abstain", terminal="abstain") for w in ("A", "B")}
                atomic = {w: extra(w, "ambiguous", ["journal"], "atomic_journal", atomic=True) for w in ("A", "B")}
                idempotent = {w: extra(w, "ambiguous", fields5 + ["retry"], "idempotent_retry", idempotent=True, terminal="accept") for w in ("A", "B")}
            (output / "http-calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in service.calls))
            measured = service.summary()

        checks = {"all_2048_main_pairs_equal": not differences and pair_count == 2048,
                  "main_4096_actual_episodes": main_rows == 4096,
                  "ambiguous_retry_A_one_B_two": retry["ambiguous"]["A"]["terminal_audit"]["record_count"] == 1 and retry["ambiguous"]["B"]["terminal_audit"]["record_count"] == 2,
                  "short_retry_both_duplicate": all(retry["short"][w]["terminal_audit"]["duplicate"] for w in ("A", "B")),
                  "abstain_controller_success_false": all(not abstain[w]["controller_success"] and abstain[w]["controller_terminal"] == "abstain" for w in ("A", "B")),
                  "abstain_final_worlds_differ": abstain["A"]["terminal_audit"]["record_count"] == 0 and abstain["B"]["terminal_audit"]["record_count"] == 1,
                  "atomic_journal_breaks_equivalence": atomic["A"]["public"]["steps"][0]["response"] == {"state": "not_found"} and atomic["B"]["public"]["steps"][0]["response"] == {"state": "pending"},
                  "idempotent_retry_both_one_complete": all(idempotent[w]["terminal_audit"]["record_count"] == idempotent[w]["terminal_audit"]["complete_count"] == 1 and not idempotent[w]["terminal_audit"]["duplicate"] for w in ("A", "B"))}
        require(measured["requests"] == 28740, "unexpected HTTP request count")
        require(actual_episodes == 4106, "unexpected total episode count")
        (output / "mechanism-cases.json").write_text(json.dumps(mechanism, indent=2) + "\n")
        report = {"schema": "p5-identifiability-v1", "status": "completed" if all(checks.values()) else "completed_with_failed_checks",
                  "started_utc": began, "ended_utc": datetime.now(timezone.utc).isoformat(),
                  "elapsed_seconds": time.monotonic() - clock, "main_episodes": main_rows,
                  "additional_episodes": actual_episodes - main_rows, "total_episodes": actual_episodes,
                  "sequence_count": 1024, "paired_branch_comparisons": pair_count,
                  "public_differences": differences, "checks": checks, "http": measured,
                  "branch_weights": frozen["branch_weights"],
                  "claim_scope": "exact normalized equality of 4096 controlled service trajectories for every fixed length-5 readonly sequence, plus equal declared mixture weights; not an estimate from natural samples",
                  "limits": ["logical ticks inject faults, not measured real network latency", "terminal audit excluded from policy observation", "no new theorem or algorithm", "does not imply indistinguishability with atomic journal, idempotency, trusted delay bounds or longer observation", "A/B differ by explicitly constructed latent delay model"]}
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        manifest = {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in output.iterdir() if p.is_file() and p.name != "manifest.json"}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"event": "completed", "status": report["status"], "episodes": actual_episodes,
                          "comparisons": pair_count, "checks": checks, "http": measured,
                          "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
        return 0 if all(checks.values()) else 2
    except BaseException as exc:
        failure = {"status": "failed", "ended_utc": datetime.now(timezone.utc).isoformat(),
                   "completed_episodes": actual_episodes, "error_type": type(exc).__name__, "error": str(exc),
                   "traceback": traceback.format_exc(), "retry_performed": False}
        (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/p5/identifiability-v1"))
    parser.add_argument("--protocol", type=Path, default=Path("docs/p5-identifiability-protocol.md"))
    args = parser.parse_args()
    raise SystemExit(run(args.output, args.protocol))


if __name__ == "__main__":
    main()
