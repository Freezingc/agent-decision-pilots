"""P1: enumerable timeout recovery POMDP. Synthetic CPU-only feasibility probe.

State is (execution count capped at 2, ticks until first write is visible).
The count and delay are NEVER given to deployable policies. Reads have false
negatives but no false positives. Every retry is non-idempotent and its reply
times out, so a retry is an uncertain write rather than a free recovery oracle.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

State = tuple[int, int]
Belief = dict[State, float]
NONTERMINAL = ("wait", "check", "retry")
TERMINAL = ("accept", "abandon")


@dataclass(frozen=True)
class Parameters:
    not_executed: float = .35
    stale_read: float = .2
    delay: tuple[tuple[int, float], ...] = ((0, .25), (1, .25), (3, .5))
    retry_success: float = .95
    time_cost: float = .2
    check_cost: float = 1.0
    retry_cost: float = 1.0
    missing_loss: float = 40.0
    duplicate_loss: float = 80.0
    abandon_loss: float = 8.0

    def __post_init__(self):
        for probability in (self.not_executed, self.stale_read, self.retry_success):
            if not 0 <= probability <= 1:
                raise ValueError("invalid probability")
        if not self.delay or any(d < 0 or int(d) != d or p < 0 for d, p in self.delay):
            raise ValueError("invalid delay distribution")
        if not math.isclose(sum(p for _, p in self.delay), 1, abs_tol=1e-12):
            raise ValueError("delay probabilities must sum to one")
        if min(self.time_cost, self.check_cost, self.retry_cost,
               self.missing_loss, self.duplicate_loss, self.abandon_loss) < 0:
            raise ValueError("negative cost")


def _merge(items):
    merged = {}
    for state, observation, probability in items:
        key = (state, observation)
        merged[key] = merged.get(key, 0.0) + probability
    return [(s, o, p) for (s, o), p in sorted(merged.items()) if p > 0]


def initial_belief(p: Parameters) -> Belief:
    b = {(0, 0): p.not_executed}
    for delay, prob in p.delay:
        state = (1, delay)
        b[state] = b.get(state, 0) + (1 - p.not_executed) * prob
    return {s: probability for s, probability in b.items() if probability > 0}


@lru_cache(maxsize=None)
def transition(state: State, action: str, p: Parameters):
    """Exact joint transition-observation kernel; one tick per nonterminal action."""
    count, delay = state
    delay = max(0, delay - 1)
    if action == "wait":
        return [((count, delay), "none", 1.0)]
    if action == "check":
        present = (1 - p.stale_read) if count > 0 and delay == 0 else 0.0
        return _merge([((count, delay), "present", present),
                       ((count, delay), "absent", 1 - present)])
    if action == "retry":
        outcomes = [((count, delay), "timeout", 1 - p.retry_success)]
        for new_delay, probability in p.delay:
            first_visible = new_delay if count == 0 else min(delay, new_delay)
            outcomes.append(((min(2, count + 1), first_visible), "timeout",
                             p.retry_success * probability))
        return _merge(outcomes)
    raise ValueError(f"not a nonterminal action: {action}")


def predict_observations(belief: Belief, action: str, p: Parameters):
    joint = {}
    for state, probability in belief.items():
        for next_state, observation, conditional in transition(state, action, p):
            joint.setdefault(observation, {})
            bucket = joint[observation]
            bucket[next_state] = bucket.get(next_state, 0) + probability * conditional
    return joint


def update_belief(belief: Belief, action: str, observation: str, p: Parameters):
    bucket = predict_observations(belief, action, p).get(observation, {})
    likelihood = sum(bucket.values())
    if likelihood <= 0:
        raise ValueError("observation impossible under declared model")
    return {s: v / likelihood for s, v in bucket.items() if v > 0}, likelihood


def action_cost(action: str, p: Parameters) -> float:
    if action in TERMINAL:
        return 0.0
    return p.time_cost + {"wait": 0.0, "check": p.check_cost, "retry": p.retry_cost}[action]


def outcome(state: State, terminal: str):
    count = state[0]
    return {"success": int(count == 1 and terminal == "accept"),
            "duplicate": int(count >= 2),
            "abstention": int(terminal == "abandon"),
            "false_completion": int(count == 0 and terminal == "accept"),
            "unconfirmed_single": int(count == 1 and terminal == "abandon")}


def terminal_loss(state: State, terminal: str, p: Parameters) -> float:
    result = outcome(state, terminal)
    return (result["duplicate"] * p.duplicate_loss
            + result["false_completion"] * p.missing_loss
            + result["abstention"] * p.abandon_loss)


def _belief_key(belief: Belief):
    # Rounding is only a numerical memoization device, not a state observation.
    return tuple((s, round(v, 13)) for s, v in sorted(belief.items()) if v > 1e-14)


class BeliefPlanner:
    """Receding finite-horizon Bayes planner; known nominal parameters only."""
    def __init__(self, parameters: Parameters, lookahead: int = 4):
        self.p = parameters
        self.lookahead = lookahead
        self._value = lru_cache(maxsize=200000)(self._value_impl)

    def _value_impl(self, key, horizon):
        belief = dict(key)
        candidates = [(sum(v * terminal_loss(s, a, self.p) for s, v in belief.items()), a)
                      for a in TERMINAL]
        if horizon > 0:
            for action in NONTERMINAL:
                value = action_cost(action, self.p)
                for bucket in predict_observations(belief, action, self.p).values():
                    probability = sum(bucket.values())
                    if probability > 0:
                        posterior = {s: v / probability for s, v in bucket.items()}
                        value += probability * self._value(_belief_key(posterior), horizon - 1)[0]
                candidates.append((value, action))
        # Ties favor less intervention: accept, abandon, wait, check, retry.
        return min(candidates, key=lambda x: (round(x[0], 11),
                    ("accept", "abandon", "wait", "check", "retry").index(x[1])))

    def value(self, belief: Belief, horizon: int):
        return self._value(_belief_key(belief), horizon)

    def choose(self, belief: Belief, remaining: int):
        return self.value(belief, min(remaining, self.lookahead))[1]


@dataclass(frozen=True)
class Rule:
    wait: int = 1
    negatives: int = 1
    retries: int = 1
    immediate_retry: bool = False
    initial_terminal: str = ""

    def choose(self, history: list[dict], remaining: int):
        if not history and self.initial_terminal:
            if self.initial_terminal not in TERMINAL:
                raise ValueError("invalid initial terminal action")
            return self.initial_terminal
        if history and history[-1]["observation"] == "present":
            return "accept"
        if remaining == 0:
            return "abandon"
        if not history and self.immediate_retry:
            return "retry"
        last_retry = max((i for i, x in enumerate(history) if x["action"] == "retry"), default=-1)
        recent = history[last_retry + 1:]
        n_retries = sum(x["action"] == "retry" for x in history)
        if len(recent) < self.wait:
            return "wait"
        negatives = sum(x["action"] == "check" and x["observation"] == "absent" for x in recent)
        if negatives >= self.negatives and n_retries < self.retries:
            return "retry"
        return "check"


class Oracle:
    """Full-state finite-horizon optimal policy: diagnostic upper bound ONLY."""
    def __init__(self, p: Parameters):
        self.p = p

    @lru_cache(maxsize=None)
    def value(self, state, remaining):
        candidates = [(terminal_loss(state, a, self.p), a) for a in TERMINAL]
        if remaining:
            for a in NONTERMINAL:
                q = action_cost(a, self.p) + sum(prob * self.value(s, remaining - 1)[0]
                                                for s, _, prob in transition(state, a, self.p))
                candidates.append((q, a))
        return min(candidates, key=lambda x: (round(x[0], 11), ("accept", "abandon", "wait", "check", "retry").index(x[1])))


def _uniform(seed: int, turn: int, event: str):
    raw = hashlib.sha256(f"p1-v1:{seed}:{turn}:{event}".encode()).digest()
    return int.from_bytes(raw[:8], "big") / 2**64


def _draw(items, u):
    total = 0.0
    for value, probability in items:
        total += probability
        if u < total:
            return value
    if not math.isclose(total, 1, abs_tol=1e-10):
        raise ValueError("sampling distribution does not sum to one")
    return items[-1][0]


def simulate(actual: Parameters, policy, seed: int, horizon: int = 8,
             nominal: Parameters | None = None):
    state = _draw(list(initial_belief(actual).items()), _uniform(seed, -1, "initial"))
    belief = initial_belief(nominal or actual)
    history, trajectory = [], []
    cost = 0.0
    terminal = None
    for turn in range(horizon + 1):
        remaining = horizon - turn
        if isinstance(policy, BeliefPlanner):
            action = policy.choose(belief, remaining)
        elif isinstance(policy, Oracle):
            action = policy.value(state, remaining)[1]
        else:
            action = policy.choose(history, remaining)
        before = state
        if action in TERMINAL:
            terminal = action
            trajectory.append({"turn": turn, "action": action, "observation": "terminal",
                               "state_before": list(before), "state_after": list(state), "cost": 0.0})
            break
        if remaining <= 0:
            raise ValueError("policy violated hard horizon")
        kernel = transition(state, action, actual)
        state, observation = _draw([((s, o), prob) for s, o, prob in kernel],
                                   _uniform(seed, turn, action))
        step_cost = action_cost(action, actual)
        cost += step_cost
        item = {"turn": turn, "action": action, "observation": observation,
                "state_before": list(before), "state_after": list(state), "cost": step_cost}
        trajectory.append(item)
        # Public history excludes all simulator state. Only a planner uses beliefs.
        history.append({"action": action, "observation": observation})
        if isinstance(policy, BeliefPlanner):
            belief, likelihood = update_belief(belief, action, observation, nominal or actual)
            item["belief_after"] = [[list(s), v] for s, v in sorted(belief.items())]
            item["observation_likelihood_nominal"] = likelihood
    if terminal is None:
        raise RuntimeError("unterminated trajectory")
    return {"seed": seed, "initial_state": trajectory[0]["state_before"],
            "terminal_state": list(state), "terminal_action": terminal,
            **outcome(state, terminal), "cost": cost,
            "loss": cost + terminal_loss(state, terminal, actual),
            "checks": sum(x["action"] == "check" for x in history),
            "retries": sum(x["action"] == "retry" for x in history),
            "waits": sum(x["action"] == "wait" for x in history),
            "trajectory": trajectory}


def aggregate(records):
    n = len(records)
    fields = ("success", "duplicate", "abstention", "false_completion", "unconfirmed_single",
              "cost", "loss", "checks", "retries", "waits")
    return {"n": n, **{k: sum(r[k] for r in records) / n for k in fields},
            "counts": {k: sum(r[k] for r in records) for k in fields[:5]}}


def _paired(left, right):
    differences = {key: [a[key] - b[key] for a, b in zip(left, right)]
                   for key in ("success", "duplicate", "abstention", "cost", "loss")}
    result = {}
    for key, values in differences.items():
        mean = sum(values) / len(values)
        variance = sum((v - mean)**2 for v in values) / (len(values) - 1)
        half = 1.96 * math.sqrt(variance / len(values))
        result[key] = {"difference": mean, "normal_ci95": [mean - half, mean + half]}
    return result


def scenarios():
    result = []
    for failure, not_executed, stale in (("low", .1, .05), ("medium", .35, .2), ("high", .65, .4)):
        for cost_name, cost in (("low", .1), ("medium", 1.0), ("high", 4.0)):
            nominal = Parameters(not_executed=not_executed, stale_read=stale, check_cost=cost)
            for shift in (False, True):
                actual = replace(nominal, delay=((0, .1), (3, .2), (6, .7))) if shift else nominal
                name = f"{failure}-fault_{cost_name}-check_{'delay-shift' if shift else 'matched'}"
                result.append((name, actual, nominal, shift))
    return result


def run(output: Path, dev_n=200, test_n=600, seed=20261019):
    output.mkdir(parents=True, exist_ok=True)
    started, clock = datetime.now(timezone.utc).isoformat(), time.monotonic()
    protocol = {"schema": "p1-v1", "evidence": "synthetic enumerable POMDP; not LLM evidence",
                "horizon": 8, "planner_lookahead": 4, "dev_n_per_matched_condition": dev_n,
                "test_n_per_condition": test_n, "seed": seed,
                "dev_seed_formula": "seed*1000000+i", "test_seed_formula": "seed*1000000+100000+block*10000+i",
                "test_blocks": 3, "objective": "minimize cost+40*false_completion+80*duplicate+8*abstention",
                "rules_grid": {"wait": [0, 1, 2, 3], "negatives": [1, 2, 3], "retries": [1, 2],
                               "extra_trivial_rules": ["immediate_accept", "immediate_abandon"]},
                "mismatch": "planner and selected rule retain matched dev parameters; actual delays shift to 0/3/6",
                "all_conditions": [{"name": n, "actual": asdict(a), "nominal": asdict(m), "shift": s}
                                   for n, a, m, s in scenarios()]}
    protocol_bytes = (json.dumps(protocol, sort_keys=True, indent=2) + "\n").encode()
    (output / "frozen-design.json").write_bytes(protocol_bytes)
    frozen_hash = hashlib.sha256(protocol_bytes).hexdigest()
    dev_seeds = [seed * 1000000 + i for i in range(dev_n)]
    test_seeds = [seed * 1000000 + 100000 + block * 10000 + i
                  for block in range(3) for i in range(test_n // 3)]
    if len(test_seeds) != test_n:
        raise ValueError("test_n must be divisible by three")
    grid = [Rule(w, n, r) for w in (0, 1, 2, 3) for n in (1, 2, 3) for r in (1, 2)]
    grid += [Rule(initial_terminal=a) for a in TERMINAL]
    chosen, dev_records = {}, []
    # Complete all dev selection before the first test result is generated.
    for name, actual, _, shifted in scenarios():
        if shifted:
            continue
        results = []
        for rule in grid:
            metrics = aggregate([simulate(actual, rule, s) for s in dev_seeds])
            results.append({"rule": asdict(rule), "metrics": metrics})
        best = min(results, key=lambda x: (x["metrics"]["loss"], x["metrics"]["cost"],
                                          tuple(x["rule"].values())))
        chosen[name] = Rule(**best["rule"])
        dev_records.append({"condition": name, "selected": best, "grid": results})
    (output / "dev-selection.json").write_text(json.dumps(dev_records, indent=2) + "\n")
    dev_hash = hashlib.sha256((output / "dev-selection.json").read_bytes()).hexdigest()
    selection_frozen = datetime.now(timezone.utc).isoformat()
    summaries, paired, blocks = [], [], []
    with gzip.open(output / "test-trajectories.jsonl.gz", "wt") as traces:
        for name, actual, nominal, shifted in scenarios():
            dev_name = name.replace("delay-shift", "matched")
            policies = {"immediate_retry": Rule(0, 99, 1, True),
                        "fixed_wait_check": Rule(2, 1, 1),
                        "consistent_checks": Rule(0, 2, 1),
                        "dev_best_rule": chosen[dev_name],
                        "belief_h4_nominal": BeliefPlanner(nominal, 4),
                        "hidden_state_oracle": Oracle(actual)}
            all_records = {}
            for method, policy in policies.items():
                records = []
                for task_seed in test_seeds:
                    record = simulate(actual, policy, task_seed, nominal=nominal)
                    traces.write(json.dumps({"condition": name, "method": method,
                                             "parameters": asdict(actual), **record}, separators=(",", ":")) + "\n")
                    records.append(record)
                all_records[method] = records
                summaries.append({"condition": name, "method": method, "shifted": shifted,
                                  **aggregate(records)})
                for block in range(3):
                    part = records[block * (test_n // 3):(block + 1) * (test_n // 3)]
                    blocks.append({"condition": name, "method": method, "block": block, **aggregate(part)})
            paired.append({"condition": name, "left": "belief_h4_nominal", "right": "dev_best_rule",
                           "n": test_n, "differences": _paired(all_records["belief_h4_nominal"],
                                                              all_records["dev_best_rule"])})
            print(json.dumps({"completed_condition": name, "planner": aggregate(all_records["belief_h4_nominal"]),
                              "rule": aggregate(all_records["dev_best_rule"])}), flush=True)
    ended = datetime.now(timezone.utc).isoformat()
    report = {"schema": "p1-v1", "status": "completed", "evidence": "synthetic simulation",
              "design_sha256": frozen_hash, "dev_selection_sha256": dev_hash,
              "started_utc": started, "dev_selection_frozen_utc": selection_frozen,
              "ended_utc": ended, "elapsed_seconds": time.monotonic() - clock,
              "python": sys.version, "platform": platform.platform(), "command": sys.argv,
              "paid_api_calls": 0, "paid_api_cost": 0, "gpu_used": False,
              "uncertainty": "Paired normal 95% intervals, descriptive only; no rare-event certification or multiplicity adjustment",
              "summary": summaries, "paired_planner_minus_dev_rule": paired, "seed_blocks": blocks}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = ["condition", "method", "shifted", "n", "success", "duplicate", "abstention", "false_completion",
              "unconfirmed_single", "cost", "loss", "checks", "retries", "waits"]
    with (output / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in summaries)
    evidence = {}
    for name in ("frozen-design.json", "dev-selection.json", "summary.json", "summary.csv", "test-trajectories.jsonl.gz"):
        path = output / name
        evidence[name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    evidence["source"] = {"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/p1/run-v1"))
    parser.add_argument("--dev-n", type=int, default=200)
    parser.add_argument("--test-n", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()
    run(args.output, args.dev_n, args.test_n, args.seed)


if __name__ == "__main__":
    main()
