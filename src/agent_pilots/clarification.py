"""P4: finite information-acquisition simulation; no LLM or real user claims.

Independent hidden bits E and P represent an environmental attribute and a
user preference. Probe observes E only; ask observes P only. Policies receive
public context, beliefs and channel parameters, never an episode's hidden bits.
The planner knows the true simulator likelihoods: it is an idealized diagnostic.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import time


@dataclass(frozen=True)
class Context:
    p_env: float
    p_pref: float
    w_env: float

    def __post_init__(self):
        if any(not 0 <= x <= 1 for x in (self.p_env, self.p_pref, self.w_env)):
            raise ValueError("Context probabilities and weights must be in [0, 1]")


@dataclass(frozen=True)
class Belief:
    p_env: float
    p_pref: float


@dataclass(frozen=True)
class Condition:
    name: str = "default"
    probe_cost: float = 0.03
    ask_cost: float = 0.12
    probe_accuracy: float = 0.9
    ask_accuracy: float = 0.9
    probe_missing: float = 0.0
    ask_missing: float = 0.0
    allow_probe: bool = True
    allow_ask: bool = True

    def __post_init__(self):
        if self.probe_cost < 0 or self.ask_cost < 0:
            raise ValueError("Costs must be nonnegative")
        if any(not 0 <= x <= 1 for x in (self.probe_accuracy, self.ask_accuracy,
                                        self.probe_missing, self.ask_missing)):
            raise ValueError("Channel parameters must be in [0, 1]")


@dataclass(frozen=True)
class Case:
    case_id: str
    split: str
    seed: int
    context: Context
    env: int
    pref: int


@dataclass(frozen=True)
class Rule:
    """Myopic value-to-cost thresholds; None disables that channel."""

    probe_multiplier: float | None
    ask_multiplier: float | None
    order: str = "probe_first"


def _uniform(case: Case, channel: str) -> float:
    raw = f"p4-v1:{case.seed}:{case.case_id}:{channel}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") / 2**64


def observe(case: Case, condition: Condition, channel: str) -> int | None:
    if channel not in {"probe", "ask"}:
        raise ValueError("Unknown information channel")
    if not getattr(condition, "allow_" + channel):
        raise ValueError("Information channel is unavailable")
    if _uniform(case, channel + ":missing") < getattr(condition, channel + "_missing"):
        return None
    # Neither channel accesses the other hidden variable.
    truth = case.env if channel == "probe" else case.pref
    correct = _uniform(case, channel + ":noise") < getattr(condition, channel + "_accuracy")
    return truth if correct else 1 - truth


def update(belief: Belief, channel: str, observation: int | None, condition: Condition) -> Belief:
    if channel not in {"probe", "ask"}:
        raise ValueError("Unknown information channel")
    if observation is None:
        return belief
    p = belief.p_env if channel == "probe" else belief.p_pref
    q = getattr(condition, channel + "_accuracy")
    likelihood1 = q if observation == 1 else 1 - q
    likelihood0 = 1 - likelihood1
    denominator = p * likelihood1 + (1 - p) * likelihood0
    if denominator <= 0:
        raise ValueError("Impossible observation under supplied likelihood")
    posterior = p * likelihood1 / denominator
    return replace(belief, **{"p_env" if channel == "probe" else "p_pref": posterior})


def _stop(belief: Belief, context: Context) -> tuple[float, str]:
    value1 = context.w_env * belief.p_env + (1 - context.w_env) * belief.p_pref
    return (value1, "choose1") if value1 > 0.5 + 1e-12 else (1 - value1, "choose0")


def _outcomes(belief: Belief, condition: Condition, channel: str):
    p = belief.p_env if channel == "probe" else belief.p_pref
    q = getattr(condition, channel + "_accuracy")
    missing = getattr(condition, channel + "_missing")
    p1 = p * q + (1 - p) * (1 - q)
    for observation, probability in [(None, missing), (0, (1 - missing) * (1 - p1)),
                                     (1, (1 - missing) * p1)]:
        if probability > 1e-15:
            yield probability, update(belief, channel, observation, condition)


@lru_cache(maxsize=100000)
def information_gain(belief: Belief, context: Context, condition: Condition, channel: str) -> float:
    """Immediate gross gain from one channel followed by a final decision."""
    return max(0.0, sum(p * _stop(b, context)[0] for p, b in _outcomes(belief, condition, channel))
               - _stop(belief, context)[0])


@lru_cache(maxsize=100000)
def plan(belief: Belief, context: Context, condition: Condition,
         used: frozenset[str] = frozenset()) -> tuple[float, str]:
    """Exact horizon <= 2 Bayes planning, each information channel at most once."""
    best_value, best_action = _stop(belief, context)
    for channel in ("probe", "ask"):
        if channel in used or not getattr(condition, "allow_" + channel):
            continue
        value = -getattr(condition, channel + "_cost") + sum(
            probability * plan(posterior, context, condition, used | {channel})[0]
            for probability, posterior in _outcomes(belief, condition, channel)
        )
        if value > best_value + 1e-12:
            best_value, best_action = value, channel
    return best_value, best_action


def _action(belief, context, condition, used, policy):
    stop = _stop(belief, context)[1]
    available = [channel for channel in ("probe", "ask")
                 if channel not in used and getattr(condition, "allow_" + channel)]
    if policy == "planner":
        return plan(belief, context, condition, frozenset(used))[1]
    schedules = {"no_information": (), "never_ask": ("probe",), "always_ask": ("ask",),
                 "probe_then_ask": ("probe", "ask")}
    if isinstance(policy, str):
        if policy not in schedules:
            raise ValueError("Unknown policy")
        return next((a for a in schedules[policy] if a in available), stop)
    if not isinstance(policy, Rule):
        raise TypeError("Policy must be a fixed name or a Rule")
    gains = {}
    for channel in available:
        multiplier = getattr(policy, channel + "_multiplier")
        if multiplier is None:
            continue
        gain = information_gain(belief, context, condition, channel)
        if gain > multiplier * getattr(condition, channel + "_cost") + 1e-12:
            gains[channel] = gain - getattr(condition, channel + "_cost")
    order = ("ask", "probe") if policy.order == "ask_first" else ("probe", "ask")
    if policy.order == "max_net" and gains:
        return max(order, key=lambda channel: gains.get(channel, -math.inf))
    return next((channel for channel in order if channel in gains), stop)


def run_episode(case: Case, condition: Condition, policy: str | Rule, *, keep_trace=True) -> dict:
    belief = Belief(case.context.p_env, case.context.p_pref)
    used, trace, cost = set(), [], 0.0
    for _ in range(3):
        # The policy boundary receives no Case, truth or random outcome.
        action = _action(belief, case.context, condition, used, policy)
        if action.startswith("choose"):
            decision = int(action[-1])
            if keep_trace:
                trace.append({"action": action, "belief": asdict(belief), "cost": 0.0})
            break
        if action in used:
            raise RuntimeError("Repeated information action")
        used.add(action)
        observation = observe(case, condition, action)
        charge = getattr(condition, action + "_cost")
        cost += charge
        posterior = update(belief, action, observation, condition)
        if keep_trace:
            trace.append({"action": action, "observation": observation, "cost": charge,
                          "belief_before": asdict(belief), "belief_after": asdict(posterior)})
        belief = posterior
    else:
        raise RuntimeError("Episode did not terminate within two information actions")
    environment_match, preference_match = int(decision == case.env), int(decision == case.pref)
    reward = case.context.w_env * environment_match + (1 - case.context.w_env) * preference_match
    return {"case_id": case.case_id, "seed": case.seed, "split": case.split, "decision": decision,
            "reward": reward, "cost": cost, "net_utility": reward - cost,
            "environment_match": environment_match, "preference_match": preference_match,
            "probe_count": int("probe" in used), "ask_count": int("ask" in used), "trace": trace}


@lru_cache(maxsize=100000)
def exact_value(belief: Belief, context: Context, condition: Condition, policy: str | Rule,
                used: frozenset[str] = frozenset()) -> float:
    """Known-model expectation of a frozen policy, not an empirical outcome."""
    action = _action(belief, context, condition, used, policy)
    if action.startswith("choose"):
        value1 = context.w_env * belief.p_env + (1 - context.w_env) * belief.p_pref
        return value1 if action == "choose1" else 1 - value1
    return -getattr(condition, action + "_cost") + sum(
        probability * exact_value(posterior, context, condition, policy, used | {action})
        for probability, posterior in _outcomes(belief, condition, action)
    )


def make_cases(n: int, seed: int, split: str) -> list[Case]:
    if split not in {"dev", "test"} or n < 1:
        raise ValueError("Require dev/test split and positive number of cases")
    rng = random.Random(f"p4-v1:{split}:{seed}")
    result = []
    for index in range(n):
        context = Context(rng.choice((0.2, 0.5, 0.8)), rng.choice((0.2, 0.5, 0.8)),
                          rng.choice((0.2, 0.5, 0.8)))
        env = int(rng.random() < context.p_env)
        pref = int(rng.random() < context.p_pref)
        result.append(Case(f"{split}/{seed}/{index:05d}", split, seed, context, env, pref))
    return result


def rule_grid():
    # Fixed policies are eligible too: a complex threshold need not win.
    policies = ["no_information", "never_ask", "always_ask", "probe_then_ask"]
    multipliers = (0.0, 0.5, 1.0, 2.0, None)
    policies += [Rule(p, a, order) for p, a, order in itertools.product(
        multipliers, multipliers, ("probe_first", "ask_first", "max_net"))]
    return policies


def policy_spec(policy):
    return asdict(policy) if isinstance(policy, Rule) else policy


def _tune(cases, condition):
    if not cases or any(case.split != "dev" for case in cases):
        raise ValueError("Simple rule fitting accepts nonempty dev cases only")
    ledger = []
    policies = rule_grid()
    for index, policy in enumerate(policies):
        outcomes = [run_episode(case, condition, policy, keep_trace=False) for case in cases]
        ledger.append({"index": index, "policy": policy_spec(policy),
                       "mean_net_utility": statistics.fmean(x["net_utility"] for x in outcomes),
                       "mean_calls": statistics.fmean(x["probe_count"] + x["ask_count"] for x in outcomes)})
    winner = max(ledger, key=lambda row: (round(row["mean_net_utility"], 12), -row["mean_calls"], -row["index"]))
    return policies[winner["index"]], ledger


def fit_simple(cases, condition):
    return _tune(cases, condition)[0]


def conditions():
    profiles = {
        "clean": {"probe_accuracy": 1.0, "ask_accuracy": 1.0},
        "noisy": {"probe_accuracy": 0.85, "ask_accuracy": 0.8},
        "missing": {"probe_accuracy": 0.9, "ask_accuracy": 0.9,
                    "probe_missing": 0.4, "ask_missing": 0.4},
        "no_probe": {"allow_probe": False, "ask_accuracy": 0.9},
        "no_user": {"allow_ask": False, "probe_accuracy": 0.9},
    }
    return [Condition(name=f"{profile}_p{pc:g}_a{ac:g}", probe_cost=pc, ask_cost=ac, **parameters)
            for profile, parameters in profiles.items()
            for pc in (0.01, 0.08, 0.25) for ac in (0.02, 0.12, 0.35)]


def _statistics(outcomes):
    return {key: statistics.fmean(row[key] for row in outcomes) for key in (
        "net_utility", "reward", "cost", "environment_match", "preference_match", "probe_count", "ask_count")}


def _paired(left, right):
    differences = [a["net_utility"] - b["net_utility"] for a, b in zip(left, right, strict=True)]
    mean = statistics.fmean(differences)
    halfwidth = 1.96 * statistics.stdev(differences) / math.sqrt(len(differences)) if len(differences) > 1 else 0.0
    return {"mean": mean, "ci95_normal_descriptive": [mean - halfwidth, mean + halfwidth],
            "n": len(differences), "unit": "paired simulated episode"}


def run_study(output: Path, dev_per_seed=128, test_per_seed=256):
    output = Path(output)
    if (output / "summary.json").exists():
        raise ValueError("Output already exists; choose a new subdirectory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    dev_seeds, test_seeds = (17, 23), (101, 211, 307)
    dev = [case for seed in dev_seeds for case in make_cases(dev_per_seed, seed, "dev")]
    test = [case for seed in test_seeds for case in make_cases(test_per_seed, seed, "test")]
    design = {"version": "p4-v1-diagnostic-supplement", "evidence_type": "structural_simulation_only",
              "dev_seeds": dev_seeds, "test_seeds": test_seeds,
              "dev_per_seed": dev_per_seed, "test_per_seed": test_per_seed,
              "conditions": [asdict(c) for c in conditions()],
              "context_grid": {"p_env": [0.2, 0.5, 0.8], "p_pref": [0.2, 0.5, 0.8], "w_env": [0.2, 0.5, 0.8]},
              "rule_grid": [policy_spec(p) for p in rule_grid()],
              "selection": "dev mean net utility; tie fewer calls, then grid order",
              "likelihoods": "planner AND myopic rules know true simulator parameters",
              "supplement": "After retained run-v1: add untrained greedy control and exact finite-model expectations; no environment, seeds or tuned grid changed",
              "reward": "w_env*1[decision==E]+(1-w_env)*1[decision==P]; net=reward-paid action costs",
              "pairing": "identical latent cases and potential observations across policies and conditions",
              "cost_unit": "dimensionless utility, not money/tokens/user effort measurements",
              "ex_ante_decision": "Do not advance if threshold rules explain gains; any residual is known-model upper-bound diagnostic"}
    def write(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    write("design.json", design)
    # Freeze design before opening test outcomes; save actual latent cases for audit.
    write("cases.json", {"dev": [asdict(c) for c in dev], "test": [asdict(c) for c in test]})
    transfer_condition = Condition(name="transfer_training", probe_cost=0.08, ask_cost=0.12,
                                   probe_accuracy=0.85, ask_accuracy=0.8)
    transfer_rule, transfer_ledger = _tune(dev, transfer_condition)
    write("transfer_dev_selection.json", {"condition": asdict(transfer_condition),
          "selected": policy_spec(transfer_rule), "ledger": transfer_ledger})
    results, tuning = [], []
    with (output / "episodes.jsonl.gz").open("wb") as raw_stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0) as stream:
            for condition in conditions():
                chosen, ledger = _tune(dev, condition)
                tuning.append({"condition": condition.name, "selected": policy_spec(chosen), "ledger": ledger})
                # Strongest fixed policy is selected on dev, never on test.
                fixed_winner = max(ledger[:4], key=lambda r: (
                    round(r["mean_net_utility"], 12), -r["mean_calls"], -r["index"]))["policy"]
                policies = {name: name for name in ("no_information", "never_ask", "always_ask", "probe_then_ask", "planner")}
                policies.update(simple_dev=chosen, simple_frozen_transfer=transfer_rule, fixed_dev=fixed_winner,
                                myopic_known=Rule(1.0, 1.0, "max_net"))
                outcomes = {}
                for name, policy in policies.items():
                    outcomes[name] = []
                    for case in test:
                        row = run_episode(case, condition, policy)
                        outcomes[name].append(row)
                        stream.write((json.dumps({"condition": condition.name, "policy": name, **row},
                                                  separators=(",", ":"), allow_nan=False) + "\n").encode())
                contexts = [Context(e, p, w) for e, p, w in itertools.product((0.2, 0.5, 0.8), repeat=3)]
                exact = {name: statistics.fmean(exact_value(Belief(c.p_env, c.p_pref), c, condition, policy)
                                                for c in contexts) for name, policy in policies.items()}
                results.append({"condition": asdict(condition), "simple_selected": policy_spec(chosen),
                                "fixed_selected": fixed_winner,
                                "known_model_expected_utility": exact,
                                "test": {name: {"overall": _statistics(rows),
                                    "by_seed": {str(seed): _statistics([r for r in rows if r["seed"] == seed])
                                                for seed in test_seeds}} for name, rows in outcomes.items()},
                                "paired_planner_minus": {name: _paired(outcomes["planner"], outcomes[name])
                                                          for name in outcomes if name != "planner"}})
                print(json.dumps({"condition": condition.name, "done": len(results), "total": 45}), flush=True)
    write("dev_grid_scores.json", tuning)
    summary = {"evidence_type": "structural_simulation_only", "n_conditions": len(results),
               "n_dev_per_condition": len(dev), "n_test_per_condition": len(test),
               "transfer_rule": policy_spec(transfer_rule), "results": results,
               "limitations": ["No LLM, learned policy, real user or measured user-cost evidence",
                               "Known likelihoods and calibrated public priors are idealized assumptions",
                               "Normal paired intervals are descriptive and unadjusted across conditions",
                               "Conditions reuse cases; do not pool condition-by-case cells as independent tasks",
                               "Threshold parameters selected on dev separately per condition; transfer rule frozen on one dev condition"]}
    write("summary.json", summary)
    files = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
             for p in sorted(output.iterdir()) if p.is_file()}
    write("manifest.json", {"status": "completed", "seconds": time.monotonic() - started,
          "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "files": files})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/p4/run-v1"))
    parser.add_argument("--dev-per-seed", type=int, default=128)
    parser.add_argument("--test-per-seed", type=int, default=256)
    args = parser.parse_args()
    run_study(args.output, args.dev_per_seed, args.test_per_seed)


if __name__ == "__main__":
    main()
