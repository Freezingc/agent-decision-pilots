"""Execute frozen P5 controllers against an independent HTTP/SQLite service."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sqlite3
import statistics
import time

from .diagnosis import (ABSTAIN_LOSS, COSTS, DUPLICATE_LOSS, HORIZON, TIME_COST,
                        DiagnosticModel, FixedRule, cases, expected_metrics, select_fixed)
from .state_service import LocalStateService

TEMPLATES = {
    'reservation': {'guest': 'pilot-user', 'date': '2030-01-12', 'room': 'A12'},
    'invoice': {'customer': 'pilot-customer', 'amount_cents': 2500, 'currency': 'CNY'},
    'shipment': {'recipient': 'pilot-user', 'address': 'test-address', 'parcel_count': 2},
}
BASELINES = ('myopic', 'deadline', 'rollout', 'posterior', 'fixed-loss', 'fixed-quality')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths():
    root = Path(__file__).resolve().parents[2]
    return [root / 'docs/p5-protocol.md', Path(__file__),
            root / 'src/agent_pilots/diagnosis.py', root / 'src/agent_pilots/state_service.py']


def public_record_id(action, response):
    if action == 'fields':
        return response['records'][0]['record_id'] if response['records'] else None
    return response.get('record_id')


def actual_episode(service, model, policy, case, template, idempotent=False):
    config = {'template': template, 'expected_fields': TEMPLATES[template],
              'outcome': case.outcome, 'apply_delay': case.apply_delay,
              'cache_lag': case.cache_lag, 'journal_lag': case.journal_lag,
              'journal_mode': model.journal_mode, 'idempotent': idempotent}
    public = service.start_episode(config)
    belief, history, trajectory = model.initial(), [], []
    cost, ticks, record_id, model_miss = 0., 0, None, False
    request_ms, planning_ms = [], []
    for _ in range(HORIZON + 1):
        t0 = time.perf_counter()
        action = 'abstain' if model_miss else ('retry' if idempotent else model.choose(policy, belief, history))
        planning_ms.append((time.perf_counter()-t0)*1000)
        if action in ('accept', 'abstain'):
            break
        if ticks >= HORIZON:
            raise RuntimeError('controller exceeded budget')
        t0 = time.perf_counter()
        response = service.action(public['episode_id'], action, record_id if action == 'repair' else None)
        request_ms.append((time.perf_counter()-t0)*1000)
        ticks += 1
        cost += model.cost(action)
        observation = model.canonical(action, response, public['expected_fields'])
        record_id = public_record_id(action, response) or record_id
        trajectory.append({'tick': ticks, 'action': action, 'response': response,
                           'observation': observation, 'cost': model.cost(action)})
        if action in ('retry', 'repair'):
            break
        history.append({'tick': ticks, 'action': action, 'observation': observation})
        try:
            belief = model.update(belief, action, observation)
        except ValueError as exc:
            if 'impossible public history' not in str(exc):
                raise
            model_miss = True
    else:
        raise RuntimeError('controller did not terminate')
    audit = service.audit(public['episode_id'], drain=True)
    declared = action != 'abstain'
    success = declared and audit['business_success']
    duplicate = audit['duplicate']
    metrics = {'success': int(success), 'duplicate': int(duplicate),
               'false_completion': int(declared and not audit['business_success']),
               'abstention': int(not declared), 'model_miss': int(model_miss),
               'cost': cost, 'ticks': ticks,
               'loss': cost + ABSTAIN_LOSS*int(not declared) + DUPLICATE_LOSS*int(duplicate)}
    return {'public_start': public, 'case': asdict(case), 'template': template,
            'model_regime': model.regime, 'journal_mode': model.journal_mode,
            'cost_profile': model.cost_profile, 'idempotent': idempotent,
            'terminal_action': action, 'metrics': metrics, 'trajectory': trajectory,
            'audit': audit, 'http_request_ms': request_ms, 'online_planning_ms': planning_ms}


def weighted_metrics(rows):
    keys = rows[0]['metrics'].keys()
    return {k: sum(row['case']['weight']*row['metrics'][k] for row in rows) for k in keys}


def quality_compatible(row, reference):
    return (row['success'] >= reference['success']-.01-1e-12 and
            all(row[k] <= reference[k]+1e-12 for k in ('duplicate', 'false_completion', 'abstention')))


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError('output exists; retain prior evidence and choose a fresh directory')
    output.mkdir(parents=True)
    started = time.monotonic()
    frozen = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
              'protocol': str(source_paths()[0]), 'horizon': HORIZON, 'time_cost': TIME_COST,
              'abstain_loss': ABSTAIN_LOSS, 'duplicate_loss': DUPLICATE_LOSS,
              'costs': COSTS, 'templates': TEMPLATES,
              'regimes': {r: [asdict(c) for c in cases(r)] for r in ('nominal', 'shifted')},
              'sources': {str(p): digest(p) for p in source_paths()},
              'measurement': 'exhaustive weighted controlled HTTP/SQLite cases; zero LLM/API requests'}
    write_json(output/'frozen.json', frozen)
    for p in source_paths():
        (output/p.name).write_bytes(p.read_bytes())
    trace = gzip.open(output/'trajectories.jsonl.gz', 'wt', encoding='utf-8')
    summary, selections = {}, {}
    errors, total, requests, online, preplanning = [], 0, [], [], {}
    planner_models = {}
    policies_by_condition = {}
    try:
        # Fixed rules are selected with exact assumed-model expectations, not service test outcomes.
        for regime in ('nominal', 'shifted'):
            for journal_mode in ('atomic', 'lagged', 'unavailable'):
                for cost_profile in COSTS:
                    key = f'{regime}/{journal_mode}/{cost_profile}'
                    model = DiagnosticModel(regime, journal_mode, cost_profile)
                    t0 = time.perf_counter()
                    target = expected_metrics(model, 'optimal')
                    selected = select_fixed(model, target)
                    preplanning[key] = time.perf_counter()-t0
                    selected.update(optimal=target, exact_root_value=model.optimal(model.initial())[0])
                    selections[key] = selected
                    planner_models[key] = model
                    rules = {'fixed-loss': FixedRule(**selected['loss_selected']['rule'])}
                    if selected['quality_selected']:
                        rules['fixed-quality'] = FixedRule(**selected['quality_selected']['rule'])
                    policies_by_condition[key] = {**{x: x for x in ('optimal', 'myopic', 'deadline', 'rollout', 'posterior')}, **rules}
                    print(json.dumps({'stage': 'preplanning', 'condition': key, 'seconds': preplanning[key]}, ensure_ascii=False), flush=True)
        write_json(output/'selections-before-service.json', selections)
        write_json(output/'preplanning-seconds.json', preplanning)
        with LocalStateService(output/'service.sqlite') as service:
            for regime in ('nominal', 'shifted'):
                for journal_mode in ('atomic', 'lagged', 'unavailable'):
                    for cost_profile in COSTS:
                        key = f'{regime}/{journal_mode}/{cost_profile}'
                        matched = planner_models[key]
                        nominal = planner_models[f'nominal/{journal_mode}/{cost_profile}']
                        policies = dict(policies_by_condition[key])
                        policies['idempotent-retry'] = 'idempotent-retry'
                        if regime == 'shifted':
                            policies['nominal-on-shift'] = 'optimal'
                        for template in TEMPLATES:
                            condition = f'{key}/{template}'
                            summary[condition] = {}
                            for name, policy in policies.items():
                                rows = []
                                model = nominal if name == 'nominal-on-shift' else matched
                                for case_index, case in enumerate(cases(regime)):
                                    row = actual_episode(service, model, policy, case, template, name == 'idempotent-retry')
                                    row.update(condition=condition, policy=name, case_index=case_index)
                                    rows.append(row)
                                    trace.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
                                    total += 1
                                    requests.extend(row['http_request_ms'])
                                    online.extend(row['online_planning_ms'])
                                    if name not in ('nominal-on-shift', 'idempotent-retry'):
                                        # Independent service must agree with the policy's symbolic prediction.
                                        from .diagnosis import model_episode
                                        predicted = model_episode(model, policy, case)
                                        for metric in ('success', 'duplicate', 'false_completion', 'abstention', 'cost', 'ticks', 'loss'):
                                            if abs(predicted[metric]-row['metrics'][metric]) > 1e-9:
                                                errors.append({'condition': condition, 'policy': name, 'case': case_index,
                                                               'metric': metric, 'predicted': predicted[metric], 'actual': row['metrics'][metric]})
                                        predicted_obs = [(x['action'], x['observation']) for x in predicted['trajectory']]
                                        actual_obs = [(x['action'], x['observation']) for x in row['trajectory'] if x['action'] not in ('retry', 'repair')]
                                        if predicted_obs != actual_obs or predicted['terminal_action'] != row['terminal_action']:
                                            errors.append({'condition': condition, 'policy': name, 'case': case_index, 'trajectory_mismatch': True})
                                summary[condition][name] = weighted_metrics(rows)
                            trace.flush()
                            print(json.dumps({'stage': 'service', 'condition': condition, 'episodes': total,
                                              'elapsed_s': time.monotonic()-started}, ensure_ascii=False), flush=True)
    finally:
        trace.close()
    aggregate = {}
    envelopes = {}
    for regime in ('nominal', 'shifted'):
        conditions = {k: v for k, v in summary.items() if k.startswith(regime+'/')}
        optimal = {metric: statistics.mean(v['optimal'][metric] for v in conditions.values()) for metric in next(iter(conditions.values()))['optimal']}
        chosen = {}
        for key, results in conditions.items():
            eligible = [n for n in BASELINES if n in results and quality_compatible(results[n], results['optimal'])]
            if not eligible:
                raise RuntimeError(f'no quality-compatible baseline in {key}')
            name = min(eligible, key=lambda n: (results[n]['cost'], n))
            chosen[key] = {'policy': name, 'metrics': results[name]}
        envelopes.update(chosen)
        baseline = {metric: statistics.mean(v['metrics'][metric] for v in chosen.values()) for metric in optimal}
        saved = 1 - optimal['cost']/baseline['cost']
        zero_risk = all(abs(optimal[m]) < 1e-12 for m in ('duplicate', 'false_completion', 'abstention')) and abs(optimal['success']-1)<1e-12
        aggregate[regime] = {'optimal': optimal, 'quality_baseline_envelope': baseline,
                             'cost_saved_fraction': saved, 'practical_gate': saved >= .1-1e-12,
                             'zero_risk_scalar_cost_equivalence': zero_risk}
        aggregate[regime]['quality_gate'] = all(
            results['optimal']['success'] >= chosen[key]['metrics']['success']-.01-1e-12 and
            all(results['optimal'][m] <= chosen[key]['metrics'][m]+1e-12
                for m in ('duplicate', 'false_completion', 'abstention'))
            for key, results in conditions.items())
    with sqlite3.connect(output/'service.sqlite') as con:
        counts = {table: con.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] for table in
                  ('episodes', 'records', 'cache_entries', 'operation_journal', 'scheduled_events', 'event_log')}
        remaining = con.execute('SELECT COUNT(*) FROM scheduled_events WHERE done=0').fetchone()[0]
        writer_pids = [r[0] for r in con.execute('SELECT DISTINCT writer_pid FROM event_log')]
    write_json(output/'summary.json', {'conditions': summary, 'aggregate': aggregate, 'envelopes': envelopes})
    verification = {'episodes': total, 'model_service_disagreements': errors, 'remaining_events': remaining,
                    'db_table_counts': counts, 'service_writer_pids': writer_pids,
                    'positive_probability_atoms_per_condition': 25,
                    'exhaustive_not_iid_samples': True, 'api_requests': 0, 'llm_requests': 0,
                    'http_action_requests': len(requests), 'elapsed_s': time.monotonic()-started,
                    'http_request_ms_p50': statistics.median(requests),
                    'http_request_ms_p95': sorted(requests)[int(.95*(len(requests)-1))],
                    'online_planning_ms_p50': statistics.median(online),
                    'online_planning_ms_p95': sorted(online)[int(.95*(len(online)-1))],
                    'ended_at_utc': datetime.now(timezone.utc).isoformat()}
    write_json(output/'verification.json', verification)
    write_json(output/'decision.json', {'continue_to_model_learning': not errors and all(v['practical_gate'] and v['quality_gate'] for v in aggregate.values()),
                                      'global_zero_risk_cost_bound_valid': all(v['zero_risk_scalar_cost_equivalence'] for v in aggregate.values()),
                                      'aggregate': aggregate,
                                      'scope': 'This finite controlled service only; not a general impossibility or learned-agent claim.'})
    if errors or remaining:
        raise RuntimeError(f'service/model verification failed: {len(errors)} disagreements, {remaining} pending events')
    with (output/'service.sqlite').open('rb') as src, gzip.open(output/'service.sqlite.gz', 'wb') as dst:
        shutil.copyfileobj(src, dst)
    write_json(output/'manifest.json', {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size}
                                       for p in output.iterdir() if p.is_file() and p.name != 'manifest.json' and not p.name.startswith('service.sqlite-')})
    print(json.dumps({'completed': True, 'output': str(output), 'verification': verification, 'aggregate': aggregate}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == '__main__':
    main()
