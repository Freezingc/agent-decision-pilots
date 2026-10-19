"""Independent, read-only P5 audit. Uses only Python's standard library.

No project modules are imported. The finite model, controller values, event
queue, database state, costs, baseline selection and decision are recomputed.
Run only after manifest.json and decision.json exist in the completed run.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import hashlib
import heapq
import itertools
import json
import math
from pathlib import Path
import re
import sqlite3
import sys

HORIZON = 10
TIME = .2
ABSTAIN = 50.
DUPLICATE = 200.
COSTS = {
    'status-cheap': dict(cache=.5, journal=1., fields=4., wait=0., retry=1., repair=4.),
    'primary-cheap': dict(cache=.25, journal=3., fields=2., wait=0., retry=1., repair=1.5),
}
TEMPLATES = {
    'reservation': {'guest': 'pilot-user', 'date': '2030-01-12', 'room': 'A12'},
    'invoice': {'customer': 'pilot-customer', 'amount_cents': 2500, 'currency': 'CNY'},
    'shipment': {'recipient': 'pilot-user', 'address': 'test-address', 'parcel_count': 2},
}
RISK_KEYS = ('duplicate', 'false_completion', 'abstention')
METRICS = ('success', 'duplicate', 'false_completion', 'abstention', 'cost', 'ticks', 'loss', 'model_miss')
BASELINES = ('myopic', 'deadline', 'rollout', 'posterior', 'fixed-loss', 'fixed-quality')
UUID = re.compile(r'^[0-9a-f]{32}$')


def atoms(regime):
    if regime == 'nominal':
        delay = {'full': ((0, .50), (2, .35), (5, .15)), 'partial': ((0, .10), (2, .30), (5, .60))}
        cl, jl = ((0, .40), (3, .60)), ((0, .60), (2, .40))
    elif regime == 'shifted':
        delay = {'full': ((0, .20), (3, .30), (9, .50)), 'partial': ((0, .05), (3, .20), (9, .75))}
        cl, jl = ((1, .25), (5, .75)), ((0, .30), (4, .70))
    else:
        raise ValueError(regime)
    result = [dict(outcome='none', apply_delay=0, cache_lag=0, journal_lag=0, weight=.20)]
    for outcome, prior in (('full', .65), ('partial', .15)):
        for (d, pd), (c, pc), (j, pj) in itertools.product(delay[outcome], cl, jl):
            result.append(dict(outcome=outcome, apply_delay=d, cache_lag=c, journal_lag=j, weight=prior*pd*pc*pj))
    return result


def abstract_observation(case, mode, action, tick):
    exists = case['outcome'] != 'none' and case['apply_delay'] <= tick
    state = 'complete' if case['outcome'] == 'full' else 'partial'
    if action == 'wait':
        return 'waited'
    if action == 'fields':
        return state if exists else 'absent'
    if action == 'cache':
        return 'present' if exists and case['apply_delay']+case['cache_lag'] <= tick else 'absent'
    if action == 'journal':
        if mode == 'unavailable':
            return 'unavailable'
        if mode == 'atomic':
            return 'not_found' if case['outcome'] == 'none' else (state if exists else 'pending')
        return state if exists and case['apply_delay']+case['journal_lag'] <= tick else 'unknown'
    raise ValueError(action)


def complete(fields, expected):
    return all(k in fields and fields[k] == v for k, v in expected.items())


def canonical(action, response, expected):
    if action != 'fields':
        return response['state']
    records = response['records']
    if not records:
        return 'absent'
    return 'complete' if len(records) == 1 and complete(records[0]['fields'], expected) else 'partial'


def equivalent(a, b, tolerance=1e-9):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tolerance
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(equivalent(a[k], b[k], tolerance) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(equivalent(x, y, tolerance) for x, y in zip(a, b))
    return a == b


class Audit:
    def __init__(self):
        self.failures = []
        self.checks = 0

    def check(self, predicate, label, **context):
        self.checks += 1
        if not predicate:
            def safe(value):
                if isinstance(value, dict):
                    return {str(k): safe(v) for k, v in value.items()}
                if isinstance(value, (tuple, list, set)):
                    return [safe(v) for v in value]
                if isinstance(value, float) and not math.isfinite(value):
                    return repr(value)
                return value
            self.failures.append(safe(dict(check=label, **context)))

    def equal(self, actual, expected, label, **context):
        self.check(equivalent(actual, expected), label, actual=actual, expected=expected, **context)


class IndependentModel:
    """Finite exact expectation using a support tuple, not runner belief classes."""
    def __init__(self, regime, mode, profile):
        self.regime, self.mode, self.profile = regime, mode, profile
        self.cases = atoms(regime)
        self.initial = (0, tuple(range(25)), False, False)
        self.value = lru_cache(None)(self._value)
        self.deadline_value = lru_cache(None)(self._deadline_value)

    def fee(self, action):
        return 0. if action in ('accept', 'abstain') else COSTS[self.profile][action]+TIME

    def posterior(self, belief):
        total = sum(self.cases[i]['weight'] for i in belief[1])
        return [(i, self.cases[i]['weight']/total) for i in belief[1]]

    def terminals(self, belief):
        tick, support, known, certified = belief
        options = [(ABSTAIN, 'abstain')]
        if certified:
            options.append((0., 'accept'))
        if tick < HORIZON:
            if known:
                options.append((self.fee('repair'), 'repair'))
            pnone = sum(p for i, p in self.posterior(belief) if self.cases[i]['outcome'] == 'none')
            if not known and pnone >= .99-1e-12:
                options.append((self.fee('retry')+DUPLICATE*(1-pnone), 'retry'))
        return options

    @staticmethod
    def minimum(options):
        return min(options, key=lambda item: (round(item[0], 10), item[1]))

    def terminal(self, belief):
        return self.minimum(self.terminals(belief))

    def checks(self, belief):
        return () if belief[0] >= HORIZON else ('cache', 'fields', 'wait') + (() if self.mode == 'unavailable' else ('journal',))

    def update(self, belief, action, obs):
        tick, support, known, certified = belief
        support = tuple(i for i in support if abstract_observation(self.cases[i], self.mode, action, tick+1) == obs)
        if not support:
            return None
        return tick+1, support, known or obs in ('present', 'partial', 'complete'), certified or obs == 'complete'

    def branches(self, belief, action):
        masses = defaultdict(float)
        for i, p in self.posterior(belief):
            masses[abstract_observation(self.cases[i], self.mode, action, belief[0]+1)] += p
        return [(p, self.update(belief, action, obs)) for obs, p in sorted(masses.items())]

    def _value(self, belief):
        options = self.terminals(belief)
        if not belief[3]:
            options += [(self.fee(a)+sum(p*self.value(child)[0] for p, child in self.branches(belief, a)), a)
                        for a in self.checks(belief)]
        return self.minimum(options)

    def deadline(self, belief):
        terminal = self.terminal(belief)[1]
        if terminal != 'abstain' or not self.checks(belief):
            return terminal
        latest = max((self.cases[i]['apply_delay'] for i in belief[1] if self.cases[i]['outcome'] != 'none'), default=0)
        return 'fields' if belief[0]+1 >= latest else 'wait'

    def _deadline_value(self, belief):
        action = self.deadline(belief)
        if action in ('accept', 'abstain', 'repair', 'retry'):
            return next(v for v, a in self.terminals(belief) if a == action)
        return self.fee(action)+sum(p*self.deadline_value(child) for p, child in self.branches(belief, action))

    def choose(self, name, belief, history, rule=None):
        if belief is None:
            return 'abstain'
        if rule is not None:
            best = self.terminal(belief)[1]
            if best in ('accept', 'retry') or belief[0] >= HORIZON:
                return best
            if belief[2]:
                return best if rule['repair_on_presence'] or history[-1][2] == 'partial' else 'fields'
            checks = [h for h in history if h[1] != 'wait']
            if not checks:
                return 'wait' if belief[0] < rule['initial_wait'] else rule['order'][0]
            interval = rule['interval'] * (2**(len(checks)-1) if rule['exponential'] else 1)
            if belief[0]-checks[-1][0] < interval-1:
                return 'wait'
            return rule['order'][len(checks) % len(rule['order'])]
        if name in ('optimal', 'nominal-on-shift'):
            return self.value(belief)[1]
        if name == 'deadline':
            return self.deadline(belief)
        if name in ('myopic', 'rollout'):
            options = self.terminals(belief)
            child_value = (lambda b: self.terminal(b)[0]) if name == 'myopic' else self.deadline_value
            options += [(self.fee(a)+sum(p*child_value(child) for p, child in self.branches(belief, a)), a)
                        for a in self.checks(belief)]
            return self.minimum(options)[1]
        if name == 'posterior':
            best = self.terminal(belief)[1]
            if best != 'abstain' or not self.checks(belief):
                return best
            actions = sorted((a for a in self.checks(belief) if a != 'wait'), key=lambda a: (self.fee(a), a))
            return actions[len(history) % len(actions)]
        if name == 'idempotent-retry':
            return 'retry'
        raise ValueError(name)

    def episode(self, name, true_case, rule=None):
        belief, history, cost = self.initial, [], 0.
        for tick in range(HORIZON+1):
            action = self.choose(name, belief, history, rule)
            if action in ('accept', 'abstain', 'repair', 'retry'):
                cost += self.fee(action)
                duplicate = action == 'retry' and true_case['outcome'] != 'none'
                success = (action == 'accept' and true_case['outcome'] == 'full') or (action == 'repair' and true_case['outcome'] != 'none') or (action == 'retry' and true_case['outcome'] == 'none')
                return dict(success=int(success), duplicate=int(duplicate), false_completion=int(action != 'abstain' and not success),
                            abstention=int(action == 'abstain'), cost=cost, ticks=tick+int(action in ('repair', 'retry')),
                            loss=cost+ABSTAIN*int(action == 'abstain')+DUPLICATE*int(duplicate))
            obs = abstract_observation(true_case, self.mode, action, tick+1)
            cost += self.fee(action)
            history.append((tick+1, action, obs))
            belief = self.update(belief, action, obs)
        raise AssertionError('independent policy exceeded horizon')

    def expected(self, name, rule=None):
        rows = [self.episode(name, c, rule) for c in self.cases]
        result = {k: sum(c['weight']*r[k] for c, r in zip(self.cases, rows)) for k in rows[0]}
        result['max_ticks'] = max(r['ticks'] for r in rows)
        return result


def rules(mode):
    actions = ('cache', 'fields') + (() if mode == 'unavailable' else ('journal',))
    for length in range(1, len(actions)+1):
        for order in itertools.permutations(actions, length):
            for initial_wait, interval, exponential, repair in itertools.product((0, 1, 2, 4, 8), (1, 2, 4), (False, True), (False, True)):
                yield dict(order=list(order), initial_wait=initial_wait, interval=interval, exponential=exponential, repair_on_presence=repair)


def compatible(row, reference):
    return row['success'] >= reference['success']-.01-1e-12 and all(row[k] <= reference[k]+1e-12 for k in RISK_KEYS)


class StorageReplay:
    """Replay protocol from scratch, taking only opaque new IDs from DB writes."""
    def __init__(self, config, operation_id, event_rows):
        self.config, self.op = config, operation_id
        self.original_ids = [e['record_id'] for e in event_rows if e['kind'] == 'apply_original']
        self.retry_ids = [e['record_id'] for e in event_rows if e['kind'] == 'retry']
        self.records, self.observed = {}, set()
        self.cache = None
        self.journal = None if config['journal_mode'] != 'atomic' else ('not_found' if config['outcome'] == 'none' else 'pending', None)
        self.logs = [(0, 'start', None)]
        self.queue, self.scheduled, self.serial = [], [], 0
        self.tick = 0
        if config['outcome'] != 'none':
            self.schedule(config['apply_delay'], 'apply_original', None)
        self.due(0)

    def schedule(self, tick, kind, rid):
        self.serial += 1
        heapq.heappush(self.queue, (tick, self.serial, kind, rid))
        self.scheduled.append((tick, kind, {'record_id': rid}, 1))

    def journal_write(self, rid):
        state = 'complete' if complete(self.records[rid], self.config['expected_fields']) else 'partial'
        self.journal = state, rid

    def publish(self, tick, rid):
        self.schedule(tick+self.config['cache_lag'], 'publish_cache', rid)
        if self.config['journal_mode'] == 'atomic':
            self.journal_write(rid)
            self.logs.append((tick, 'publish_journal', rid))
        elif self.config['journal_mode'] == 'lagged':
            self.schedule(tick+self.config['journal_lag'], 'publish_journal', rid)

    def write(self, tick, kind, rid, fields):
        if self.config['idempotent'] and self.records:
            assert rid == next(iter(self.records)), 'idempotent write changed record ID'
        else:
            assert rid not in self.records and UUID.fullmatch(rid), 'new write reused/nonopaque ID'
        self.records.setdefault(rid, {}).update(fields)
        self.logs.append((tick, kind, rid))
        self.publish(tick, rid)

    def due(self, tick):
        while self.queue and self.queue[0][0] <= tick:
            when, _, kind, rid = heapq.heappop(self.queue)
            if kind == 'apply_original':
                assert len(self.original_ids) == 1, 'wrong original write count'
                fields = self.config['expected_fields']
                if self.config['outcome'] == 'partial':
                    first = next(iter(fields))
                    fields = {first: fields[first]}
                self.write(when, kind, self.original_ids[0], fields)
            elif kind == 'publish_cache':
                self.cache = rid
                self.logs.append((when, kind, rid))
            elif kind == 'publish_journal':
                self.journal_write(rid)
                self.logs.append((when, kind, rid))
            else:
                raise AssertionError(kind)

    def action(self, action, repair_id):
        self.tick += 1
        self.due(self.tick)
        self.logs.append((self.tick, 'action_'+action, None))
        if action == 'wait':
            return {'state': 'waited'}
        if action == 'cache':
            if self.cache is None:
                return {'state': 'absent'}
            self.observed.add(self.cache)
            return {'state': 'present', 'record_id': self.cache}
        if action == 'journal':
            if self.config['journal_mode'] == 'unavailable':
                return {'state': 'unavailable'}
            if self.journal is None:
                return {'state': 'unknown'}
            state, rid = self.journal
            if rid:
                self.observed.add(rid)
            return {'state': state, **({'record_id': rid} if rid else {})}
        if action == 'fields':
            self.observed.update(self.records)
            return {'records': [{'record_id': rid, 'fields': dict(fields)} for rid, fields in self.records.items()]}
        if action == 'retry':
            assert len(self.retry_ids) == 1, 'wrong retry write count'
            rid = self.retry_ids[0]
            self.write(self.tick, 'retry', rid, self.config['expected_fields'])
        elif action == 'repair':
            rid = repair_id
            assert rid in self.observed and rid in self.records, 'repair used hidden/nonexistent ID'
            self.records[rid].update(self.config['expected_fields'])
            self.logs.append((self.tick, 'repair', rid))
            self.publish(self.tick, rid)
        else:
            raise AssertionError(action)
        self.due(self.tick)
        self.observed.add(rid)
        return {'state': 'complete', 'record_id': rid}

    def drain(self):
        while self.queue:
            self.tick = max(self.tick, self.queue[0][0])
            self.due(self.tick)


def read_json(path):
    return json.loads(path.read_text())


def file_hash(path, compressed=False):
    digest = hashlib.sha256()
    opener = gzip.open if compressed else open
    with opener(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def whitebox(run, audit):
    """Narrow static checks complement a human read; not an adversarial sandbox."""
    runner = ast.parse((run/'run_diagnosis.py').read_text())
    controller = ast.parse((run/'diagnosis.py').read_text())
    service = ast.parse((run/'state_service.py').read_text())
    actual = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == 'actual_episode')
    calls = [n for n in ast.walk(actual) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'choose']
    audit.equal(len(calls), 1, 'whitebox_single_policy_call')
    if calls:
        audit.equal([ast.unparse(a) for a in calls[0].args], ['policy', 'belief', 'history'], 'whitebox_public_policy_arguments')
    choose = next(n for n in ast.walk(controller) if isinstance(n, ast.FunctionDef) and n.name == 'choose')
    audit.equal([n.arg for n in choose.args.args], ['self', 'model', 'belief', 'history'] if choose.lineno < 84 else ['self', 'policy', 'belief', 'history'], 'whitebox_policy_signature')
    methods = [n for n in ast.walk(controller) if isinstance(n, ast.FunctionDef) and n.name == 'choose']
    for method in methods:
        forbidden = sorted({n.id for n in ast.walk(method) if isinstance(n, ast.Name)} & {'case', 'config', 'audit', 'service', 'db_path'})
        audit.equal(forbidden, [], 'whitebox_no_true_case_in_policy', line=method.lineno)
    allowed_imports = {'argparse', 'dataclasses', 'datetime', 'gzip', 'hashlib', 'json', 'pathlib', 'platform', 'shutil', 'sqlite3', 'statistics', 'time',
                       'diagnosis', 'state_service', '__future__', 'functools', 'itertools', 'math', 'contextlib', 'http.server', 'multiprocessing',
                       'os', 'urllib.error', 'urllib.request', 'uuid'}
    for tree, name in ((runner, 'runner'), (controller, 'controller'), (service, 'service')):
        modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        audit.equal(sorted(modules-allowed_imports), [], 'whitebox_no_external_model_client_imports', source=name)
    text = (run/'state_service.py').read_text()
    audit.check('HTTPServer(("127.0.0.1", 0)' in text and 'http://127.0.0.1:' in text, 'whitebox_loopback_http_only')


def verify(run):
    audit = Audit()
    assert (run/'manifest.json').is_file() and (run/'decision.json').is_file(), 'run is not marked complete'
    frozen, manifest = read_json(run/'frozen.json'), read_json(run/'manifest.json')
    selections = read_json(run/'selections-before-service.json')
    for name, receipt in manifest.items():
        p = run/name
        audit.check(p.is_file(), 'manifest_file_exists', file=name)
        if p.is_file():
            audit.equal(p.stat().st_size, receipt['bytes'], 'manifest_bytes', file=name)
            audit.equal(file_hash(p), receipt['sha256'], 'manifest_sha256', file=name)
    audit.equal(sorted(p.name for p in run.iterdir() if p.is_file() and p.name != 'manifest.json' and not p.name.startswith('service.sqlite-')),
                sorted(manifest), 'manifest_covers_every_run_file')
    for path, digest in frozen['sources'].items():
        audit.equal(file_hash(run/Path(path).name), digest, 'frozen_snapshot_hash', source=path)
    audit.equal(file_hash(run/'service.sqlite.gz', compressed=True), file_hash(run/'service.sqlite'), 'compressed_database_matches')
    audit.equal(frozen['horizon'], HORIZON, 'frozen_horizon')
    audit.equal(frozen['time_cost'], TIME, 'frozen_time_cost')
    audit.equal(frozen['abstain_loss'], ABSTAIN, 'frozen_abstain_loss')
    audit.equal(frozen['duplicate_loss'], DUPLICATE, 'frozen_duplicate_loss')
    audit.equal(frozen['costs'], COSTS, 'frozen_costs')
    audit.equal(frozen['templates'], TEMPLATES, 'frozen_templates')
    audit.equal(frozen['regimes'], {r: atoms(r) for r in ('nominal', 'shifted')}, 'frozen_atoms')
    whitebox(run, audit)
    models, expected_policy_sets = {}, {}
    for regime, mode, profile in itertools.product(('nominal', 'shifted'), ('atomic', 'lagged', 'unavailable'), COSTS):
        key = '/'.join((regime, mode, profile))
        m = models[key] = IndependentModel(regime, mode, profile)
        selected = selections[key]
        target = m.expected('optimal')
        audit.equal(selected['optimal'], target, 'independent_optimal_expectation', condition=key)
        audit.equal(selected['exact_root_value'], m.value(m.initial)[0], 'independent_optimal_root', condition=key)
        minimum_loss, minimum_quality_cost, count = math.inf, math.inf, 0
        for rule in rules(mode):
            metrics = m.expected('fixed', rule)
            minimum_loss = min(minimum_loss, metrics['loss'])
            if compatible(metrics, target):
                minimum_quality_cost = min(minimum_quality_cost, metrics['cost'])
            count += 1
        audit.equal(selected['count'], count, 'full_fixed_library_count', condition=key)
        for kind, minimum, metric in (('loss_selected', minimum_loss, 'loss'), ('quality_selected', minimum_quality_cost, 'cost')):
            choice = selected[kind]
            if math.isinf(minimum):
                audit.equal(choice, None, 'no_quality_fixed_rule', condition=key)
                continue
            audit.check(choice is not None, 'fixed_rule_selected', condition=key, kind=kind)
            if choice is not None:
                predicted = m.expected('fixed', choice['rule'])
                audit.equal(choice['metrics'], predicted, 'selected_fixed_metrics', condition=key, kind=kind)
                audit.equal(predicted[metric], minimum, 'selected_fixed_global_minimum', condition=key, kind=kind)
                if kind == 'quality_selected':
                    audit.check(compatible(predicted, target), 'selected_fixed_quality', condition=key)
        policies = {'optimal', 'myopic', 'deadline', 'rollout', 'posterior', 'fixed-loss', 'idempotent-retry'}
        if selected['quality_selected'] is not None:
            policies.add('fixed-quality')
        if regime == 'shifted':
            policies.add('nominal-on-shift')
        for template in TEMPLATES:
            expected_policy_sets[key+'/'+template] = policies
    audit.equal(set(selections), set(models), 'all_preplanned_conditions')
    con = sqlite3.connect('file:'+str((run/'service.sqlite').resolve())+'?mode=ro&immutable=1', uri=True)
    con.row_factory = sqlite3.Row
    audit.equal(con.execute('PRAGMA integrity_check').fetchone()[0], 'ok', 'sqlite_integrity')
    tables = {}
    for table in ('episodes', 'records', 'cache_entries', 'operation_journal', 'observed_records', 'scheduled_events', 'event_log'):
        rows = [dict(r) for r in con.execute('SELECT * FROM '+table+' ORDER BY rowid')]
        groups = defaultdict(list)
        for row in rows:
            groups[row['episode_id']].append(row)
        tables[table] = groups
    con.close()
    audit.equal(sum(r['done'] != 1 for rs in tables['scheduled_events'].values() for r in rs), 0, 'no_pending_events')
    grouped = defaultdict(list)
    seen_ids, group_atoms, template_paths = set(), defaultdict(list), defaultdict(dict)
    row_count, action_count, model_misses, exception_count = 0, 0, 0, 0
    writer_pids = set()
    with gzip.open(run/'trajectories.jsonl.gz', 'rt', encoding='utf8') as source:
        for line_index, line in enumerate(source, 1):
            row_count += 1
            try:
                row = json.loads(line)
                regime, mode, profile, template = row['condition'].split('/')
                name, ci = row['policy'], row['case_index']
                key = '/'.join((regime, mode, profile))
                ctx = dict(line=line_index, condition=row['condition'], policy=name, atom=ci)
                audit.check(row['condition'] in expected_policy_sets and name in expected_policy_sets.get(row['condition'], set()), 'planned_condition_policy', **ctx)
                case = atoms(regime)[ci]
                audit.equal(row['case'], case, 'atom_identity_and_weight', **ctx)
                group_atoms[(row['condition'], name)].append(ci)
                audit.equal(row['template'], template, 'template_metadata', **ctx)
                audit.equal(row['journal_mode'], mode, 'journal_metadata', **ctx)
                audit.equal(row['cost_profile'], profile, 'cost_metadata', **ctx)
                model_regime = 'nominal' if name == 'nominal-on-shift' else regime
                audit.equal(row['model_regime'], model_regime, 'assumed_model_regime', **ctx)
                audit.equal(row['idempotent'], name == 'idempotent-retry', 'idempotence_metadata', **ctx)
                start = row['public_start']
                eid, op = start['episode_id'], start['operation_id']
                audit.check(eid not in seen_ids, 'unique_episode', **ctx)
                seen_ids.add(eid)
                expected_start = dict(episode_id=eid, operation_id=op, template=template, expected_fields=TEMPLATES[template],
                                      journal_mode=mode, idempotent=name == 'idempotent-retry', status='timeout')
                audit.equal(start, expected_start, 'public_start_no_hidden_fields', **ctx)
                audit.check(bool(UUID.fullmatch(eid) and UUID.fullmatch(op)), 'opaque_start_ids', **ctx)
                config = dict(template=template, expected_fields=TEMPLATES[template], outcome=case['outcome'], apply_delay=case['apply_delay'],
                              cache_lag=case['cache_lag'], journal_lag=case['journal_lag'], journal_mode=mode, idempotent=name == 'idempotent-retry')
                db_episode = tables['episodes'][eid]
                audit.equal(len(db_episode), 1, 'exactly_one_database_episode', **ctx)
                audit.equal(json.loads(db_episode[0]['config_json']), config, 'database_hidden_config_matches_atom', **ctx)
                audit.equal(db_episode[0]['operation_id'], op, 'database_operation_id', **ctx)
                event_rows = tables['event_log'][eid]
                writer_pids.update(e['writer_pid'] for e in event_rows)
                storage = StorageReplay(config, op, event_rows)
                m = models['/'.join((model_regime, mode, profile))]
                belief, history, cost, record_id, miss = m.initial, [], 0., None, False
                rule = selections[key]['loss_selected' if name == 'fixed-loss' else 'quality_selected']['rule'] if name in ('fixed-loss', 'fixed-quality') else None
                for tick, step in enumerate(row['trajectory'], 1):
                    action = step['action']
                    audit.equal(step['tick'], tick, 'sequential_tick', **ctx)
                    audit.check(tick <= HORIZON, 'action_budget', **ctx)
                    audit.equal(action, m.choose(name, belief, history, rule), 'independent_policy_action', tick=tick, **ctx)
                    expected_response = storage.action(action, record_id)
                    audit.equal(step['response'], expected_response, 'independent_storage_response', tick=tick, **ctx)
                    obs = canonical(action, expected_response, TEMPLATES[template])
                    audit.equal(step['observation'], obs, 'canonical_public_observation', tick=tick, **ctx)
                    fee = COSTS[profile][action]+TIME
                    audit.equal(step['cost'], fee, 'step_cost', tick=tick, **ctx)
                    cost += fee
                    action_count += 1
                    if action == 'fields':
                        record_id = expected_response['records'][0]['record_id'] if expected_response['records'] else record_id
                    else:
                        record_id = expected_response.get('record_id') or record_id
                    if action in ('retry', 'repair'):
                        audit.equal(tick, len(row['trajectory']), 'write_is_last_tool_action', **ctx)
                        audit.equal(row['terminal_action'], action, 'terminal_write_action', **ctx)
                        continue
                    audit.equal(obs, abstract_observation(case, mode, action, tick), 'actual_case_response_matches_independent_model', tick=tick, **ctx)
                    history.append((tick, action, obs))
                    belief = m.update(belief, action, obs)
                    miss = miss or belief is None
                terminal = row['terminal_action']
                if terminal in ('accept', 'abstain'):
                    audit.equal(terminal, m.choose(name, belief, history, rule), 'independent_terminal_action', **ctx)
                else:
                    audit.check(bool(row['trajectory']) and row['trajectory'][-1]['action'] == terminal, 'terminal_has_tool_receipt', **ctx)
                ticks = len(row['trajectory'])
                audit.equal(len(row['http_request_ms']), ticks, 'all_action_requests_accounted', **ctx)
                audit.equal(len(row['online_planning_ms']), ticks+int(terminal in ('accept', 'abstain')), 'all_online_decisions_accounted', **ctx)
                audit.check(all(x >= 0 and math.isfinite(x) for x in row['http_request_ms']+row['online_planning_ms']), 'finite_infrastructure_timings', **ctx)
                storage.drain()
                actual_logs = [(e['tick'], e['kind'], e['record_id']) for e in event_rows]
                audit.equal(actual_logs, storage.logs, 'database_events_and_logical_times', **ctx)
                scheduled = [(e['due_tick'], e['kind'], json.loads(e['payload_json']), e['done']) for e in tables['scheduled_events'][eid]]
                audit.equal(scheduled, storage.scheduled, 'all_scheduled_events_executed', **ctx)
                audit.equal(db_episode[0]['tick'], storage.tick, 'database_drain_tick', **ctx)
                db_records = tables['records'][eid]
                expected_rows = [{'record_id': rid, 'fields': fields} for rid, fields in storage.records.items()]
                actual_rows = [{'record_id': r['record_id'], 'fields': json.loads(r['fields_json'])} for r in db_records]
                audit.equal(actual_rows, expected_rows, 'database_final_fields_and_ids', **ctx)
                for r in db_records:
                    audit.equal(r['operation_id'], op, 'record_operation_id', **ctx)
                    audit.equal(r['idempotency_key'], op if config['idempotent'] else None, 'record_idempotency_key', **ctx)
                audit.equal({r['record_id'] for r in tables['observed_records'][eid]}, storage.observed, 'only_public_ids_observed', **ctx)
                cache_rows = tables['cache_entries'][eid]
                audit.equal([r['record_id'] for r in cache_rows], [] if storage.cache is None else [storage.cache], 'database_cache', **ctx)
                journal_rows = tables['operation_journal'][eid]
                audit.equal([(r['state'], r['record_id']) for r in journal_rows], [] if storage.journal is None else [storage.journal], 'database_journal', **ctx)
                count = len(storage.records)
                full = sum(complete(f, TEMPLATES[template]) for f in storage.records.values())
                business = count == full == 1
                expected_audit = dict(tick=storage.tick, record_count=count, complete_count=full, partial_count=count-full,
                                      duplicate=count > 1, business_success=business, rows=expected_rows,
                                      events=[{k: e[k] for k in ('tick', 'kind', 'record_id', 'writer_pid')} for e in event_rows])
                audit.equal(row['audit'], expected_audit, 'audit_receipt_matches_actual_database', **ctx)
                declared = terminal != 'abstain'
                metrics = dict(success=int(declared and business), duplicate=int(count > 1), false_completion=int(declared and not business),
                               abstention=int(not declared), cost=cost, ticks=ticks, loss=cost+ABSTAIN*int(not declared)+DUPLICATE*int(count > 1), model_miss=int(miss))
                audit.equal(row['metrics'], metrics, 'independent_row_metrics', **ctx)
                grouped[row['condition'], name].append((case['weight'], metrics))
                model_misses += int(miss)
                abstract_path = ([(s['action'], s['observation']) for s in row['trajectory']], terminal, metrics)
                template_paths[(regime, mode, profile, name, ci)][template] = abstract_path
            except Exception as exc:
                exception_count += 1
                audit.check(False, 'row_verification_exception', line=line_index, error=f'{type(exc).__name__}: {exc}')
    expected_groups = {(condition, name) for condition, names in expected_policy_sets.items() for name in names}
    audit.equal(set(grouped), expected_groups, 'all_planned_policy_condition_groups')
    for key in expected_groups:
        audit.equal(sorted(group_atoms[key]), list(range(25)), 'all_25_atoms_once', condition=key[0], policy=key[1])
        audit.equal(sum(w for w, _ in grouped[key]), 1., 'atom_weights_sum_one', condition=key[0], policy=key[1])
    database_ids = set(tables['episodes'])
    audit.equal(seen_ids, database_ids, 'no_missing_or_extra_database_episodes')
    for table, group in tables.items():
        audit.check(set(group) <= database_ids, 'no_orphan_database_rows', table=table)
    audit.equal(len(writer_pids), 1, 'one_service_writer_process')
    summary = read_json(run/'summary.json')
    reconstructed = defaultdict(dict)
    for (condition, name), rows in grouped.items():
        reconstructed[condition][name] = {k: sum(w*m[k] for w, m in rows) for k in METRICS}
    audit.equal(summary['conditions'], dict(reconstructed), 'independent_condition_summary')
    envelopes, aggregates = {}, {}
    scalar_by_atom = {}
    for regime in ('nominal', 'shifted'):
        conditions = {c: v for c, v in reconstructed.items() if c.startswith(regime+'/')}
        for condition, methods in conditions.items():
            eligible = [n for n in BASELINES if n in methods and compatible(methods[n], methods['optimal'])]
            audit.check(bool(eligible), 'quality_compatible_baseline_exists', condition=condition)
            if eligible:
                selected = min(eligible, key=lambda n: (methods[n]['cost'], n))
                envelopes[condition] = dict(policy=selected, metrics=methods[selected])
        optimal = {k: sum(v['optimal'][k] for v in conditions.values())/len(conditions) for k in METRICS}
        chosen = [envelopes[c] for c in conditions if c in envelopes]
        baseline = {k: sum(v['metrics'][k] for v in chosen)/len(chosen) for k in METRICS}
        saving = 1-optimal['cost']/baseline['cost']
        quality = all(compatible(methods['optimal'], envelopes[c]['metrics']) for c, methods in conditions.items() if c in envelopes)
        optimal_rows = [m for (c, name), rows in grouped.items() if c.startswith(regime+'/') and name == 'optimal' for _, m in rows]
        zero = len(optimal_rows) == 6*3*25 and all(m['success'] == 1 and all(m[k] == 0 for k in RISK_KEYS) and abs(m['loss']-m['cost']) <= 1e-12 for m in optimal_rows)
        scalar_by_atom[regime] = dict(atoms_checked=len(optimal_rows), all_success_zero_risk_loss_equals_cost=zero)
        aggregates[regime] = dict(optimal=optimal, quality_baseline_envelope=baseline, cost_saved_fraction=saving,
                                 practical_gate=saving >= .1-1e-12, quality_gate=quality, zero_risk_scalar_cost_equivalence=zero)
    audit.equal(summary['envelopes'], envelopes, 'independent_baseline_envelopes')
    audit.equal(summary['aggregate'], aggregates, 'independent_aggregate_and_gates')
    for key, paths in template_paths.items():
        audit.equal(set(paths), set(TEMPLATES), 'all_templates_present', abstract_cell=list(key))
        first = next(iter(paths.values()))
        audit.check(all(equivalent(path, first) for path in paths.values()), 'templates_same_abstract_path_and_metrics', abstract_cell=list(key))
    decision = read_json(run/'decision.json')
    audit.equal(decision['aggregate'], aggregates, 'decision_uses_recomputed_aggregates')
    audit.equal(decision['continue_to_model_learning'], all(v['practical_gate'] and v['quality_gate'] for v in aggregates.values()), 'independent_continue_gate')
    audit.equal(decision['global_zero_risk_cost_bound_valid'], all(v['zero_risk_scalar_cost_equivalence'] for v in aggregates.values()), 'independent_scalar_bound_condition')
    receipt = read_json(run/'verification.json')
    audit.equal(receipt['episodes'], row_count, 'runner_episode_accounting')
    audit.equal(receipt['http_action_requests'], action_count, 'runner_action_accounting')
    audit.equal(receipt['remaining_events'], 0, 'runner_pending_events')
    audit.equal(receipt['model_service_disagreements'], [], 'runner_no_model_disagreements')
    audit.equal(receipt['api_requests'], 0, 'external_api_claim_zero')
    audit.equal(receipt['llm_requests'], 0, 'llm_claim_zero')
    audit.equal(receipt['service_writer_pids'], sorted(writer_pids), 'runner_service_pid_accounting')
    db_counts = {table: sum(len(rs) for rs in groups.values()) for table, groups in tables.items() if table != 'observed_records'}
    audit.equal(receipt['db_table_counts'], db_counts, 'runner_database_counts')
    mismatch = {}
    for key in models:
        if key.startswith('shifted/'):
            rows = [reconstructed[key+'/'+template]['nominal-on-shift'] for template in TEMPLATES]
            mismatch[key] = {k: sum(r[k] for r in rows)/len(rows) for k in METRICS}
    return dict(audited_at_utc=datetime.now(timezone.utc).isoformat(), run=str(run.resolve()), verifier_sha256=file_hash(Path(__file__)),
                status='PASS' if not audit.failures else 'FAIL', checks=audit.checks, failure_count=len(audit.failures), failures=audit.failures,
                episodes=row_count, action_requests=action_count, row_exceptions=exception_count, model_miss_episodes=model_misses,
                scalar_condition_by_atom=scalar_by_atom, aggregate=aggregates, nominal_on_shift_conditions=mismatch,
                db_table_counts=db_counts, service_writer_pids=sorted(writer_pids),
                scope='核验代码另行实现，但使用同一系统模型及本地受控数据；不构成外部团队复现、生产或 LLM 效果证据。',
                template_interpretation='三个字段结构共用同一抽象情景，不是三个独立真实任务样本。',
                external_api_requests=0, llm_requests=0,
                whitebox_scope='冻结源码及调用／导入检查；为逻辑分离，不是对抗性认证边界。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.run_dir)
    except Exception as exc:
        result = dict(status='FAIL', error=f'{type(exc).__name__}: {exc}', audited_at_utc=datetime.now(timezone.utc).isoformat())
    serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    if args.output:
        if args.output.exists():
            raise SystemExit('refusing to replace previous audit evidence')
        args.output.write_text(serialized)
    print(serialized, end='')
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
