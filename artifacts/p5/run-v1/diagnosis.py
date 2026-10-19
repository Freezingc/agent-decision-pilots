"""P5 finite, known-model diagnostic upper bound; not a learning algorithm.

Beliefs contain only posterior support of deterministic latent cases. Exact case
weights and delays are deliberately supplied as an oracle. Controllers consume
canonical PUBLIC tool responses; only the experiment runner accesses true cases.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import itertools
import math

HORIZON = 10
ABSTAIN_LOSS = 50.
DUPLICATE_LOSS = 200.
TIME_COST = .2
COSTS = {
    'status-cheap': {'cache': .5, 'journal': 1., 'fields': 4., 'wait': 0., 'retry': 1., 'repair': 4.},
    'primary-cheap': {'cache': .25, 'journal': 3., 'fields': 2., 'wait': 0., 'retry': 1., 'repair': 1.5},
}


@dataclass(frozen=True)
class Case:
    outcome: str
    apply_delay: int
    cache_lag: int
    journal_lag: int
    weight: float


def cases(regime):
    if regime == 'nominal':
        delays = {'full': ((0, .5), (2, .35), (5, .15)), 'partial': ((0, .1), (2, .3), (5, .6))}
        cache, journal = ((0, .4), (3, .6)), ((0, .6), (2, .4))
    elif regime == 'shifted':
        delays = {'full': ((0, .2), (3, .3), (9, .5)), 'partial': ((0, .05), (3, .2), (9, .75))}
        cache, journal = ((1, .25), (5, .75)), ((0, .3), (4, .7))
    else:
        raise ValueError(regime)
    result = [Case('none', 0, 0, 0, .2)]
    for outcome, prior in (('full', .65), ('partial', .15)):
        for (delay, pd), (cl, pc), (jl, pj) in itertools.product(delays[outcome], cache, journal):
            result.append(Case(outcome, delay, cl, jl, prior * pd * pc * pj))
    assert math.isclose(sum(c.weight for c in result), 1.)
    return tuple(result)


@dataclass(frozen=True)
class Belief:
    tick: int
    mask: int
    known_record: bool = False
    certified: bool = False


@dataclass(frozen=True)
class FixedRule:
    order: tuple
    initial_wait: int = 0
    interval: int = 1
    exponential: bool = False
    repair_on_presence: bool = True

    def choose(self, model, belief, history):
        best = model.best_terminal(belief)
        if best[1] in ('accept', 'retry') or belief.tick >= HORIZON:
            return best[1]
        if belief.known_record:
            if self.repair_on_presence or history[-1]['observation'] == 'partial':
                return best[1]
            return 'fields'
        checks = [h for h in history if h['action'] != 'wait']
        if not checks:
            return 'wait' if belief.tick < self.initial_wait else self.order[0]
        last_tick = checks[-1]['tick']
        gap = self.interval * (2 ** (len(checks)-1) if self.exponential else 1)
        if belief.tick - last_tick < gap - 1:
            return 'wait'
        return self.order[len(checks) % len(self.order)]


class DiagnosticModel:
    def __init__(self, regime, journal_mode, cost_profile):
        self.regime, self.journal_mode, self.cost_profile = regime, journal_mode, cost_profile
        self.cases = cases(regime)
        self.costs = COSTS[cost_profile]
        if journal_mode not in ('atomic', 'lagged', 'unavailable'):
            raise ValueError(journal_mode)
        self._optimal = lru_cache(maxsize=None)(self._solve)
        self._deadline = lru_cache(maxsize=None)(self._deadline_cost)

    def initial(self):
        return Belief(0, (1 << len(self.cases)) - 1)

    def indices(self, belief):
        return tuple(i for i in range(len(self.cases)) if belief.mask & (1 << i))

    def posterior(self, belief):
        indices = self.indices(belief)
        total = sum(self.cases[i].weight for i in indices)
        return tuple((i, self.cases[i].weight / total) for i in indices)

    def cost(self, action):
        return 0. if action in ('accept', 'abstain') else self.costs[action] + TIME_COST

    def observation(self, case, action, tick):
        if action == 'wait':
            return 'waited'
        exists = case.outcome != 'none' and tick >= case.apply_delay
        if action == 'fields':
            return ('complete' if case.outcome == 'full' else 'partial') if exists else 'absent'
        if action == 'cache':
            return 'present' if exists and tick >= case.apply_delay + case.cache_lag else 'absent'
        if action == 'journal':
            if self.journal_mode == 'unavailable':
                return 'unavailable'
            if self.journal_mode == 'atomic':
                if case.outcome == 'none':
                    return 'not_found'
                return ('complete' if case.outcome == 'full' else 'partial') if exists else 'pending'
            visible = exists and tick >= case.apply_delay + case.journal_lag
            return ('complete' if case.outcome == 'full' else 'partial') if visible else 'unknown'
        raise ValueError(action)

    def update(self, belief, action, observation):
        tick = belief.tick + 1
        mask = sum(1 << i for i in self.indices(belief)
                   if self.observation(self.cases[i], action, tick) == observation)
        if not mask:
            raise ValueError(f'impossible public history: tick={tick}, action={action}, observation={observation}')
        return Belief(tick, mask, belief.known_record or observation in ('present', 'complete', 'partial'),
                      belief.certified or observation == 'complete')

    def branches(self, belief, action):
        buckets = {}
        for i, p in self.posterior(belief):
            obs = self.observation(self.cases[i], action, belief.tick+1)
            buckets[obs] = buckets.get(obs, 0.) + p
        return tuple((obs, p, self.update(belief, action, obs)) for obs, p in sorted(buckets.items()))

    def not_accepted_probability(self, belief):
        return sum(p for i, p in self.posterior(belief) if self.cases[i].outcome == 'none')

    def terminals(self, belief):
        options = [(ABSTAIN_LOSS, 'abstain')]
        if belief.certified:
            options.append((0., 'accept'))
        if belief.tick < HORIZON:
            if belief.known_record:
                options.append((self.cost('repair'), 'repair'))
            pnone = self.not_accepted_probability(belief)
            if not belief.known_record and pnone >= .99 - 1e-12:
                options.append((self.cost('retry') + DUPLICATE_LOSS * (1-pnone), 'retry'))
        return options

    def best_terminal(self, belief):
        return min(self.terminals(belief), key=lambda x: (round(x[0], 10), x[1]))

    def checks(self, belief):
        if belief.tick >= HORIZON:
            return ()
        return ('cache', 'fields', 'wait') + (() if self.journal_mode == 'unavailable' else ('journal',))

    def _solve(self, belief):
        candidates = list(self.terminals(belief))
        if not belief.certified:
            for action in self.checks(belief):
                value = self.cost(action) + sum(p * self._optimal(child)[0] for _, p, child in self.branches(belief, action))
                candidates.append((value, action))
        return min(candidates, key=lambda x: (round(x[0], 10), x[1]))

    def optimal(self, belief):
        return self._optimal(belief)

    def deadline_action(self, belief):
        best = self.best_terminal(belief)
        if best[1] != 'abstain' or not self.checks(belief):
            return best[1]
        latest = max((self.cases[i].apply_delay for i in self.indices(belief)
                      if self.cases[i].outcome != 'none'), default=0)
        return 'fields' if belief.tick + 1 >= latest else 'wait'

    def _deadline_cost(self, belief):
        action = self.deadline_action(belief)
        if action in ('accept', 'abstain', 'retry', 'repair'):
            return dict((a, v) for v, a in self.terminals(belief))[action]
        return self.cost(action) + sum(p*self._deadline(child) for _, p, child in self.branches(belief, action))

    def deadline_value(self, belief):
        return self._deadline(belief)

    def choose(self, policy, belief, history):
        if isinstance(policy, FixedRule):
            return policy.choose(self, belief, history)
        if policy == 'optimal':
            return self.optimal(belief)[1]
        if policy == 'deadline':
            return self.deadline_action(belief)
        candidates = list(self.terminals(belief))
        if policy in ('myopic', 'rollout'):
            for action in self.checks(belief):
                next_value = self.best_terminal if policy == 'myopic' else lambda b: (self.deadline_value(b), '')
                candidates.append((self.cost(action) + sum(p*next_value(child)[0] for _, p, child in self.branches(belief, action)), action))
            return min(candidates, key=lambda x: (round(x[0], 10), x[1]))[1]
        if policy == 'posterior':
            best = self.best_terminal(belief)[1]
            if best != 'abstain' or not self.checks(belief):
                return best
            checks = sorted((a for a in self.checks(belief) if a != 'wait'), key=lambda a: (self.cost(a), a))
            # Rotate cheap checks so an unavailable negative channel cannot monopolize the policy.
            return checks[len(history) % len(checks)]
        raise ValueError(policy)

    @staticmethod
    def canonical(action, response, expected_fields):
        if action == 'fields':
            rows = response['records']
            if not rows:
                return 'absent'
            return 'complete' if len(rows) == 1 and all(rows[0]['fields'].get(k) == v for k, v in expected_fields.items()) else 'partial'
        return response['state']


def fixed_rules(journal_mode):
    actions = ('cache', 'fields') + (() if journal_mode == 'unavailable' else ('journal',))
    for length in range(1, len(actions)+1):
        for order in itertools.permutations(actions, length):
            for initial_wait, interval, exponential, repair in itertools.product((0, 1, 2, 4, 8), (1, 2, 4), (False, True), (False, True)):
                yield FixedRule(order, initial_wait, interval, exponential, repair)


def model_episode(model, policy, case):
    belief, history, cost = model.initial(), [], 0.
    for _ in range(HORIZON+1):
        action = model.choose(policy, belief, history)
        if action in ('accept', 'abstain', 'retry', 'repair'):
            cost += model.cost(action)
            duplicate = action == 'retry' and case.outcome != 'none'
            success = ((action == 'accept' and case.outcome == 'full') or
                       (action == 'repair' and case.outcome != 'none') or
                       (action == 'retry' and case.outcome == 'none'))
            declared = action != 'abstain'
            return {'success': int(success), 'duplicate': int(duplicate),
                    'false_completion': int(declared and not success), 'abstention': int(not declared),
                    'cost': cost, 'ticks': belief.tick + int(action in ('retry', 'repair')),
                    'loss': cost + ABSTAIN_LOSS*int(not declared) + DUPLICATE_LOSS*int(duplicate),
                    'terminal_action': action, 'trajectory': history}
        observation = model.observation(case, action, belief.tick+1)
        cost += model.cost(action)
        belief = model.update(belief, action, observation)
        history.append({'tick': belief.tick, 'action': action, 'observation': observation})
    raise RuntimeError('policy exceeded horizon')


def expected_metrics(model, policy):
    rows = [model_episode(model, policy, case) for case in model.cases]
    keys = ('success', 'duplicate', 'false_completion', 'abstention', 'cost', 'loss', 'ticks')
    metrics = {key: sum(c.weight*r[key] for c, r in zip(model.cases, rows, strict=True)) for key in keys}
    metrics['max_ticks'] = max(r['ticks'] for r in rows)
    return metrics


def select_fixed(model, target):
    best_loss = best_quality = None
    count = 0
    for rule in fixed_rules(model.journal_mode):
        metrics = expected_metrics(model, rule)
        row = {'rule': asdict(rule), 'metrics': metrics}
        key = (metrics['loss'], metrics['cost'], repr(rule))
        if best_loss is None or key < best_loss[0]:
            best_loss = key, row
        quality = (metrics['success'] >= target['success']-.01-1e-12 and
                   metrics['duplicate'] <= target['duplicate']+1e-12 and
                   metrics['false_completion'] <= target['false_completion']+1e-12 and
                   metrics['abstention'] <= target['abstention']+1e-12)
        if quality:
            qkey = (metrics['cost'], metrics['loss'], repr(rule))
            if best_quality is None or qkey < best_quality[0]:
                best_quality = qkey, row
        count += 1
    return {'count': count, 'loss_selected': best_loss[1],
            'quality_selected': best_quality[1] if best_quality else None}
