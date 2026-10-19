"""Independently audit stored P4 costs, beliefs, terminal rewards and replay hashes."""

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(run, replay):
    manifest = json.loads((run / 'manifest.json').read_text())
    for name, receipt in manifest['files'].items():
        assert digest(run / name) == receipt['sha256'], name
    assert digest(run / 'source.py') == manifest['source_sha256']
    compared = ('summary.json', 'design.json', 'cases.json', 'episodes.jsonl.gz',
                'dev_grid_scores.json', 'transfer_dev_selection.json')
    for name in compared:
        assert digest(run / name) == digest(replay / name), ('replay', name)
    cases = json.loads((run / 'cases.json').read_text())
    test = {case['case_id']: case for case in cases['test']}
    assert not set(test) & {case['case_id'] for case in cases['dev']}
    summary = json.loads((run / 'summary.json').read_text())
    by_condition = {row['condition']['name']: row for row in summary['results']}
    totals, counts, seen, observed = defaultdict(lambda: defaultdict(float)), defaultdict(int), set(), {}
    with gzip.open(run / 'episodes.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            key = (row['condition'], row['policy'])
            identity = (*key, row['case_id'])
            assert identity not in seen
            seen.add(identity)
            case, condition = test[row['case_id']], by_condition[row['condition']]['condition']
            belief = {k: case['context'][k] for k in ('p_env', 'p_pref')}
            cost, actions = 0.0, set()
            for step in row['trace'][:-1]:
                action = step['action']
                assert action in ('probe', 'ask') and action not in actions
                assert condition['allow_' + action]
                actions.add(action)
                assert step['belief_before'] == belief
                assert step['cost'] == condition[action + '_cost']
                cost += step['cost']
                observation = step['observation']
                paired_key = (row['condition'], row['case_id'], action)
                assert paired_key not in observed or observed[paired_key] == observation
                observed[paired_key] = observation
                if observation is not None:
                    field = 'p_env' if action == 'probe' else 'p_pref'
                    prior, q = belief[field], condition[action + '_accuracy']
                    likelihood = q if observation else 1 - q
                    belief[field] = prior * likelihood / (prior * likelihood + (1 - prior) * (1 - likelihood))
                assert all(abs(belief[k] - step['belief_after'][k]) < 1e-12 for k in belief)
            assert len(actions) <= 2
            assert row['trace'][-1]['action'] == 'choose' + str(row['decision'])
            assert abs(cost - row['cost']) < 1e-12
            env_match, pref_match = int(row['decision'] == case['env']), int(row['decision'] == case['pref'])
            reward = case['context']['w_env'] * env_match + (1 - case['context']['w_env']) * pref_match
            assert abs(reward - row['reward']) < 1e-12
            assert abs(reward - cost - row['net_utility']) < 1e-12
            assert row['environment_match'] == env_match and row['preference_match'] == pref_match
            assert row['probe_count'] == int('probe' in actions) and row['ask_count'] == int('ask' in actions)
            counts[key] += 1
            for metric in by_condition[row['condition']]['test'][row['policy']]['overall']:
                totals[key][metric] += row[metric]
    for key, count in counts.items():
        assert count == len(test)
        expected = by_condition[key[0]]['test'][key[1]]['overall']
        for metric, total in totals[key].items():
            assert abs(total / count - expected[metric]) < 1e-10, (key, metric)
    assert len(seen) == 45 * 9 * len(test)
    exact_gaps = [row['known_model_expected_utility']['planner'] - row['known_model_expected_utility']['myopic_known']
                  for row in summary['results']]
    assert all(gap >= -1e-12 for gap in exact_gaps)
    report = {'status': 'verified_simulation_only', 'episode_rows_audited': len(seen),
              'conditions': len(by_condition), 'policies': 9, 'test_cases_per_condition': len(test),
              'replay_exact_files': {name: digest(run / name) for name in compared},
              'exact_planner_myopic_equal_conditions': sum(abs(gap) < 1e-12 for gap in exact_gaps),
              'maximum_exact_planner_minus_myopic': max(exact_gaps),
              'scope': 'Stored manifest, paired observations, posterior updates, costs, rewards, aggregate means, exact replay; no independent model/theorem/LLM validation'}
    with (run.parent / 'sensitivity.csv').open('w') as stream:
        fields = ['condition', 'policy', 'net_utility', 'reward', 'cost', 'environment_match', 'preference_match',
                  'probe_count', 'ask_count', 'known_model_expected_utility']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in summary['results']:
            for policy, values in result['test'].items():
                writer.writerow({'condition': result['condition']['name'], 'policy': policy, **values['overall'],
                                 'known_model_expected_utility': result['known_model_expected_utility'][policy]})
    (run.parent / 'verification.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path('artifacts/p4/run-v1-supplement'))
    parser.add_argument('--replay', type=Path, default=Path('artifacts/p4/replay-v1-supplement'))
    args = parser.parse_args()
    validate(args.run, args.replay)
