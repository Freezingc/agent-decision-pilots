"""Independent receipt-to-table accounting audit; never chooses a new policy."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .execution import extract_source
from .llm import digest, read_jsonl, write_json


def audit(directory):
    directory = Path(directory)
    tasks = json.loads((directory / 'tasks.json').read_text())
    frozen = json.loads((directory / 'frozen.json').read_text())
    assert digest(tasks) == frozen['tasks_hash']
    dev_ids = {t['task_id'] for t in tasks if t['pilot_split'] == 'dev'}
    test_ids = {t['task_id'] for t in tasks if t['pilot_split'] == 'test'}
    assert dev_ids.isdisjoint(test_ids)
    stages = ['draft', 'repair', 'review_base', 'review_repair']
    receipts = {}
    verified_requests = 0
    for stage in stages:
        batch = read_jsonl(directory / f'responses-{stage}.jsonl')
        assert len(batch) == len(tasks) * 2
        receipts.update({r['id']: r for r in batch})
        reqpath = directory / f'requests-{stage}.jsonl'
        if reqpath.exists():
            for q in read_jsonl(reqpath):
                r = receipts[q['id']]
                assert digest({**q, 'temperature': .7, 'top_p': .95}) == r['request_hash']
                verified_requests += 1
    evaluations = {r['key']: r for r in read_jsonl(directory / 'evaluation-cache.jsonl')}
    public = {r['id']: r for r in read_jsonl(directory / 'public-feedback.jsonl')}
    outcomes = read_jsonl(directory / 'outcomes.jsonl')
    assert len(outcomes) == len(tasks) * 2
    # Deliberately independent of workflow.PATHS/branch_cost/choose_mask.
    routes = {'00': ['draft'], '10': ['draft', 'repair'],
              '01': ['draft', 'review_base'], '11': ['draft', 'repair', 'review_repair']}
    by_task = {t['task_id']: t for t in tasks}
    for row in outcomes:
        iid = row['id']
        assert row['split'] == by_task[row['task_id']]['pilot_split']
        assert receipts[iid + '/review_base']['seed'] == receipts[iid + '/review_repair']['seed']
        assert row['visible_pass'] == all(r['status'] == 'pass' for r in public[iid + '/draft']['results'])
        for mask, route in routes.items():
            branch = row['branches'][mask]
            total = sum(receipts[iid + '/' + s]['input_tokens'] + receipts[iid + '/' + s]['output_tokens'] for s in route)
            output = sum(receipts[iid + '/' + s]['output_tokens'] for s in route)
            assert branch['tokens'] == total and branch['output_tokens'] == output
            leaf = receipts[iid + '/' + route[-1]]
            code = extract_source(leaf['response'])
            assert code['source_hash'] == branch['source_hash']
            label = evaluations[row['task_id'] + '/' + code['source_hash']]
            assert branch['pass'] == (label['base_status'] == label['plus_status'] == 'pass')
    summary = json.loads((directory / 'summary.json').read_text())
    test = [r for r in outcomes if r['split'] == 'test']
    for mask in routes:
        assert summary['test_fixed'][mask]['successes'] == sum(r['branches'][mask]['pass'] for r in test)
        assert abs(summary['test_fixed'][mask]['mean_tokens'] - sum(r['branches'][mask]['tokens'] for r in test)/len(test)) < 1e-9
    rule = summary['gate_selected_only_on_dev']
    masks = [rule['pass_mask'] if r['visible_pass'] and r['draft_output_tokens'] <= rule['length_cutoff'] else rule['fail_mask'] for r in test]
    assert summary['test_gate']['successes'] == sum(r['branches'][m]['pass'] for r,m in zip(test,masks,strict=True))
    assert abs(summary['test_gate']['mean_tokens'] - sum(r['branches'][m]['tokens'] for r,m in zip(test,masks,strict=True))/len(test)) < 1e-9
    assert summary['test_gate']['mask_counts'] == {m: Counter(masks)[m] for m in routes}
    manifest = {p.name: {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
                for p in directory.glob('*') if p.is_file() and p.name not in {'verification.json','manifest.json'}}
    write_json(directory / 'manifest.json', manifest)
    report = {'status': 'verified', 'scope': 'independent accounting, not independent scientific replication',
              'tasks': len(tasks), 'dev_tasks': len(dev_ids), 'test_tasks': len(test_ids),
              'instances': len(outcomes), 'receipts': len(receipts), 'requests_hash_checked': verified_requests,
              'unique_official_evaluations': len(evaluations), 'branches_checked': len(outcomes)*4,
              'token_paths_verified': True, 'public_features_verified': True, 'labels_verified': True,
              'table_vs_raw_verified': True, 'common_final_review_rng_verified': True}
    write_json(directory / 'verification.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='artifacts/p3')
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), indent=2))


if __name__ == '__main__':
    main()
