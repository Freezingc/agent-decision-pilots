"""Four actually generated code workflows, with task-grouped dev/test gates.

Bits mean execute (1) / skip (0) repair and final review respectively.
No reference answers or hidden tests are imported during generation.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import itertools
import json
from pathlib import Path
import time

from .llm import LocalRunner, digest, read_jsonl, write_json

MASKS = ('00', '10', '01', '11')
STAGES = ('draft', 'repair', 'review_base', 'review_repair')
LEAVES = {'00': 'draft', '10': 'repair', '01': 'review_base', '11': 'review_repair'}
PATHS = {'00': ('draft',), '10': ('draft', 'repair'),
         '01': ('draft', 'review_base'), '11': ('draft', 'repair', 'review_repair')}
SYSTEM = 'You are a careful Python programmer. Return only the complete implementation in one Python code block. Preserve the required function name and interface. Include necessary imports. Do not include tests or explanations.'


def freeze_tasks(rows, count=96):
    eligible = [row for row in rows if row['split'] == 'heldout']
    if len({r['task_id'] for r in eligible}) != len(eligible) or len(eligible) < count:
        raise ValueError('insufficient or duplicated heldout tasks')
    ordered = sorted(eligible, key=lambda r: digest('p3-v0|' + r['task_id']))[:count]
    return [{**r, 'pilot_split': 'dev' if i < count // 2 else 'test'} for i, r in enumerate(ordered)]


def branch_cost(mask, costs):
    return sum(costs[stage] for stage in PATHS[mask])


def choose_mask(features, rule):
    # Restrict policy inputs explicitly; hidden outcome fields are never read.
    short = features['draft_output_tokens'] <= rule.get('length_cutoff', 10**9)
    passed = bool(features['visible_pass']) and short
    return rule['pass_mask'] if passed else rule['fail_mask']


def interaction(outcomes):
    return outcomes['00'] - outcomes['10'] - outcomes['01'] + outcomes['11']


def visible_tests(prompt, task_id):
    tree = ast.parse(prompt)
    doc = ast.get_docstring(tree) or ''
    result = []
    for line in doc.splitlines():
        if line.strip().startswith('assert '):
            assertion = line.strip()
            if not isinstance(ast.parse(assertion).body[0], ast.Assert):
                raise ValueError('invalid public assertion')
            result.append({'test_id': f'{task_id}/public/{len(result)}', 'assertion': assertion})
    if not result:
        raise ValueError('no public assertions: ' + task_id)
    return result


def prepare(args):
    selected = freeze_tasks(read_jsonl(args.views), args.count)
    for row in selected:
        row['public_tests'] = visible_tests(row['prompt'], row['task_id'])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'tasks.json', selected)
    write_json(out / 'frozen.json', {'protocol': 'pilot-v0', 'count': args.count, 'seeds': [0, 1],
               'source_public_views': str(Path(args.views).resolve()),
               'source_sha256': hashlib.sha256(Path(args.views).read_bytes()).hexdigest(),
               'tasks_hash': digest(selected), 'system_hash': digest(SYSTEM),
               'workflow_source_hash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'model': 'Qwen/Qwen2.5-Coder-7B-Instruct',
               'model_revision': 'c03e6d358207e414f1eca0bb1891e29f1db0e242',
               'temperature': 0.7, 'top_p': 0.95, 'max_tokens': 384,
               'notes': 'Frozen before generation. Labels remain outside generation process.'})
    print(json.dumps({'prepared': len(selected), 'public_tests': sum(len(r['public_tests']) for r in selected), 'hash': digest(selected)}))


def instance_id(task_id, seed):
    return f'{task_id}/s{seed}'


def stage_seed(iid, stage):
    # Same random stream for final review under its two parent interventions.
    # Other stages retain the predeclared stream, so draft/repair caches remain valid.
    rng_stage = 'review' if stage.startswith('review_') else stage
    return int(digest(f'p3-v0|{iid}|{rng_stage}')[:15], 16)


def generation(args):
    from .execution import execute_candidate, extract_source
    out = Path(args.output)
    tasks = json.loads((out / 'tasks.json').read_text())
    frozen = json.loads((out / 'frozen.json').read_text())
    if digest(tasks) != frozen['tasks_hash']:
        raise ValueError('tasks changed after freeze')
    runner = LocalRunner(temperature=0.7, batch_size=args.batch_size)
    receipts = {}
    public = {}
    public_path = out / 'public-feedback.jsonl'
    public_cache = {row['id']: row for row in read_jsonl(public_path)}
    total_start = time.monotonic()
    for stage in STAGES:
        requests = []
        for task in tasks:
            for seed in (0, 1):
                iid = instance_id(task['task_id'], seed)
                if stage == 'draft':
                    user = 'Implement the following task.\n\n' + task['prompt']
                else:
                    parent = 'draft' if stage in ('repair', 'review_base') else 'repair'
                    previous = receipts[f'{iid}/{parent}']
                    code = extract_source(previous['response'])['source']
                    if stage == 'repair':
                        feedback = public[f'{iid}/{parent}']['results']
                        instruction = ('Repair the current implementation using the public-example execution feedback. '
                                       'If it is already correct, retain its behavior. Return the full implementation.\n'
                                       'Public execution feedback:\n' + json.dumps(feedback))
                    else:
                        instruction = ('Review the current implementation for logical mistakes, generalization beyond the examples, '
                                       'and boundary cases implied by the task. Make necessary corrections. '
                                       'If no correction is needed, return the same implementation. Return the full implementation.')
                    user = task['prompt'] + '\nCurrent implementation:\n```python\n' + code + '\n```\n' + instruction
                requests.append({'id': f'{iid}/{stage}', 'system': SYSTEM, 'user': user, 'max_tokens': 384,
                                 'seed': stage_seed(iid, stage)})
        request_path = out / f'requests-{stage}.jsonl'
        request_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in requests))
        receipts.update(runner.run(requests, out / f'responses-{stage}.jsonl'))
        # Public feedback is a legitimate tool observation; it never uses hidden labels.
        if stage in ('draft', 'repair'):
            with public_path.open('a') as stream:
                for task in tasks:
                    for seed in (0, 1):
                        rid = f'{instance_id(task["task_id"], seed)}/{stage}'
                        code = extract_source(receipts[rid]['response'])
                        if rid in public_cache:
                            row = public_cache[rid]
                            if row['source_hash'] != code['source_hash']:
                                raise ValueError('public feedback cache mismatch')
                        else:
                            row = {'id': rid, 'source_hash': code['source_hash'],
                                   **execute_candidate(code['source'], task['public_tests'], per_test_timeout=1)}
                            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                            stream.flush()
                            public_cache[rid] = row
                        public[rid] = row
        print(json.dumps({'event': 'stage_complete', 'stage': stage, 'elapsed': time.monotonic() - total_start}), flush=True)
    write_json(out / 'generation-metadata.json', {**runner.backend.metadata, 'wall_seconds': time.monotonic() - total_start,
                                                  'actual_requests': len(receipts), 'hidden_labels_available': False})


def evaluation(args):
    # Importing the official data/evaluator is deliberately isolated from inference.
    import os
    os.environ['MBPP_OVERRIDE_PATH'] = args.dataset
    from evalplus.data.mbpp import get_mbpp_plus
    from .evaluation import evaluate_candidate, reference_suites
    from .execution import extract_source
    out = Path(args.output)
    tasks = json.loads((out / 'tasks.json').read_text())
    problems = get_mbpp_plus()
    receipts = {r['id']: r for stage in STAGES for r in read_jsonl(out / f'responses-{stage}.jsonl')}
    if len(receipts) != len(tasks) * 2 * 4:
        raise ValueError('incomplete generation; hidden evaluation prohibited')
    cache_path = out / 'evaluation-cache.jsonl'
    cache = {r['key']: r for r in read_jsonl(cache_path)}
    suites = {t['task_id']: reference_suites(problems[t['task_id']]) for t in tasks}
    jobs = {}
    for task in tasks:
        tid = task['task_id']
        for seed in (0, 1):
            for stage in STAGES:
                parsed = extract_source(receipts[f'{instance_id(tid, seed)}/{stage}']['response'])
                key = tid + '/' + parsed['source_hash']
                if key not in cache:
                    jobs[key] = (tid, parsed)
    with ThreadPoolExecutor(max_workers=args.workers) as pool, cache_path.open('a') as stream:
        futures = {pool.submit(evaluate_candidate, parsed['source'], problems[tid]['entry_point'], suites[tid],
                               problems[tid]['atol']): (key, parsed) for key, (tid, parsed) in jobs.items()}
        for index, future in enumerate(as_completed(futures), 1):
            key, parsed = futures[future]
            result = future.result()
            # Do not duplicate source strings in the publishable evaluation receipt.
            result.pop('solution')
            row = {'key': key, 'source_status': parsed['status'], **result}
            cache[key] = row
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            if index % 24 == 0 or index == len(jobs):
                print(json.dumps({'evaluated': index, 'total': len(jobs)}), flush=True)
    public = {r['id']: r for r in read_jsonl(out / 'public-feedback.jsonl')}
    rows = []
    for task in tasks:
        tid = task['task_id']
        for seed in (0, 1):
            iid = instance_id(tid, seed)
            draft = receipts[iid + '/draft']
            row = {'id': iid, 'task_id': tid, 'split': task['pilot_split'], 'seed': seed,
                   'draft_output_tokens': draft['output_tokens'],
                   'visible_pass': all(x['status'] == 'pass' for x in public[iid + '/draft']['results']),
                   'branches': {}}
            costs = {stage: receipts[iid + '/' + stage]['input_tokens'] + receipts[iid + '/' + stage]['output_tokens'] for stage in STAGES}
            output_costs = {stage: receipts[iid + '/' + stage]['output_tokens'] for stage in STAGES}
            for mask in MASKS:
                receipt = receipts[iid + '/' + LEAVES[mask]]
                parsed = extract_source(receipt['response'])
                label = cache[tid + '/' + parsed['source_hash']]
                row['branches'][mask] = {'pass': label['base_status'] == label['plus_status'] == 'pass',
                                         'base_status': label['base_status'], 'plus_status': label['plus_status'],
                                         'tokens': branch_cost(mask, costs), 'output_tokens': branch_cost(mask, output_costs),
                                         'finish_reason': receipt['finish_reason'], 'source_hash': parsed['source_hash']}
            rows.append(row)
    (out / 'outcomes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print(json.dumps({'outcomes': len(rows), 'unique_evaluations': len(cache)}))


def policy_metrics(rows, decisions):
    n = len(rows)
    outcomes = [r['branches'][mask] for r, mask in zip(rows, decisions, strict=True)]
    return {'instances': n, 'successes': sum(x['pass'] for x in outcomes),
            'success_rate': sum(x['pass'] for x in outcomes) / n,
            'mean_tokens': sum(x['tokens'] for x in outcomes) / n,
            'mean_output_tokens': sum(x['output_tokens'] for x in outcomes) / n,
            'mask_counts': {m: decisions.count(m) for m in MASKS}}


def select_gate_on_dev(dev, allowed=MASKS):
    full = policy_metrics(dev, ['11'] * len(dev))
    candidates = []
    for passed, failed, cutoff in itertools.product(allowed, allowed, (64, 128, 256, 10**9)):
        rule = {'pass_mask': passed, 'fail_mask': failed, 'length_cutoff': cutoff}
        metrics = policy_metrics(dev, [choose_mask(r, rule) for r in dev])
        if metrics['successes'] >= full['successes']:
            candidates.append((metrics['mean_tokens'], -metrics['successes'], str(rule), rule))
    return min(candidates)[-1]


def analysis(args):
    import numpy as np
    out = Path(args.output)
    rows = read_jsonl(out / 'outcomes.jsonl')
    dev = [r for r in rows if r['split'] == 'dev']
    test = [r for r in rows if r['split'] == 'test']
    # Predeclared utility sweep displays tradeoffs; it does not certify a risk cap.
    # Operational rule chosen only on development, under no drop in dev success.
    full_dev = policy_metrics(dev, ['11'] * len(dev))
    chosen = select_gate_on_dev(dev)
    single_gate = select_gate_on_dev(dev, ('10', '01', '11'))
    results = {'gate_selected_only_on_dev': chosen, 'dev_full': full_dev, 'dev_gate': policy_metrics(dev, [choose_mask(r, chosen) for r in dev]),
               'test_fixed': {m: policy_metrics(test, [m] * len(test)) for m in MASKS},
               'test_gate': policy_metrics(test, [choose_mask(r, chosen) for r in test])}
    results['single_skip_gate_selected_only_on_dev'] = single_gate
    results['test_single_skip_gate'] = policy_metrics(test, [choose_mask(r, single_gate) for r in test])
    visible_only_gate = select_gate_on_dev(dev, MASKS)
    # A gate with no length feature is a necessary simpler policy family.
    visible_candidates = []
    for passed, failed in itertools.product(MASKS, MASKS):
        rule = {'pass_mask': passed, 'fail_mask': failed, 'length_cutoff': 10**9}
        metrics = policy_metrics(dev, [choose_mask(r, rule) for r in dev])
        if metrics['successes'] >= full_dev['successes']:
            visible_candidates.append((metrics['mean_tokens'], -metrics['successes'], str(rule), rule))
    visible_only_gate = min(visible_candidates)[-1]
    results['visible_only_gate_selected_only_on_dev'] = visible_only_gate
    results['test_visible_only_gate'] = policy_metrics(test, [choose_mask(r, visible_only_gate) for r in test])
    fixed_choice = min((policy_metrics(dev, [m] * len(dev))['mean_tokens'], m)
                       for m in MASKS if policy_metrics(dev, [m] * len(dev))['successes'] >= full_dev['successes'])[1]
    results['dev_selected_fixed_mask'] = fixed_choice
    results['test_dev_selected_fixed'] = policy_metrics(test, [fixed_choice] * len(test))
    # Oracle is a descriptive upper bound: cheapest successful path where any
    # exists, otherwise initial draft. Its cost is unattainable without labels.
    for name, allowed in [('all_four', MASKS), ('at_most_one_skip', ('10', '01', '11'))]:
        oracle = [min((m for m in allowed if r['branches'][m]['pass']),
                      key=lambda m: (r['branches'][m]['tokens'], m), default=min(allowed, key=lambda m: r['branches'][m]['tokens'])) for r in test]
        results['oracle_' + name] = policy_metrics(test, oracle)
        # A second upper bound preserves full-workflow success per instance,
        # so extra success and cost-saving opportunities are not conflated.
        preserve = [min((m for m in allowed if r['branches'][m]['pass'] or not r['branches']['11']['pass']),
                        key=lambda m: (r['branches'][m]['tokens'], m)) for r in test]
        results['oracle_preserve_full_' + name] = policy_metrics(test, preserve)
    results['test_nonzero_interaction'] = sum(interaction({m: int(r['branches'][m]['pass']) for m in MASKS}) != 0 for r in test)
    results['test_full_correct_but_double_skip_wrong'] = sum(r['branches']['11']['pass'] and not r['branches']['00']['pass'] for r in test)
    results['test_full_wrong_but_double_skip_correct'] = sum(not r['branches']['11']['pass'] and r['branches']['00']['pass'] for r in test)
    # Paired task-cluster bootstrap, preserving both seeds of each task.
    groups = sorted({r['task_id'] for r in test})
    success_delta, cost_saving = [], []
    for tid in groups:
        group = [r for r in test if r['task_id'] == tid]
        gate = policy_metrics(group, [choose_mask(r, chosen) for r in group])
        full = policy_metrics(group, ['11'] * len(group))
        success_delta.append(gate['success_rate'] - full['success_rate'])
        cost_saving.append(full['mean_tokens'] - gate['mean_tokens'])
    rng = np.random.default_rng(20261019)
    indices = rng.integers(0, len(groups), (5000, len(groups)))
    results['gate_minus_full_success_cluster_ci95'] = np.quantile(np.array(success_delta)[indices].mean(axis=1), [.025, .975]).tolist()
    results['full_minus_gate_tokens_cluster_ci95'] = np.quantile(np.array(cost_saving)[indices].mean(axis=1), [.025, .975]).tolist()
    results['all_requests_total_input_tokens'] = sum(r['input_tokens'] for s in STAGES for r in read_jsonl(out / f'responses-{s}.jsonl'))
    results['all_requests_total_output_tokens'] = sum(r['output_tokens'] for s in STAGES for r in read_jsonl(out / f'responses-{s}.jsonl'))
    results['limitations'] = ['exploratory; 48 test tasks; no rare-risk guarantee', 'single 7B code model and short MBPP+ tasks',
                              'oracle uses hidden labels and is not a deployable policy', 'token totals include input and output; wall latency is not token count']
    write_json(out / 'summary.json', results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'generate', 'evaluate', 'analyze'])
    parser.add_argument('--output', default='artifacts/p3')
    parser.add_argument('--views', default='/home/zhangyudong/FDU/paper/Eval/.local/dataset/public/views.jsonl')
    parser.add_argument('--dataset', default='/home/zhangyudong/FDU/paper/Eval/.local/dataset/private/mbppplus.jsonl')
    parser.add_argument('--count', type=int, default=96)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    {'prepare': prepare, 'generate': generation, 'evaluate': evaluation, 'analyze': analysis}[args.command](args)


if __name__ == '__main__':
    main()
