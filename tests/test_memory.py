import importlib.util
import json
import pytest


def module():
    assert importlib.util.find_spec('agent_pilots.memory') is not None, 'P2 memory adapter/analyzer missing'
    from agent_pilots import memory
    return memory


def fixture_records():
    def sess(i, text, **extra):
        return {'Session_ID': i, 'Date': f'2020-01-{i+1:02d}',
                'Session_Dialogue': {'dialogue_turn_1': [{'role': 'user', 'content': text},
                                                         {'role': 'assistant', 'content': 'okay'}]}, **extra}
    def cond(item, condition, role):
        return {'Conflict_ID': 'CC_1', 'Role': role, 'Item': item, 'Condition': condition}
    return [{'ID': 'profile', 'Full_Session_Chain': [
        sess(0, 'I live in Oldtown.'),
        sess(1, 'I moved to Newtown.', Updated_Attributes=[{'Attribute': 'Residence', 'Before': 'Oldtown', 'After': 'Newtown'}]),
        sess(2, 'At work I like coffee.', Conditional_Conflict_Information=[cond('coffee', 'at work', 'Point_A')]),
        sess(3, 'At bedtime I like tea.', Conditional_Conflict_Information=[cond('tea', 'at bedtime', 'Point_B')]),
    ]}]


def test_adapter_uses_actual_quotes_and_rejects_unanchored_metadata():
    m = module()
    records = fixture_records()
    rows = m.adapt_records(records, pairs_per_label=1)
    assert len(rows) == 2
    assert {x['gold_label'] for x in rows} == {'supersede', 'coexist'}
    assert all('assistant' not in x['old_quote'] for x in rows)
    records[0]['Full_Session_Chain'][1]['Updated_Attributes'][0]['After'] = 'Never mentioned'
    assert m.adapt_records(records, pairs_per_label=1) == []


def test_group_split_is_order_invariant_and_has_no_profile_overlap():
    m = module()
    rows = [{'id': str(i), 'group_id': str(i // 2)} for i in range(12)]
    a = m.assign_splits(rows, dev_groups=2)
    b = m.assign_splits(list(reversed(rows)), dev_groups=2)
    assert {r['id']: r['split'] for r in a} == {r['id']: r['split'] for r in b}
    assert len({r['group_id'] for r in a if r['split'] == 'dev'}) == 2
    assert not ({r['group_id'] for r in a if r['split'] == 'dev'} & {r['group_id'] for r in a if r['split'] == 'test'})


def test_requests_do_not_contain_gold_or_source_type():
    m = module()
    cases = m.assign_splits(m.adapt_records(fixture_records(), pairs_per_label=1), dev_groups=0)
    for row, request in zip(cases, m.make_requests(cases)):
        assert set(request) == {'id', 'system', 'user', 'max_tokens'}
        assert row['old_quote'] in request['user'] and row['new_quote'] in request['user']
        assert 'gold_label' not in request['user'] and 'conditional_conflict' not in request['user']
        assert row['group_id'] not in request['user']


@pytest.mark.parametrize('response,expected,valid', [
    ('{"label":"supersede","confidence":0.8}', ('supersede', .8), True),
    ('```json\n{"label":"coexist","confidence":1}\n```', ('coexist', 1.), True),
    ('{"label":"supersede","confidence":2}', ('uncertain', 0.), False),
    ('{"label":"supersede","confidence":true}', ('uncertain', 0.), False),
    ('not JSON', ('uncertain', 0.), False),
])
def test_prediction_parser_fail_closed(response, expected, valid):
    p = module().parse_prediction(response)
    assert (p['label'], p['confidence']) == expected
    assert p['valid'] is valid


def test_counts_keep_abstention_in_missed_update_denominator():
    m = module()
    rows = [{'gold_label': x} for x in ['supersede', 'supersede', 'coexist', 'coexist']]
    result = m.score(rows, ['uncertain', 'supersede', 'supersede', 'coexist'])
    assert result['n'] == 4 and result['covered'] == 3
    assert result['false_overwrite'] == 1 and result['missed_update'] == 1
    assert result['true_updates'] == 2 and result['non_updates'] == 2
    assert result['coverage'] == .75 and result['missed_update_rate'] == .5


def test_threshold_selection_reads_dev_only():
    m = module()
    rows = [{'id': 'd1', 'split': 'dev', 'gold_label': 'coexist'},
            {'id': 'd2', 'split': 'dev', 'gold_label': 'supersede'},
            {'id': 't1', 'split': 'test', 'gold_label': 'supersede'}]
    pred = {x['id']: {'label': 'supersede', 'confidence': c, 'valid': True}
            for x, c in zip(rows, [.7, .9, .2])}
    a = m.choose_threshold(rows, pred)
    rows[2]['gold_label'] = 'coexist'
    pred['t1']['confidence'] = 1.
    assert m.choose_threshold(rows, pred) == a
    assert .7 < a <= .9


def test_analysis_requires_exact_frozen_response_ids(tmp_path):
    m = module()
    labels = [{'id': 'd', 'group_id': 'a', 'split': 'dev', 'gold_label': 'coexist'},
              {'id': 't', 'group_id': 'b', 'split': 'test', 'gold_label': 'supersede'}]
    responses = [{'id': 'd', 'response': '{"label":"coexist","confidence":1}', 'input_tokens': 10, 'output_tokens': 4}]
    with pytest.raises(ValueError, match='IDs'):
        m.analyze_rows(labels, responses)
    responses += [dict(responses[0], id='t')]
    out = m.analyze_rows(labels, responses)
    assert out['cost']['input_tokens'] == 20 and out['cost']['output_tokens'] == 8
    assert out['splits']['test']['latest_wins']['missed_update'] == 0
    assert out['splits']['test']['direct_model']['missed_update'] == 1
    with pytest.raises(ValueError, match='duplicate'):
        m.analyze_rows(labels, responses + [responses[0]])


def test_analysis_rejects_profile_leakage_and_bad_token_accounting():
    m = module()
    labels = [{'id': 'd', 'group_id': 'a', 'split': 'dev', 'gold_label': 'coexist'},
              {'id': 't', 'group_id': 'a', 'split': 'test', 'gold_label': 'supersede'}]
    responses = [{'id': x['id'], 'response': '{}', 'input_tokens': 1, 'output_tokens': 1} for x in labels]
    with pytest.raises(ValueError, match='group'):
        m.analyze_rows(labels, responses)
    labels[1]['group_id'] = 'b'
    responses[0]['input_tokens'] = -1
    with pytest.raises(ValueError, match='tokens'):
        m.analyze_rows(labels, responses)


def test_scalar_anchor_must_state_residence_not_birthplace():
    m = module()
    records = fixture_records()
    records[0]['Full_Session_Chain'][0]['Session_Dialogue']['dialogue_turn_1'][0]['content'] = 'I was born in Oldtown.'
    assert m.adapt_records(records, pairs_per_label=1) == []


def test_conditional_anchor_must_be_first_person_with_visible_condition():
    m = module()
    records = fixture_records()
    records[0]['Full_Session_Chain'][2]['Session_Dialogue']['dialogue_turn_1'][0]['content'] = 'She likes coffee at work.'
    assert m.adapt_records(records, pairs_per_label=1) == []
    records[0]['Full_Session_Chain'][2]['Session_Dialogue']['dialogue_turn_1'][0]['content'] = 'I like coffee.'
    assert m.adapt_records(records, pairs_per_label=1) == []


def test_analyze_rejects_quarantined_adaptation_before_reading_responses(tmp_path):
    m = module()
    (tmp_path / 'quarantine.json').write_text('{"reason":"source anchoring failed"}')
    with pytest.raises(ValueError, match='quarantined'):
        m.analyze(tmp_path, tmp_path / 'never-generated-responses.jsonl')


def test_analyze_rejects_changed_frozen_labels(tmp_path):
    m = module()
    (tmp_path / 'labels.jsonl').write_text('{}\n')
    (tmp_path / 'manifest.json').write_text(json.dumps({'files': {'labels': {'path': 'labels.jsonl', 'sha256': 'bad'}}}))
    with pytest.raises(ValueError, match='hash'):
        m.analyze(tmp_path, tmp_path / 'not-needed.jsonl')


def test_zero_model_rule_reads_new_visible_text_and_ignores_birthplace():
    m = module()
    request = {'id': 'x', 'user': 'Earlier statement (2020-01-01):\nI like coffee.\n\nLater statement (2020-01-02):\nI was born in Rome.'}
    assert m.lexical_prediction(request, 'new_residence') == 'coexist'
    request['user'] = request['user'].replace('I was born in Rome.', 'I just moved to Rome.')
    assert m.lexical_prediction(request, 'new_residence') == 'supersede'
    assert m.lexical_prediction(request, 'both_residence') == 'coexist'


def test_zero_model_rule_is_selected_using_dev_only():
    m = module()
    labels = [{'id': 'd', 'split': 'dev', 'gold_label': 'supersede'},
              {'id': 't', 'split': 'test', 'gold_label': 'coexist'}]
    requests = [{'id': x['id'], 'user': 'Earlier statement (2020):\nI like coffee.\n\nLater statement (2021):\nI moved to Rome.'} for x in labels]
    rule = m.choose_lexical_rule(labels, requests)
    assert rule == 'new_residence'
    labels[1]['gold_label'] = 'supersede'
    requests[1]['user'] = 'Malformed but test-only request'
    assert m.choose_lexical_rule(labels, requests) == rule


def test_optional_inference_receipt_is_bound_to_exact_request_and_config():
    from hashlib import sha256
    m = module()
    request = {'id': 'x', 'system': 'classify', 'user': 'visible evidence', 'max_tokens': 96}
    hash_value = sha256(json.dumps({**request, 'temperature': 0., 'top_p': .95}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    response = {'id': 'x', 'request_hash': hash_value, 'temperature': 0., 'max_tokens': 96}
    assert m.verify_response_receipts([request], [response], top_p=.95)['verified'] == 1
    request['user'] = 'changed evidence'
    with pytest.raises(ValueError, match='request/config'):
        m.verify_response_receipts([request], [response], top_p=.95)
    assert m.verify_response_receipts([request], [{'id': 'x'}], top_p=.95)['verified'] == 0
