import unittest

from agent_pilots.workflow import branch_cost, choose_mask, freeze_tasks, interaction, stage_seed


class WorkflowTests(unittest.TestCase):
    def test_logical_cost_uses_only_executed_path(self):
        costs = {'draft': 10, 'repair': 20, 'review_base': 30, 'review_repair': 40}
        self.assertEqual(branch_cost('00', costs), 10)
        self.assertEqual(branch_cost('10', costs), 30)
        self.assertEqual(branch_cost('01', costs), 40)
        self.assertEqual(branch_cost('11', costs), 70)
        with self.assertRaises(KeyError):
            branch_cost('11', {'draft': 10, 'repair': 20})

    def test_task_split_is_grouped_and_excludes_old_explored_tasks(self):
        rows = [{'task_id': f'Mbpp/{i}', 'split': 'heldout', 'prompt': 'p',
                 'entry_point': 'f'} for i in range(120)]
        rows.append({'task_id': 'excluded', 'split': 'e1'})
        selected = freeze_tasks(rows, 96)
        self.assertEqual(len(selected), 96)
        self.assertEqual(sum(x['pilot_split'] == 'test' for x in selected), 48)
        self.assertEqual(selected, freeze_tasks(list(reversed(rows)), 96))
        self.assertNotIn('excluded', [x['task_id'] for x in selected])
        self.assertEqual(len({x['task_id'] for x in selected}), 96)

    def test_visible_gate_ignores_hidden_labels(self):
        gate = {'pass_mask': '00', 'fail_mask': '11', 'length_cutoff': 1000}
        row = {'visible_pass': True, 'draft_output_tokens': 70, 'hidden_pass': False}
        self.assertEqual(choose_mask(row, gate), '00')
        row['hidden_pass'] = True
        self.assertEqual(choose_mask(row, gate), '00')
        row['visible_pass'] = False
        self.assertEqual(choose_mask(row, gate), '11')

    def test_nonadditive_interaction(self):
        self.assertEqual(interaction({'00': 1, '10': 1, '01': 1, '11': 0}), -1)
        self.assertEqual(interaction({'00': 1, '10': 1, '01': 1, '11': 1}), 0)

    def test_final_review_uses_common_random_numbers_across_parents(self):
        self.assertEqual(stage_seed('Mbpp/1/s0', 'review_base'),
                         stage_seed('Mbpp/1/s0', 'review_repair'))
        self.assertNotEqual(stage_seed('Mbpp/1/s0', 'draft'), stage_seed('Mbpp/1/s1', 'draft'))


if __name__ == '__main__':
    unittest.main()
