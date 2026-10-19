"""Synthetic mathematical fixtures; none are natural-agent results."""
import math
import unittest

try:
    from agent_pilots import recovery as r
except ImportError:
    r = None


class RecoveryModelTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(r, "the recovery POMDP module is not implemented")
        self.p = r.Parameters(not_executed=0.4, stale_read=0.25,
                              delay=((0, 1.0),), retry_success=1.0)

    def test_bayes_negative_check_has_hand_computed_posterior(self):
        # P(not done | negative) = .4 / (.4 + .6*.25).
        b, likelihood = r.update_belief(r.initial_belief(self.p), "check", "absent", self.p)
        self.assertAlmostEqual(likelihood, 0.55)
        self.assertAlmostEqual(b[(0, 0)], 0.4 / 0.55)
        self.assertAlmostEqual(sum(b.values()), 1.0)

    def test_positive_check_does_not_reveal_execution_multiplicity(self):
        b, likelihood = r.update_belief({(1, 0): .5, (2, 0): .5}, "check", "present", self.p)
        self.assertEqual(b, {(1, 0): .5, (2, 0): .5})
        self.assertAlmostEqual(likelihood, .75)

    def test_enumerated_action_probabilities_conserve_mass(self):
        p = r.Parameters(delay=((0, .2), (1, .3), (3, .5)))
        for count in range(3):
            for delay in range(4):
                for action in r.NONTERMINAL:
                    kernel = r.transition((count, delay), action, p)
                    self.assertTrue(all(prob >= 0 for _, _, prob in kernel))
                    self.assertAlmostEqual(sum(prob for _, _, prob in kernel), 1.0)

    def test_wait_and_check_cannot_execute_operation(self):
        for action in ("wait", "check"):
            self.assertTrue(all(s[0] == 0 for s, _, _ in r.transition((0, 0), action, self.p)))

    def test_retry_of_executed_operation_causes_irreversible_duplicate(self):
        outcomes = r.transition((1, 0), "retry", self.p)
        self.assertEqual(outcomes, [((2, 0), "timeout", 1.0)])
        self.assertTrue(all(s[0] == 2 for s, _, _ in r.transition((2, 0), "retry", self.p)))

    def test_delay_advances_before_check_and_stale_read_is_false_negative_only(self):
        p = r.Parameters(stale_read=0, delay=((1, 1.0),))
        self.assertEqual(r.transition((1, 1), "check", p), [((1, 0), "present", 1.0)])
        self.assertEqual(r.transition((0, 0), "check", p), [((0, 0), "absent", 1.0)])

    def test_terminal_scoring_distinguishes_missing_duplicate_and_abstention(self):
        self.assertEqual(r.outcome((1, 0), "accept")["success"], 1)
        self.assertEqual(r.outcome((0, 0), "accept")["false_completion"], 1)
        self.assertEqual(r.outcome((2, 0), "abandon")["duplicate"], 1)
        self.assertEqual(r.outcome((2, 0), "abandon")["abstention"], 1)

    def test_zero_horizon_planner_uses_belief_not_hidden_state(self):
        planner = r.BeliefPlanner(self.p, lookahead=3)
        self.assertEqual(planner.choose({(1, 0): 1.0}, 0), "accept")
        self.assertEqual(planner.choose({(0, 0): 1.0}, 0), "abandon")

    def test_exact_policy_tree_matches_manual_one_step_diagnosis(self):
        p = r.Parameters(not_executed=.4, stale_read=0, delay=((0, 1.0),),
                         check_cost=1, time_cost=0, abandon_loss=8, missing_loss=40)
        planner = r.BeliefPlanner(p, lookahead=1)
        # Check costs 1, then accept if present or abandon if absent: 1+.4*8=4.2.
        value, action = planner.value(r.initial_belief(p), 1)
        self.assertAlmostEqual(value, 4.2)
        self.assertEqual(action, "check")

    def test_paired_seed_replays_identical_trajectory(self):
        p = r.Parameters(delay=((0, .5), (3, .5)))
        policy = r.Rule(wait=1, negatives=2, retries=1)
        first = r.simulate(p, policy, seed=12, horizon=7)
        self.assertEqual(first, r.simulate(p, policy, seed=12, horizon=7))
        self.assertAlmostEqual(first["cost"], sum(x["cost"] for x in first["trajectory"]))

    def test_strong_rule_grid_can_accept_or_abstain_without_diagnostic_cost(self):
        for terminal in ("accept", "abandon"):
            rule = r.Rule(initial_terminal=terminal)
            record = r.simulate(self.p, rule, seed=12)
            self.assertEqual(record["terminal_action"], terminal)
            self.assertEqual(record["cost"], 0)

    def test_deployable_rule_receives_observations_without_hidden_state(self):
        case = self
        class AuditedRule:
            def choose(self, history, remaining):
                for item in history:
                    case.assertEqual(set(item), {"action", "observation"})
                return "check" if remaining else "abandon"
        r.simulate(self.p, AuditedRule(), seed=99, horizon=3)

    def test_oracle_expected_value_is_lower_bound_on_loss(self):
        planner = r.BeliefPlanner(self.p, lookahead=2)
        value, _ = planner.value(r.initial_belief(self.p), 2)
        oracle = sum(prob * min(r.terminal_loss(s, a, self.p) for a in r.TERMINAL)
                     for s, prob in r.initial_belief(self.p).items())
        self.assertGreaterEqual(value, oracle - 1e-10)


if __name__ == "__main__":
    unittest.main()
