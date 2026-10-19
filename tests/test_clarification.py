"""Behavioral checks for the P4 structural simulator, not evidence about LLMs."""

from dataclasses import replace
import importlib

import unittest


def example(p4, *, env=0, pref=1, p_env=0.5, p_pref=0.5, w_env=0.5):
    return p4.Case("test/example", "test", 123,
                   p4.Context(p_env, p_pref, w_env), env, pref)


class ClarificationTests(unittest.TestCase):
    def setUp(self):
        try:
            self.p4 = importlib.import_module("agent_pilots.clarification")
        except ModuleNotFoundError as exc:
            self.fail(f"P4 simulator is missing: {exc}")

    def test_environment_probe_is_invariant_to_hidden_user_preference(self):
        p4 = self.p4
        case = example(p4, env=0, pref=0)
        condition = p4.Condition(probe_accuracy=1.0)
        assert p4.observe(case, condition, "probe") == 0
        assert p4.observe(replace(case, pref=1), condition, "probe") == 0


    def test_user_answer_is_invariant_to_hidden_environment(self):
        p4 = self.p4
        case = example(p4, env=0, pref=1)
        condition = p4.Condition(ask_accuracy=1.0)
        assert p4.observe(case, condition, "ask") == 1
        assert p4.observe(replace(case, env=1), condition, "ask") == 1


    def test_bayes_update_changes_only_the_observed_information_source(self):
        p4 = self.p4
        condition = p4.Condition(probe_accuracy=0.8, ask_accuracy=0.8)
        belief = p4.Belief(0.5, 0.7)
        assert p4.update(belief, "probe", 1, condition) == p4.Belief(0.8, 0.7)
        assert p4.update(belief, "ask", None, condition) == belief


    def test_missing_answers_consume_cost_without_disclosing_hidden_state(self):
        p4 = self.p4
        condition = p4.Condition(probe_cost=0.1, ask_cost=0.2,
                                 probe_missing=1.0, ask_missing=1.0)
        result = p4.run_episode(example(p4), condition, "probe_then_ask")
        assert [step["action"] for step in result["trace"]] == ["probe", "ask", "choose0"]
        self.assertAlmostEqual(result["cost"], 0.3)
        assert result["reward"] == 0.5
        self.assertAlmostEqual(result["net_utility"], 0.2)
        assert result["trace"][1]["belief_after"] == {"p_env": 0.5, "p_pref": 0.5}


    def test_planner_stops_when_information_cost_exceeds_all_possible_gain(self):
        p4 = self.p4
        condition = p4.Condition(probe_cost=2.0, ask_cost=2.0)
        result = p4.run_episode(example(p4), condition, "planner")
        assert len(result["trace"]) == 1
        assert result["cost"] == 0


    def test_planner_asks_when_only_preference_can_affect_reward(self):
        p4 = self.p4
        condition = p4.Condition(ask_accuracy=1.0, ask_cost=0.1)
        result = p4.run_episode(example(p4, pref=1, w_env=0.0), condition, "planner")
        assert [step["action"] for step in result["trace"]] == ["ask", "choose1"]
        self.assertAlmostEqual(result["net_utility"], 0.9)


    def test_planner_omits_user_question_after_environment_resolves_choice(self):
        p4 = self.p4
        condition = p4.Condition(probe_accuracy=1.0, probe_cost=0.05,
                                 ask_accuracy=1.0, ask_cost=0.1)
        result = p4.run_episode(example(p4, env=1, pref=0, w_env=0.8), condition, "planner")
        assert [step["action"] for step in result["trace"]] == ["probe", "choose1"]
        assert result["reward"] == 0.8
        assert result["cost"] == 0.05


    def test_two_step_value_can_include_complementary_information(self):
        p4 = self.p4
        # Individually neither observation changes the preferred action. Two
        # negative observations reverse it. Exact enumeration gives 0.8134 net.
        context = p4.Context(0.8, 0.8, 0.5)
        belief = p4.Belief(0.8, 0.8)
        condition = p4.Condition(probe_accuracy=0.9, ask_accuracy=0.9,
                                 probe_cost=0.01, ask_cost=0.01)
        self.assertAlmostEqual(p4.information_gain(belief, context, condition, "probe"), 0.0)
        self.assertAlmostEqual(p4.information_gain(belief, context, condition, "ask"), 0.0)
        value, action = p4.plan(belief, context, condition)
        self.assertAlmostEqual(value, 0.8134)
        assert action in {"probe", "ask"}


    def test_exact_myopic_control_separates_joint_value_from_dev_overfit(self):
        p4 = self.p4
        self.assertTrue(callable(getattr(p4, "exact_value", None)), "Exact policy-value diagnostic is missing")
        context = p4.Context(0.8, 0.8, 0.5)
        belief = p4.Belief(0.8, 0.8)
        condition = p4.Condition(probe_accuracy=0.9, ask_accuracy=0.9,
                                 probe_cost=0.01, ask_cost=0.01)
        self.assertAlmostEqual(p4.exact_value(belief, context, condition, p4.Rule(1, 1, "max_net")), 0.8)
        self.assertAlmostEqual(p4.exact_value(belief, context, condition, "planner"), 0.8134)


    def test_noisy_uninformative_user_answers_do_not_create_information(self):
        p4 = self.p4
        condition = p4.Condition(allow_probe=False, ask_accuracy=0.5, ask_cost=0.01)
        result = p4.run_episode(example(p4), condition, "planner")
        assert len(result["trace"]) == 1
        assert result["cost"] == 0


    def test_dev_selection_rejects_test_cases(self):
        p4 = self.p4
        with self.assertRaisesRegex(ValueError, "dev"):
            p4.fit_simple([example(p4)], p4.Condition())


    def test_cases_reproduce_without_shared_dev_test_identifiers(self):
        p4 = self.p4
        a = p4.make_cases(8, seed=17, split="dev")
        b = p4.make_cases(8, seed=101, split="test")
        assert a == p4.make_cases(8, seed=17, split="dev")
        assert not ({case.case_id for case in a} & {case.case_id for case in b})


    def test_disabled_tools_are_not_called_by_fixed_or_planning_policies(self):
        p4 = self.p4
        condition = p4.Condition(allow_probe=False, allow_ask=False)
        for policy in ["never_ask", "always_ask", "probe_then_ask", "planner"]:
            result = p4.run_episode(example(p4), condition, policy)
            assert len(result["trace"]) == 1
            assert result["cost"] == 0

if __name__ == "__main__":
    unittest.main()
