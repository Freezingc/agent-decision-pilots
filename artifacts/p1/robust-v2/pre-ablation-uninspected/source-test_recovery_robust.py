"""Synthetic mathematical fixtures for P1 robust-v2, never natural results."""
import unittest
try:
    from agent_pilots import recovery_robust as q
except ImportError:
    q = None


class RobustRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(q, "robust recovery not implemented")
        self.models = (q.v1.Parameters(not_executed=.4, stale_read=.25, delay=((0,1.),)),
                       q.v1.Parameters(not_executed=.4, stale_read=.25, delay=((3,1.),)))

    def test_bayes_conditions_each_model_and_model_weights_on_history(self):
        belief = q.ModelBeliefs.initial(self.models)
        updated = belief.update('check','absent')
        self.assertAlmostEqual(updated.beliefs[0][(0,0)], .4/.55)
        self.assertAlmostEqual(updated.beliefs[1][(0,0)], .4)
        self.assertAlmostEqual(updated.weights[0], .55/1.55)
        self.assertAlmostEqual(sum(updated.weights),1.)
        for b in updated.beliefs: self.assertAlmostEqual(sum(b.values()),1.)

    def test_worst_case_gate_never_averages_away_bad_model(self):
        b=q.ModelBeliefs(self.models,({(0,0):1.}, {(1,0):1.}),(.999,.001))
        self.assertEqual(b.risk('executed'),1.)
        self.assertFalse(q.retry_allowed(b,0))
        self.assertFalse(q.accept_allowed(b))

    def test_retry_is_once_even_after_posterior_is_safe(self):
        b=q.ModelBeliefs(self.models,({(0,0):1.}, {(0,0):1.}),(.5,.5))
        self.assertTrue(q.retry_allowed(b,0))
        self.assertFalse(q.retry_allowed(b,1))

    def test_accept_has_explicit_false_completion_boundary(self):
        b=q.ModelBeliefs(self.models,({(0,0):.051,(1,0):.949},)*2,(.5,.5))
        self.assertFalse(q.accept_allowed(b))
        b=q.ModelBeliefs(self.models,({(0,0):.05,(1,0):.95},)*2,(.5,.5))
        self.assertTrue(q.accept_allowed(b))

    def test_minimax_terminal_value_matches_manual_worst_case(self):
        b=q.ModelBeliefs(self.models,({(0,0):.1,(1,0):.9},{(0,0):.4,(1,0):.6}),(.5,.5))
        self.assertEqual(q.RobustPlanner(self.models).value(b,0,0), (8.,'abandon'))

    def test_fixed_rules_include_direct_accept_and_early_abstention(self):
        rule=q.FixedRule(wait=0,negatives=1,direct_accept=True)
        history=[{'action':'check','observation':'absent'},{'action':'retry','observation':'timeout'}]
        self.assertEqual(rule.choose(history,10),'accept')
        self.assertEqual(q.FixedRule(abandon_after=0).choose([],12),'abandon')

    def test_nominal_and_long_models_update_on_same_visible_history(self):
        b=q.ModelBeliefs.initial(self.models)
        for _ in range(4): b=b.update('check','absent')
        for model,post in zip(self.models,b.beliefs):
            direct=q.v1.initial_belief(model)
            for _ in range(4): direct,_=q.v1.update_belief(direct,'check','absent',model)
            self.assertEqual(post,direct)

    def test_simulation_preserves_v1_counterfactual_randomization(self):
        p=self.models[0]; rule=q.FixedRule(initial_terminal='abandon')
        a=q.simulate(p,rule,41,self.models)
        b=q.v1.simulate(p,q.v1.Rule(initial_terminal='abandon'),41,horizon=12)
        self.assertEqual(a['initial_state'],b['initial_state'])
        self.assertEqual(a['terminal_state'],b['terminal_state'])
        self.assertEqual(a['cost'],0.)


if __name__=='__main__': unittest.main()
