"""Avoid duplicate work while retaining numerical and lifecycle behavior."""
import copy
import unittest
from unittest.mock import patch

from ecosystem.brain import Experience, INPUTS
from ecosystem.brains import create_brain
from ecosystem.ppo import MLP
from ecosystem.runner import Experiment
from ecosystem.contextual_ppo import central_contexts
from tests.test_advanced_algorithms import settings, environment


class AdvancedEfficiencyTests(unittest.TestCase):
    def test_variant_construction_only_builds_retained_networks(self):
        original = MLP.__init__
        for kind, expected in (('recurrent_ppo',1),('maskable_ppo',1),('mappo',2)):
            created=[]
            def track(network,*args,**kwargs):
                created.append(network)
                original(network,*args,**kwargs)
            with patch.object(MLP,'__init__',track):
                brain=create_brain(settings(kind),['rest','eat'],7,11)
            self.assertEqual(len(created),expected,kind)
            self.assertEqual(len(brain.moments),len(brain.network.layers))
            self.assertEqual(len(brain.variances),len(brain.network.layers))

    def test_mlp_can_return_input_gradient_in_same_parameter_pass(self):
        network=MLP([4,3],2,7,inputs=5)
        observation=(0.1,-0.2,0.3,0.4,-0.5)
        derivative=[0.4,-0.2,0.6]
        _,tape=network.forward(observation)
        normal,combined=network.zeros(),network.zeros()
        self.assertIsNone(network.accumulate_gradient(tape,derivative,normal))
        inputs=network.accumulate_gradient(tape,derivative,combined,return_input_gradient=True)
        self.assertEqual(normal,combined)
        for i,actual in enumerate(inputs):
            plus,minus=list(observation),list(observation)
            plus[i]+=1e-6;minus[i]-=1e-6
            numeric=sum((a-b)*d for a,b,d in zip(network.predict(plus),network.predict(minus),derivative))/2e-6
            self.assertAlmostEqual(actual,numeric,places=7)

    def test_gru_bootstrap_only_evaluates_value_head_without_advancing_memory(self):
        b=create_brain(settings('recurrent_ppo'),['rest','eat'],7,11)
        o=(0.2,)*INPUTS;action=b.choose(o)
        outputs,hidden=b.network.step(o,b.hidden)
        memory=copy.deepcopy(b.hidden)
        with patch.object(b.network.head,'predict',side_effect=AssertionError('unused policy heads')),patch.object(b.network.head,'value',wraps=b.network.head.value) as value:
            b.learn(Experience(o,action,1,o,False))
            value.assert_called_once()
        self.assertEqual(b.rollout[0].next_value,outputs[-1])
        self.assertEqual(b.hidden,memory)
        self.assertNotEqual(hidden,memory)

    def test_gru_backward_fuses_head_parameter_and_input_derivatives(self):
        b=create_brain(settings('recurrent_ppo'),['rest','eat'],7,11)
        _,_,tape=b.network.step((0.2,)*INPUTS,b.hidden,tape=True)
        with patch.object(b.network.head,'accumulate_gradient',wraps=b.network.head.accumulate_gradient) as backward:
            b.network.backward(tape,[0.2,0.4,-0.1],[0.0]*2,b.network.zeros())
            backward.assert_called_once()
            self.assertTrue(backward.call_args.kwargs['return_input_gradient'])

    def test_selected_output_ranges_match_full_inference_exactly(self):
        network=MLP([4,3],12,19,value_head=False)
        o=tuple((i-10)/20 for i in range(INPUTS))
        full=network.predict(o)
        for start,stop in ((0,3),(3,6),(9,12),(4,4),(0,12)):
            self.assertEqual(network.predict_range(o,start,stop),full[start:stop])
        for start,stop in ((-1,2),(3,2),(0,13)):
            with self.assertRaises(ValueError):network.predict_range(o,start,stop)

    def test_qrdqn_diagnostics_use_one_forward_pass(self):
        b=create_brain(settings('qrdqn'),['rest','eat'],7,11)
        o=(0.2,)*INPUTS
        expected=b.values(o)
        with patch.object(b.network,'predict',wraps=b.network.predict) as predict:
            data=b.diagnostics(o)
            predict.assert_called_once_with(o)
        self.assertEqual(list(data['action_values'].values()),expected)

    def test_qrdqn_bootstrap_uses_selected_target_heads(self):
        c=settings('qrdqn');c.update(batch_size=1,learning_starts_after=1)
        b=create_brain(c,['rest','eat'],7,11)
        o=(0.2,)*INPUTS
        with patch.object(b,'values',return_value=[-1,1]),patch.object(b.target_network,'predict',side_effect=AssertionError('all target heads')),patch.object(b.target_network,'predict_range',wraps=b.target_network.predict_range) as target:
            b.learn(Experience(o,'eat',1,o,False))
            target.assert_called_once_with(o,3,6)

    def test_mappo_post_step_context_built_once_and_shared(self):
        e=Experiment(environment('mappo'))
        with patch('ecosystem.contextual_ppo.central_contexts',wraps=central_contexts) as contexts:
            e.step()
            contexts.assert_called_once_with(e.world)
        expected=central_contexts(e.world)
        for i,animal in e.world.animals.items():
            self.assertEqual(e.brains[i].context,expected[animal.species])
            self.assertIs(e.brains[i].context,e.brains[i].next_context)

    def test_mappo_greedy_decision_does_not_evaluate_critic(self):
        b=create_brain(settings('mappo'),['rest','eat'],7,11)
        o=(0.2,)*INPUTS
        probabilities=b.action_probabilities(o)
        expected=b.actions[max(range(len(probabilities)),key=probabilities.__getitem__)]
        before=b.checkpoint()
        with patch.object(b.network.critic,'predict',side_effect=AssertionError('critic during execution')):
            self.assertEqual(b.choose(o,explore=False),expected)
        self.assertEqual(before,b.checkpoint())

    def test_grpo_masked_agents_share_trial_occupancy_without_brain_proxies(self):
        from ecosystem.contextual_ppo import action_mask
        from ecosystem.grouped_rollouts import collect_group
        from tests.test_policy_algorithms import GRPOTests
        from tests.test_simulation import animal
        world,brains=GRPOTests().encounter()
        world.animals[2]=animal(2,'prey',5,5)
        for i in (0,2):brains[i]=create_brain(settings('maskable_ppo'),['rest'],3,4)
        before={i:b.checkpoint() for i,b in brains.items() if i!=1}
        with patch('ecosystem.contextual_ppo.action_mask',wraps=action_mask) as masks,patch('copy.copy',side_effect=AssertionError('temporary controller copy')):
            collect_group(world,brains,1)
        calls=masks.call_args_list
        self.assertEqual(len(calls),16)
        for i in range(0,len(calls),2):
            self.assertIs(calls[i].args[3],calls[i+1].args[3])
            self.assertEqual(len(calls[i].args[3]),3)
        self.assertEqual(before,{i:b.checkpoint() for i,b in brains.items() if i!=1})
