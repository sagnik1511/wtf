"""Shared inference, scratch buffers and trial-history isolation; no timings."""
import copy
import json
import math
import random
import unittest
from unittest.mock import patch

from ecosystem.brain import Experience
from ecosystem.ppo import MLP, PPOBrain, policy_distribution
from ecosystem.policy_gradients import A2CBrain, ReinforceBrain, GRPOBrain, PolicyStep, normalized
from ecosystem.grouped_rollouts import collect_group
from ecosystem.world import World
from tests.test_ppo import ppo_settings
from tests.test_policy_algorithms import settings
from tests import test_policy_algorithms as policy_fixtures


class FullTapeFreshGradientMLP(MLP):
    """Reference paths retain all activations and allocate fresh gradient rows."""
    def predict(self, observation):return self.forward(observation)[0]
    def value(self, observation):return self.forward(observation)[0][-1]
    def clear_gradients(self, gradients):gradients[:]=self.zeros()


class PolicyLoopEfficiencyTests(unittest.TestCase):
    def test_inference_and_value_only_match_backprop_outputs_after_weight_edits(self):
        rng=random.Random(42)
        for hidden in ([],[4],[5,3,2]):
            for value_head in (False,True):
                network=MLP(hidden,3,7,value_head=value_head)
                for _ in range(8):
                    observation=tuple(rng.uniform(-1,1) for _ in range(20))
                    expected=network.forward(observation)[0]
                    self.assertEqual(network.predict(observation),expected)
                    if value_head:self.assertEqual(network.value(observation),expected[-1])
                    network.layers[0]['weights'][0]+=.1

    def test_gradient_reset_preserves_storage_and_removes_previous_minibatch(self):
        network=MLP([4,3],3,7);gradients=network.zeros()
        ids=[id(v) for layer in gradients for v in layer.values()]
        for layer in gradients:
            for values in layer.values():
                for i in range(len(values)):values[i]=i+1.0
        network.clear_gradients(gradients)
        self.assertEqual(gradients,network.zeros())
        self.assertEqual(ids,[id(v) for layer in gradients for v in layer.values()])

    def test_inplace_normalization_matches_independent_formula_without_changing_default(self):
        values=[-3.0,2.0,7.0,.4];mean=sum(values)/len(values)
        scale=math.sqrt(sum((v-mean)**2 for v in values)/len(values)+1e-8)
        expected=[(v-mean)/scale for v in values];original=values.copy()
        self.assertEqual(normalized(values),expected);self.assertEqual(values,original)
        result=normalized(values,inplace=True)
        self.assertIs(result,values);self.assertEqual(result,expected)

    def test_all_policy_controllers_match_full_tape_and_fresh_gradients_through_resume(self):
        for kind,cls in [('ppo',PPOBrain),('a2c',A2CBrain),('reinforce',ReinforceBrain),('grpo',GRPOBrain)]:
            with self.subTest(kind=kind):
                c=ppo_settings() if kind=='ppo' else settings(kind)
                c['network']['hidden_layers']=[4,3]
                if kind in ('ppo','grpo'):c.update(epochs=3,batch_size=3,target_kl=0)
                if 'normalize_advantages' in c:c['normalize_advantages']=True
                a,b=cls(c,['rest','attack','turn_left'],3,5),cls(c,['rest','attack','turn_left'],3,5)
                def reference_path():
                    b.network.__class__=FullTapeFreshGradientMLP
                    if kind=='grpo':b.reference_network.__class__=FullTapeFreshGradientMLP
                reference_path();rng=random.Random(11)
                for tick in range(18):
                    o=tuple(rng.uniform(-1,1) for _ in range(20));n=tuple(rng.uniform(-1,1) for _ in range(20))
                    action=a.choose(o);self.assertEqual(action,b.choose(o))
                    terminated=tick%9==8;truncated=tick%7==6 and not terminated
                    e=Experience(o,action,rng.uniform(-2,2),None if terminated else n,terminated,truncated)
                    a.learn(e);b.learn(e)
                    if kind=='grpo' and tick%3==2:
                        groups=[]
                        for branch,length in enumerate((1,3,4,2)):
                            trajectory=[]
                            for i in range(length):
                                observation=tuple(rng.uniform(-1,1) for _ in range(20));index=i%3
                                logp=policy_distribution(a.network.forward(observation)[0])[1][index]
                                trajectory.append(PolicyStep(observation,index,(branch-1)*.3+i*.1,logp,False,i==length-1))
                            groups.append(trajectory)
                        a.update_group(groups);b.update_group(groups)
                    self.assertEqual(json.dumps(a.checkpoint(),sort_keys=True),json.dumps(b.checkpoint(),sort_keys=True))
                    self.assertEqual(a.diagnostics(o),b.diagnostics(o))
                    if tick==10:
                        a=cls.restore(json.loads(json.dumps(a.checkpoint())))
                        b=cls.restore(json.loads(json.dumps(b.checkpoint())));reference_path()

    def test_grpo_trials_omit_old_deaths_without_changing_trials_or_real_world(self):
        world,brains=policy_fixtures.GRPOTests().encounter();focal=brains[1]
        focal.config['rollout_horizon']=4
        world.lifespans=[dict(id=500+i,species='prey',age=10,cause='starvation') for i in range(100)]
        world.next_animal_id=10000
        before=copy.deepcopy(world.checkpoint());group_rng=focal.group_rng.getstate()
        restore=World.restore;lengths=[]
        def compact_restore(state):
            lengths.append(len(state['lifespans']));trial=restore(state)
            self.assertEqual(trial.next_animal_id,10000);return trial
        with patch.object(World,'restore',side_effect=compact_restore):actual=collect_group(world,brains,1)
        self.assertEqual(lengths,[0]*focal.config['group_size'])
        focal.group_rng.setstate(group_rng)
        def history_restore(state):return restore(dict(state,lifespans=before['lifespans']))
        with patch.object(World,'restore',side_effect=history_restore):reference=collect_group(world,brains,1)
        self.assertEqual(actual,reference);self.assertEqual(world.checkpoint(),before)

    def test_zero_kl_grpo_does_not_evaluate_unused_reference_model(self):
        world,brains=policy_fixtures.GRPOTests().encounter();brain=brains[1];brain.config['kl_coefficient']=0
        groups=collect_group(world,brains,1)
        # The fixture includes successful hunts and misses, so learning executes.
        with patch.object(brain.reference_network,'predict',side_effect=AssertionError('unused reference inference')):
            brain.update_group(groups)
        self.assertGreater(brain.updates,0)
