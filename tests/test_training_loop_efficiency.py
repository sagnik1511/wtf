"""Exact learning equivalence and dead-buffer lifecycle; no profiling/timings."""
import copy
import json
import random
import unittest
import weakref
from unittest.mock import patch

from ecosystem.brain import Brain, Experience
from ecosystem.runner import Experiment
from tests.test_simulation import config


class FullOutputBrain(Brain):
    """The former full-output replay update, retained as an arithmetic reference."""
    def learn(self, experience):
        c=self.config
        if len(self.replay)<c['replay_capacity']:self.replay.append(experience)
        else:self.replay[self.cursor]=experience
        self.cursor=(self.cursor+1)%c['replay_capacity'];self.steps+=1
        self.epsilon=max(c['exploration']['minimum'],self.epsilon*c['exploration']['decay_per_decision'])
        if len(self.replay)<max(c['learning_starts_after'],c['batch_size']) or self.steps%c['train_every_steps']:return
        gradients={k:[0.0]*len(v) for k,v in self.weights.items()}
        gw1,gb1,gw2,gb2=(gradients[k] for k in ('w1','b1','w2','b2'))
        w2,hidden=self.weights['w2'],self.hidden
        for e in self.training_rng.sample(self.replay,c['batch_size']):
            h,q=self.forward(e.observation);a=self.actions.index(e.action);expected=e.reward
            if not e.terminated:expected+=c['discount_factor']*max(self.forward(e.next_observation,self.target)[1])
            delta=max(-1.0,min(1.0,q[a]-expected));gb2[a]+=delta
            for j in range(self.hidden):
                gw2[a*hidden+j]+=delta*h[j]
                dh=delta*w2[a*hidden+j]*(1-h[j]**2);gb1[j]+=dh;offset=j*20
                for i,x in enumerate(e.observation):gw1[offset+i]+=dh*x
        scale=c['learning_rate']/c['batch_size']
        for k,gradient in gradients.items():
            for i,value in enumerate(gradient):self.weights[k][i]-=scale*value
        self.updates+=1
        if self.updates%c['target_update_every_training_steps']==0:self.target={k:v.copy() for k,v in self.weights.items()}


class RetainedFounderExperiment(Experiment):
    def _release_dead_brains(self, ids):
        founders=sum(s['initial_population'] for s in self.config['species'].values())
        self._retire_brains(i for i in ids if i>=founders)


def breeding_config():
    c=config();c['world']['max_steps']=12
    c['brains']['small_q'].update(batch_size=1,learning_starts_after=1,train_every_steps=1)
    c['species']['prey']['reproduction'].update(enabled=True,health_threshold=0,min_age_steps=1,cooldown_steps=2,min_energy=10,energy_cost=5,max_population=8)
    return c


def living_state(e):
    state=e.checkpoint();alive=e.world.animals.keys()
    state['brains']={i:s for i,s in state['brains'].items() if int(i) in alive}
    state['inherited_updates']={i:n for i,n in state['inherited_updates'].items() if int(i) in alive}
    state['retired_updates']=e.total_updates-sum(b.updates-e.inherited_updates.get(i,0) for i,b in e.brains.items() if i in alive)
    return state


class TrainingLoopEfficiencyTests(unittest.TestCase):
    def test_selected_action_loss_matches_full_outputs_through_replay_wrap_and_resume(self):
        c=config()['brains']['small_q'];c.update(batch_size=3,replay_capacity=7,learning_starts_after=3,train_every_steps=2,target_update_every_training_steps=3)
        actions=['move_forward','turn_left','turn_right','rest','attack']
        a=Brain(c,actions,100,10);b=FullOutputBrain(c,actions,100,10);rng=random.Random(77)
        for step in range(65):
            observation=tuple(rng.uniform(-1,1) for _ in range(20));next_observation=tuple(rng.uniform(-1,1) for _ in range(20))
            self.assertEqual(a.choose(observation),b.choose(observation))
            terminated=step%7==0;truncated=not terminated and step%9==0
            feedback=Experience(observation,actions[step%len(actions)],rng.uniform(-3,3),None if terminated else next_observation,terminated,truncated)
            a.learn(feedback);b.learn(feedback)
            self.assertEqual(a.checkpoint(),b.checkpoint())
            if step==31:a=Brain.restore(json.loads(json.dumps(a.checkpoint())))
        self.assertGreater(a.updates,15)
        self.assertIsNot(a.weights['w1'],a.target['w1'])

    def test_founder_buffers_are_released_after_terminal_learning_and_champion_capture(self):
        e=Experiment(breeding_config());reference=weakref.ref(e.brains[3]);e.world.animals[3].energy=0;e.world.animals[3].health=.1
        e.observations=e.world.observe_all()
        with patch.object(Brain,'choose',lambda self,*args,**kwargs:'rest'):e.step()
        self.assertNotIn(3,e.world.animals);self.assertNotIn(3,e.brains);self.assertIsNone(reference())
        self.assertEqual(e.retired_updates,1)
        entry=e.champions.entries['predator']['female']['lifespan']
        self.assertEqual(entry['source']['optimizer_updates'],1)
        self.assertFalse(entry['source']['alive'])

    def test_releasing_buffers_preserves_world_learning_chaining_and_total_updates(self):
        a=Experiment(breeding_config());b=RetainedFounderExperiment(breeding_config())
        for e in (a,b):
            e.world.animals[3].energy=0;e.world.animals[3].health=.1;e.observations=e.world.observe_all()
        with patch.object(Brain,'choose',lambda self,*args,**kwargs:'rest'):
            steps=0
            while not a.done:
                a.step();b.step();steps+=1
                self.assertEqual(living_state(a),living_state(b))
                if steps==3:
                    child=next(i for i in a.world.animals if i>=4)
                    for e in (a,b):e.world.animals[child].health=.1;e.world.animals[child].energy=0;e.observations=e.world.observe_all()
                if steps==5:a=Experiment.restore(json.loads(json.dumps(a.checkpoint())))
        self.assertEqual(a.total_updates,b.total_updates)

    def test_old_modern_checkpoint_releases_dead_founders_without_losing_counters(self):
        original=RetainedFounderExperiment(breeding_config());original.world.animals[3].energy=0;original.world.animals[3].health=.1;original.observations=original.world.observe_all()
        with patch.object(Brain,'choose',lambda self,*args,**kwargs:'rest'):original.step()
        restored=Experiment.restore(json.loads(json.dumps(original.checkpoint())))
        self.assertNotIn(3,restored.brains);self.assertEqual(restored.total_updates,original.total_updates)
        self.assertEqual(living_state(restored),living_state(original))

    def test_legacy_checkpoint_founder_retention_remains_available(self):
        e=Experiment(breeding_config());e.legacy_brain_retention=True;brain=e.brains[3]
        e.world.animals[3].energy=0;e.world.animals[3].health=.1;e.observations=e.world.observe_all()
        with patch.object(Brain,'choose',lambda self,*args,**kwargs:'rest'):
            while not e.finished_run:e.step()
        self.assertIs(e.brains[3],brain);e.next_run();self.assertIs(e.brains[3],brain)
