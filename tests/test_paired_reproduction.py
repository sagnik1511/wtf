"""Sex-specific founder templates, biparental crossover and paired persistence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from ecosystem.brains import mix_parent_brains, model_snapshot
from ecosystem.checkpoint_io import read_checkpoint
from ecosystem.config import load
from ecosystem.runner import Experiment
from tests.reproduction_support import force_reproduction
from ecosystem.world import World
from tests.test_mutation import KINDS, environment
from tests.test_reproduction import breeding_world


def arrays(value):
    if isinstance(value, dict):
        for v in value.values():yield from arrays(v)
    elif value and isinstance(value[0],dict):
        for v in value:yield from arrays(v)
    else:yield value


def parameters(brain):
    return brain.weights if brain.config['type']=='q_network' else brain.network.layers


class PairedGeneticsTests(unittest.TestCase):
    def test_crossover_every_algorithm_mixes_parameters_without_parent_or_rng_changes(self):
        for kind in KINDS:
            e=force_reproduction(Experiment(environment(kind,0)));mother=e.brains[0];father=e.brains[1]
            for brain, value in ((mother,1.0),(father,-1.0)):
                for values in arrays(parameters(brain)):values[:]=type(values)([value]*len(values))
                brain.steps=13;brain.updates=7
            before=[copy.deepcopy(b.checkpoint()) for b in (mother,father)]
            world_rng=e.world.rng.getstate()
            child=mix_parent_brains(mother,father,12345)
            values=[v for a in arrays(parameters(child)) for v in a]
            self.assertEqual(set(values),{-1.0,1.0},kind)
            self.assertEqual(child.steps,0);self.assertEqual(child.updates,0)
            self.assertEqual(before,[b.checkpoint() for b in (mother,father)])
            self.assertEqual(world_rng,e.world.rng.getstate())
            for name in ('replay','trajectory','rollout'):
                if hasattr(child,name):self.assertEqual(getattr(child,name),[])
            if hasattr(child,'moments'):
                self.assertTrue(all(v==0 for a in arrays(child.moments) for v in a))
                self.assertTrue(all(v==0 for a in arrays(child.variances) for v in a))
            if hasattr(child,'target'):self.assertEqual(child.target,child.weights)
            if hasattr(child,'target_network'):self.assertEqual(child.target_network.layers,child.network.layers)
            if hasattr(child,'reference_network'):self.assertEqual(child.reference_network.layers,child.network.layers)
            if kind=='recurrent_ppo':self.assertTrue(all(v==0 for v in child.hidden))
            identical=mix_parent_brains(mother,father,12345)
            self.assertEqual(child.checkpoint(),identical.checkpoint())
            self.assertNotEqual(model_snapshot(child),model_snapshot(mix_parent_brains(mother,father,54321)))
            next(arrays(parameters(child)))[0]=99.0
            self.assertEqual(before,[b.checkpoint() for b in (mother,father)])

    def test_founders_have_two_independent_templates_with_balanced_sexes(self):
        for kind in KINDS:
            e=force_reproduction(Experiment(environment(kind,0)));groups={}
            for i,a in e.world.animals.items():
                if a.species=='prey':groups.setdefault(a.sex,[]).append(i)
            self.assertEqual(set(groups),{'male','female'})
            self.assertLessEqual(abs(len(groups['male'])-len(groups['female'])),1)
            self.assertNotEqual(model_snapshot(e.brains[groups['male'][0]]),model_snapshot(e.brains[groups['female'][0]]),kind)
            females=groups['female']
            self.assertEqual(model_snapshot(e.brains[females[0]]),model_snapshot(e.brains[females[1]]))
            before=copy.deepcopy(e.brains[females[1]].checkpoint())
            next(arrays(parameters(e.brains[females[0]])))[0]+=1
            self.assertEqual(before,e.brains[females[1]].checkpoint())

    def test_newborn_sex_and_both_parents_survive_json_binary_resume(self):
        for kind in KINDS:
            a=force_reproduction(Experiment(environment(kind)));a.step()
            children=[x for x in a.world.animals.values() if x.parent_id>=0]
            self.assertTrue(children,kind)
            for c in children:
                self.assertEqual(a.world.animals[c.parent_id].sex,'female')
                self.assertEqual(a.world.animals[c.second_parent_id].sex,'male')
            b=Experiment.restore(json.loads(json.dumps(a.checkpoint())))
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'state.bin.xz';a.write_checkpoint(path)
                c=Experiment.restore(read_checkpoint(path))
            while not a.done:
                a.step();b.step();c.step()
                self.assertEqual(a.checkpoint(),b.checkpoint(),kind)
                self.assertEqual(a.checkpoint(),c.checkpoint(),kind)

    def test_cap_applies_to_pairs_and_newborns_never_overlap(self):
        w=breeding_world()
        for i,x,sex in ((2,4,'female'),(3,5,'male')):
            a=copy.copy(w.animals[0]);a.id=i;a.x=x;a.sex=sex;w.animals[i]=a
        w.next_animal_id=4
        w.config['species']['prey']['reproduction']['max_population']=5
        w.step({i:'reproduce' for i in w.animals})
        self.assertEqual(w.births['prey'],1)
        self.assertEqual(len(w.animals),5)
        self.assertEqual(len({a.position for a in w.animals.values()}),5)
        self.assertEqual(sum(a.last_reproduced_age==1 for a in w.animals.values()),2)

    def test_solo_survivor_never_reproduces_even_with_high_health(self):
        w=breeding_world();del w.animals[1]
        for _ in range(20):w.step({0:'rest'})
        self.assertEqual(len(w.animals),1)
        self.assertEqual(dict(w.births),{})
        self.assertEqual(w.animals[0].energy,60)

    def test_legacy_founder_templates_restore_into_sex_banks(self):
        e=force_reproduction(Experiment(environment('ppo',0)));state=json.loads(json.dumps(e.checkpoint()))
        state['initial_weights']={name:weights['female'] for name,weights in state['initial_weights'].items()}
        for a in state['world']['snapshot']['animals']:
            a.pop('sex');a.pop('second_parent_id')
        restored=Experiment.restore(state)
        self.assertTrue(all(a.sex in ('female','male') for a in restored.world.animals.values()))
        self.assertEqual(set(restored.initial_weights['prey']),{'female','male'})
        restored.step()

    def test_analysis_counts_both_parents_cost_but_keeps_old_recording_semantics(self):
        from ecosystem.analyze import Statistics
        w=breeding_world();r=w.step({i:'reproduce' for i in w.animals})
        event=vars(next(x for x in r.events if x.kind=='born'))
        current=Statistics(w.config);current.event(event,vars(w.animals[0]))
        self.assertEqual(current.counts['reproduction_energy'],10)
        self.assertEqual(current.counts['offspring_energy'],60)
        legacy=dict(event);legacy.pop('mate')
        old=Statistics(w.config);old.event(legacy,vars(w.animals[0]))
        self.assertEqual(old.counts['reproduction_energy'],5)

    def test_incompatible_parents_are_rejected(self):
        mother=force_reproduction(Experiment(environment('ppo'))).brains[0]
        father=force_reproduction(Experiment(environment('a2c'))).brains[1]
        with self.assertRaisesRegex(ValueError,'compatible'):mix_parent_brains(mother,father,1)

    def test_presets_expose_mate_radius(self):
        for p in (Path(__file__).resolve().parents[1]/'environments').glob('*.json'):
            for s in load(p)['species'].values():self.assertEqual(s['reproduction']['mate_radius'],2)


if __name__=='__main__':unittest.main()
