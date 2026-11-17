"""Seeded reproduction timing windows, resource gates and resume boundaries."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ecosystem.checkpoint_io import read_checkpoint
from ecosystem.config import load, validate
from ecosystem.replay_codec import encode_chunk, decode_chunk
from ecosystem.runner import Experiment
from ecosystem.world import World
from tests.test_simulation import animal, config
from tests.test_reproduction import breeding_world, reproduction


def window_world(**overrides):
    w = breeding_world()
    w.config['species']['prey']['reproduction'] = reproduction(
        min_age_steps=10, cooldown_steps=8, age_deviation_pct=30,
        cooldown_deviation_pct=25, **overrides)
    w.animals[1].reproduction_due_age=1
    return w


class ReproductionWindowTests(unittest.TestCase):
    def test_both_inclusive_window_endpoints(self):
        for due in (7,13):
            w=window_world();parent=w.animals[0]
            with patch.object(w.rng,'randint',return_value=due) as draw:
                self.assertFalse(w._reproduction_ready(parent,w.config['species']['prey']['reproduction']))
                draw.assert_called_once_with(7,13)
            self.assertEqual(parent.reproduction_due_age,due)
            parent.age=due-1
            self.assertFalse(w._reproduction_ready(parent,w.config['species']['prey']['reproduction']))
            parent.age=due
            self.assertTrue(w._reproduction_ready(parent,w.config['species']['prey']['reproduction']))

    def test_draw_once_not_every_step_and_stay_due_when_blocked(self):
        w=window_world();w.config['species']['prey']['reproduction']['max_population']=1
        with patch.object(w.rng,'randint',return_value=7) as draw:
            for _ in range(16):w.step({i:'rest' for i in w.animals})
            draw.assert_called_once_with(7,13)
        self.assertEqual(w.animals[0].last_reproduced_age,-1)
        self.assertEqual(w.animals[0].energy,60)
        self.assertEqual(w.animals[0].reproduction_due_age,7)
        w.config['species']['prey']['reproduction']['max_population']=3
        w.step({i:'rest' for i in w.animals})
        self.assertEqual(w.births['prey'],1)
        self.assertEqual(w.animals[0].last_reproduced_age,17)

    def test_successful_birth_schedules_cooldown_from_actual_birth(self):
        for cooldown in (6,10):
            w=window_world();parent=w.animals[0];parent.age=6
            with patch.object(w.rng,'randint',side_effect=[7,cooldown,cooldown]) as draw:
                w.step({i:'rest' for i in w.animals})
                self.assertEqual(draw.call_args_list[1].args,(6,10))
            self.assertEqual(parent.last_reproduced_age,7)
            self.assertEqual(parent.reproduction_due_age,7+cooldown)
            child=next(a for a in w.animals.values() if a.parent_id==0)
            self.assertEqual(child.reproduction_due_age,-1)
            # Block further births, then check the newborn draws its own first clock.
            w.config['species']['prey']['reproduction']['max_population']=3
            with patch.object(w.rng,'randint',return_value=13) as draw:
                w.step({i:'rest' for i in w.animals})
                draw.assert_called_once_with(7,13)
            self.assertEqual(child.reproduction_due_age,13)
            self.assertEqual(parent.reproduction_due_age,7+cooldown)

    def test_health_energy_and_space_can_delay_beyond_window(self):
        for block in ('health','energy','space'):
            w=window_world();parent=w.animals[0]
            if block=='health':parent.health=24
            elif block=='energy':parent.energy=1
            else:w.obstacles={(2,1),(3,2),(2,3),(1,2)}
            with patch.object(w.rng,'randint',return_value=7):
                for _ in range(14):w.step({i:'rest' for i in w.animals})
            self.assertEqual(len(w.animals),2,block)
            self.assertEqual(parent.reproduction_due_age,7)
            parent.health=30;parent.energy=60;w.obstacles.clear()
            w.step({i:'rest' for i in w.animals})
            self.assertEqual(w.births['prey'],1,block)
            self.assertEqual(parent.last_reproduced_age,15)
            self.assertTrue(21<=parent.reproduction_due_age<=25)

    def test_windows_scatter_animals_and_clip_to_positive_steps(self):
        w=window_world();r=w.config['species']['prey']['reproduction']
        values=[]
        for i in range(100):
            a=animal(i,'prey',2,2)
            w._reproduction_ready(a,r);values.append(a.reproduction_due_age)
        self.assertEqual(set(values),set(range(7,14)))
        r.update(min_age_steps=2,age_deviation_pct=100)
        values=[]
        for i in range(100):
            a=animal(i,'prey',2,2);w._reproduction_ready(a,r);values.append(a.reproduction_due_age)
        self.assertEqual(set(values),set(range(1,5)))

    def test_disabled_or_zero_deviation_preserves_old_rng_and_births(self):
        a=breeding_world();b=World.restore(json.loads(json.dumps(a.checkpoint())))
        b.config['species']['prey']['reproduction'].update(age_deviation_pct=0,cooldown_deviation_pct=0)
        for _ in range(7):
            a.step({i:'rest' for i in a.animals});b.step({i:'rest' for i in b.animals})
            self.assertEqual(a.snapshot(),b.snapshot())
            self.assertEqual(a.rng.getstate(),b.rng.getstate())
        w=window_world();w.config['species']['prey']['reproduction']['enabled']=False
        with patch.object(w.rng,'randint',side_effect=AssertionError('unexpected timing draw')):
            w.step({i:'rest' for i in w.animals})
        self.assertEqual(w.animals[0].reproduction_due_age,-1)

    def test_one_zero_deviation_only_randomizes_requested_clock(self):
        w=window_world();r=w.config['species']['prey']['reproduction'];r['age_deviation_pct']=0
        with patch.object(w.rng,'randint',side_effect=AssertionError('first clock is fixed')):
            self.assertFalse(w._reproduction_ready(w.animals[0],r))
        self.assertEqual(w.animals[0].reproduction_due_age,10)
        r['cooldown_deviation_pct']=0
        with patch.object(w.rng,'randint',side_effect=AssertionError('cooldown is fixed')):
            self.assertEqual(w._reproduction_interval(r),8)

    def test_old_checkpoint_with_no_clock_or_deviation_restores(self):
        w=breeding_world();state=json.loads(json.dumps(w.checkpoint()))
        for a in state['snapshot']['animals']:a.pop('reproduction_due_age')
        restored=World.restore(state)
        self.assertEqual(restored.animals[0].reproduction_due_age,-1)
        w.step({i:'rest' for i in w.animals});restored.step({i:'rest' for i in restored.animals})
        self.assertEqual(w.checkpoint(),restored.checkpoint())

    def test_json_compact_resume_and_delta_preserve_pending_clocks(self):
        c=config();c['obstacles'].update(generation='explicit',explicit_cells=[])
        c['world']['max_steps']=16;c['experiment']['runs']=2
        c['species']['prey']['reproduction']=reproduction(min_age_steps=4,cooldown_steps=4,
            age_deviation_pct=50,cooldown_deviation_pct=25)
        for spec in c['species'].values():
            spec['body'].update(living_cost=0,movement_cost=0)
            if spec['diet']=='prey':spec['body'].update(attack_damage=0,attack_cost=0)
        a=Experiment(c)
        for _ in range(3):a.step()
        b=Experiment.restore(json.loads(json.dumps(a.checkpoint())))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.json.gz';a.write_checkpoint(path)
            d=Experiment.restore(read_checkpoint(path))
        frames=[copy.deepcopy(a.world.snapshot())]
        while not a.done:
            a.step();b.step();d.step()
            self.assertEqual(a.checkpoint(),b.checkpoint())
            self.assertEqual(a.checkpoint(),d.checkpoint())
            frames.append(copy.deepcopy(a.world.snapshot()))
        packet=encode_chunk(frames)
        self.assertIn('reproduction_due_age',packet['animal_fields'])
        self.assertEqual(decode_chunk(json.loads(json.dumps(packet))),frames)
        self.assertTrue(any(x.get('reproduction_due_age',-1)>=0 for x in frames[0]['animals']))

    def test_percentage_scales_with_center_and_rounds_half_up(self):
        w=window_world()
        for center,pct,deviation in ((300,10,30),(225,10,23),(100,2.5,3),(10,1,0),(200,100,200)):
            rules={'min_age_steps':center,'age_deviation_pct':pct}
            with patch.object(w.rng,'randint',return_value=center) as draw:
                self.assertEqual(w._reproduction_interval(rules,first=True),center)
                if deviation:draw.assert_called_once_with(max(1,center-deviation),center+deviation)
                else:draw.assert_not_called()

    def test_legacy_step_windows_and_ambiguous_units(self):
        w=window_world()
        legacy={'min_age_steps':10,'age_deviation_steps':3}
        with patch.object(w.rng,'randint',return_value=7) as draw:
            self.assertEqual(w._reproduction_interval(legacy,first=True),7)
            draw.assert_called_once_with(7,13)
        for prefix in ('age','cooldown'):
            c=config();c['species']['prey']['reproduction'][prefix+'_deviation_steps']=2
            with self.assertRaisesRegex(ValueError,'cannot combine'):validate(c)

    def test_validation_and_all_presets(self):
        for key in ('age_deviation_pct','cooldown_deviation_pct'):
            for bad in (-1,True,100.1,float("nan")):
                c=config();c['species']['prey']['reproduction'][key]=bad
                with self.assertRaisesRegex(ValueError,key):validate(c)
        root=Path(__file__).resolve().parents[1]
        for p in (root/'environments').glob('*.json'):
            c=load(p)
            for spec in c['species'].values():
                r=spec.get('reproduction')
                if r:
                    self.assertGreater(r['age_deviation_pct'],0,p.name)
                    self.assertGreater(r['cooldown_deviation_pct'],0,p.name)


if __name__=='__main__':unittest.main()
