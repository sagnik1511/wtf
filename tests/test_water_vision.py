import copy
import json
import unittest
from unittest.mock import patch

from ecosystem.config import validate,load
from ecosystem.runner import Experiment
from ecosystem.world import Animal,Resource,World
from tests.test_geography import configuration


def setup(columns=(2,3,4)):
    rows=[''.join('~' if x in columns else '.' for x in range(15))]*7
    c=configuration(rows=rows)
    for s in c['species'].values():s['body'].update(vision_radius=2,water_vision_bonus=3)
    w=World(validate(c),100)
    w.animals={0:Animal(0,'prey',1,3,1,60,30),1:Animal(1,'predator',6,3,3,80,55)}
    return w


class WaterVisionTests(unittest.TestCase):
    def test_bonus_is_directional_water_limited_and_capped(self):
        w=setup();a=w.animals[0]
        self.assertEqual(w.observe(a)[19],1)
        self.assertEqual(len(w.observe(a)),32)
        w.resources={(1,6):Resource(1,6,10)}
        self.assertEqual(w.observe(a)[11],0)  # Three dry cells, normal radius two.
        w.animals[1].x=7
        self.assertEqual(w.observe(a)[19],0)  # Past max radius five.
        w=setup((3,));a=w.animals[0]
        w.animals[1].x=4
        self.assertEqual(w.observe(a)[19],1)  # One water cell grants one extra.
        w.animals[1].x=5
        self.assertEqual(w.observe(a)[19],0)

    def test_wall_occlusion_and_dynamic_targets_invalidate_stationary_observer(self):
        w=setup();a=w.animals[0]
        self.assertEqual(w.observe(a)[19],1)
        w.obstacles.add((3,3));self.assertEqual(w.observe(a)[19],0)
        w.obstacles.remove((3,3));self.assertEqual(w.observe(a)[19],1)
        w.animals[1].y=4;self.assertEqual(w.observe(a)[19],0)
        w.animals[1].y=3;self.assertEqual(w.observe(a)[19],1)
        del w.animals[1];self.assertEqual(w.observe(a)[19],0)
        w.resources={(6,3):Resource(6,3,10)};self.assertEqual(w.observe(a)[11],1)
        w.resources[(6,3)].quantity=0;self.assertEqual(w.observe(a)[11],0)

    def test_cache_key_includes_base_and_bonus_even_when_max_radius_is_unchanged(self):
        w=setup((3,));a=w.animals[0];w.animals[1].x=5
        self.assertEqual(w.observe(a)[19],0)
        w.body(a).update(vision_radius=4,water_vision_bonus=1)
        self.assertEqual(w.observe(a)[19],1)

    def test_zero_bonus_preserves_old_observations_and_dry_maps_get_no_bonus(self):
        w=setup();a=w.animals[0];w.body(a)['water_vision_bonus']=0
        self.assertEqual(w.observe(a)[19],0)
        old=w.observe(a);del w.body(a)['water_vision_bonus']
        self.assertEqual(w.observe(a),old)
        w=setup(());self.assertEqual(w.observe(w.animals[0])[19],0)

    def test_ordered_static_water_cache_reuses_counts_and_stays_bounded(self):
        w=setup();w.observe(w.animals[0])
        with patch.object(w.geography,'depth',wraps=w.geography.depth) as depth:
            self.assertEqual(w._water_sight_cells((1,3),(6,3)),3)
            self.assertEqual(depth.call_count,0)
        with patch('ecosystem.world.VISIBILITY_CACHE_LIMIT',4):
            for x in range(6,12):w._water_sight_cells((1,3),(x,3))
            self.assertEqual(len(w._water_sight_cache),4)

    def test_diagonal_water_ray_still_respects_walls(self):
        w=setup();a=w.animals[0];a.x=a.y=1;w.body(a)['vision_radius']=3
        w.animals[1].x=w.animals[1].y=5
        self.assertEqual(w.observe(a)[19],1)
        w.obstacles.add((3,3));self.assertEqual(w.observe(a)[19],0)

    def test_validation_and_supplied_water_presets(self):
        c=setup().config
        for bad in (-1,33,True,1.5):
            changed=copy.deepcopy(c);changed['species']['prey']['body']['water_vision_bonus']=bad
            with self.assertRaises(ValueError):validate(changed)
        for name in ('river_crossing','river_crossing_ppo','enclosed_valleys'):
            c=load('environments/'+name+'.json')
            self.assertTrue(all(s['body']['water_vision_bonus']==3 for s in c['species'].values()))
        self.assertTrue(all('water_vision_bonus' not in s['body'] for s in load('environments/highlands.json')['species'].values()))

    def test_exact_ppo_continuation_with_warm_and_cold_perception_caches(self):
        c=configuration('ppo',rows=['..~~~..........']*7)
        for s in c['species'].values():s['body'].update(vision_radius=2,water_vision_bonus=3)
        for b in c['brains'].values():
            if b['type']=='ppo':b['rollout_steps']=8
        c['experiment'].update(runs=1,world_seed_policy='increment');c['world']['max_steps']=40
        e=Experiment(c)
        for _ in range(12):e.step()
        restored=Experiment.restore(json.loads(json.dumps(e.checkpoint())))
        for _ in range(12):e.step();restored.step()
        self.assertEqual(e.checkpoint(),restored.checkpoint())
        self.assertGreater(e.total_updates,0)


if __name__=='__main__':unittest.main()
