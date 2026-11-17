import copy
import json
import unittest
from collections import Counter, deque

from ecosystem.config import load, validate
from ecosystem.placement import zone, resource_pools
from ecosystem.runner import Experiment
from ecosystem.world import World, DIRECTIONS


class PlacementTests(unittest.TestCase):
    def test_founder_zones_food_quotas_and_dry_unique_cells_across_seeds(self):
        c = load('environments/enclosed_valleys.json')
        for seed in (0,42,100,201,999):
            w = World(c,seed)
            b = c['placement']['enclosure_bounds']
            self.assertEqual(Counter((a.species,zone(a.position,b)) for a in w.animals.values()),
                             {('prey','inside'):48,('predator','outside'):12})
            self.assertEqual(len({a.position for a in w.animals.values()}),60)
            self.assertEqual(len(w.resources),80)
            self.assertEqual({k:len(v) for k,v in resource_pools(w.resources,c['placement']).items()},
                             {'inside':30,'outside':30,'far':20})
            for p in list(w.resources)+[a.position for a in w.animals.values()]:
                self.assertFalse(w.blocked(p));self.assertFalse(w.geography.depth(p))
            self.assertEqual(w.snapshot(),World(c,seed).snapshot())
        self.assertNotEqual(World(c,100).snapshot(),World(c,101).snapshot())

    def test_two_gates_connect_the_interior_to_exterior_over_walkable_slopes(self):
        c=load('environments/enclosed_valleys.json');w=World(c,100)
        left,top,right,bottom=c['placement']['enclosure_bounds']
        perimeter={(x,y) for x in range(left,right+1) for y in (top,bottom)}
        perimeter|={(x,y) for y in range(top,bottom+1) for x in (left,right)}
        self.assertEqual(perimeter-w.obstacles,{(8,y) for y in (12,13,14)} | {(31,y) for y in (13,14,15)})
        for xs,ys in (((6,7,8,9),(12,13,14)),((30,31,32,33),(13,14,15))):
            self.assertTrue(all(not w.blocked((x,y)) and not w.geography.depth((x,y)) for x in xs for y in ys))
        start=(9,5);seen={start};queue=deque([start])
        while queue:
            p=queue.popleft()
            for dx,dy in DIRECTIONS:
                q=p[0]+dx,p[1]+dy
                if q in seen or w.blocked(q) or w.geography.depth(q):continue
                if abs(w.geography.elevation(q)-w.geography.elevation(p))>1:continue
                seen.add(q);queue.append(q)
        self.assertIn((7,13),seen);self.assertIn((32,14),seen)
        self.assertTrue(all((x,y) in seen for x in range(9,31) for y in range(5,23)))

    def test_multiple_species_share_the_zone_pool_without_overlapping(self):
        c=load('environments/enclosed_valleys.json')
        c['species']['deer']=copy.deepcopy(c['species']['prey']);c['species']['deer']['initial_population']=8
        c['placement']['species']['deer']='inside'
        w=World(validate(c),100)
        self.assertEqual(len({a.position for a in w.animals.values()}),68)
        self.assertTrue(all(zone(a.position,c['placement']['enclosure_bounds'])=='inside' for a in w.animals.values() if a.species=='deer'))

    def test_invalid_rules_and_zone_capacity_fail_before_simulation(self):
        base=load('environments/enclosed_valleys.json')
        mutations=[lambda c:c['placement'].update(enclosure_bounds=[8,4,8,23]),
                   lambda c:c['placement']['species'].update(prey='anywhere'),
                   lambda c:c['placement']['species'].pop('predator'),
                   lambda c:c['placement']['resources'].update(boundary_fraction=1.1),
                   lambda c:c['placement']['resources'].update(boundary_width=True),
                   lambda c:c['placement'].update(enclosure_bounds=[0,0,39,27]),
                   lambda c:c['species']['prey'].update(initial_population=500),
                   lambda c:c['resources'].update(patch_count=1000),
                   lambda c:c['obstacles'].update(generation='scattered')]
        for mutate in mutations:
            c=copy.deepcopy(base);mutate(c)
            with self.assertRaises(ValueError):validate(c)

    def test_exact_learning_continuation_keeps_body_zones_as_initial_conditions(self):
        c=load('environments/enclosed_valleys.json');c['experiment'].update(runs=1,world_seed_policy='increment');c['world']['max_steps']=100
        e=Experiment(c)
        for _ in range(12):e.step()
        restored=Experiment.restore(json.loads(json.dumps(e.checkpoint())))
        for _ in range(54):e.step();restored.step()
        for name in ("prey", "predator"):
            self.assertTrue(any(e.brains[a.id].updates > 0 for a in e.world.animals.values() if a.species == name))
        self.assertEqual(e.checkpoint(),restored.checkpoint())
        self.assertEqual(e.config['placement'],restored.config['placement'])


if __name__=='__main__':unittest.main()
