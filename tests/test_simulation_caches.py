"""Bounded geometry reuse and final-winner snapshots preserve simulation state."""
import copy
import json
import random
import unittest
from unittest.mock import patch

from ecosystem.brains import create_brain,model_snapshot
from ecosystem.library import ChampionBook,balanced_policy,balanced_rank,ranks
from ecosystem.runner import Experiment
from ecosystem.world import World,VISIBILITY_CACHE_LIMIT
from tests.test_simulation import config,animal,empty_world
from tests.test_efficiency import reference_observation
from tests.test_advanced_algorithms import environment


def old_consider(book,animals,brains,source):
    policies={}
    for a in animals:
        champions=book.entries.setdefault(a.species,{})
        scores=ranks(a)
        if book.config['species'][a.species]['diet']=='prey':
            if a.species not in policies:policies[a.species]=balanced_policy(book.config,a.species)
            scores['balanced']=balanced_rank(a.age,a.kills,a.energy_gained,a.health,policies[a.species])
        wins=[k for k,v in scores.items() if k not in champions or v>tuple(champions[k]['rank'])]
        if not wins:continue
        brain=brains[a.id];model=model_snapshot(brain)
        source_data=dict(source,animal_id=a.id,species=a.species,health=a.health,alive=a.health>0,
            age=a.age,kills=a.kills,energy_gained=a.energy_gained,parent_id=a.parent_id,
            body_generation=a.generation,statistics_since_age=a.statistics_since_age,optimizer_updates=brain.updates)
        for key in wins:
            rank=scores[key]
            champions[key]=dict(criterion=key,score=rank[0],rank=list(rank),source=source_data,model=model)
            if key=='balanced':champions[key]['selection_policy']=policies[a.species]


class SimulationCacheTests(unittest.TestCase):
    def test_visibility_reuses_ray_when_target_index_changes_but_geometry_does_not(self):
        w=empty_world();a=animal(0,'prey',2,2)
        from ecosystem.world import Resource
        w.animals={0:a};w.resources={(4,2):Resource(4,2,10)}
        with patch.object(w,'visible',wraps=w.visible) as ray:
            first=w.observe_all()[0]
            self.assertGreater(ray.call_count,0)
            # Force a nearest-result miss while retaining the same ray endpoints.
            w.resources[(9,7)]=Resource(9,7,10)
            for index in w._perception:index.results.clear()
            ray.reset_mock();second=w.observe_all()[0]
            self.assertEqual(first,second)
            self.assertEqual(ray.call_count,0)
        self.assertEqual(second,reference_observation(w,a))

    def test_obstacle_edits_invalidate_visibility_and_nearest_results(self):
        w=empty_world();a=animal(0,'prey',2,2)
        from ecosystem.world import Resource
        w.animals={0:a};w.resources={(4,2):Resource(4,2,10)}
        first=w.observe_all()[0];self.assertEqual(first[11],1.0)
        w.obstacles.add((3,2))
        observed=w.observe_all()[0]
        self.assertEqual(observed,reference_observation(w,a));self.assertEqual(observed[11],0.0)
        w.obstacles.clear()
        self.assertEqual(w.observe_all()[0],first)
        w.obstacles={(4,2)}
        self.assertEqual(w.observe_all()[0],reference_observation(w,a))

    def test_ray_cache_is_bounded_ordered_and_not_checkpointed(self):
        w=empty_world();w.observe_all()
        with patch.object(w,'visible',return_value=True):
            for x in range(VISIBILITY_CACHE_LIMIT+10):w._perception_visible((0,0),(x,0))
        self.assertEqual(len(w._visibility_cache),VISIBILITY_CACHE_LIMIT)
        self.assertNotIn(((0,0),(0,0)),w._visibility_cache)
        self.assertIn(((0,0),(VISIBILITY_CACHE_LIMIT+9,0)),w._visibility_cache)
        saved=w.checkpoint();self.assertNotIn('_visibility_cache',saved)
        restored=World.restore(json.loads(json.dumps(saved)))
        self.assertFalse(hasattr(restored,'_visibility_cache'))
        self.assertEqual(w.observe_all(),restored.observe_all())

    def test_cached_and_uncached_observations_match_with_movement_births_and_deaths(self):
        c=config();c['world']['max_steps']=12
        for s in c['species'].values():
            s['reproduction'].update(enabled=True,health_threshold=0,min_age_steps=2,cooldown_steps=3,
                age_deviation_pct=20,cooldown_deviation_pct=20,min_energy=10,energy_cost=5,max_population=8)
        a=World(c,42);b=World.restore(json.loads(json.dumps(a.checkpoint())))
        rng=random.Random(4)
        for tick in range(12):
            if tick==5:
                for w in (a,b):
                    target=next(iter(w.animals.values()));target.health=0.1;target.energy=0
            if tick==7:
                a.obstacles.clear();b.obstacles.clear()
            expected={i:reference_observation(a,x) for i,x in a.animals.items()}
            self.assertEqual(a.observe_all(),expected)
            with patch.object(b,'_perception_visible',side_effect=b.visible):
                self.assertEqual(b.observe_all(),expected)
                actions={i:rng.choice(c['species'][x.species]['allowed_actions']) for i,x in a.animals.items()}
                b.step(actions)
            a.step(actions)
            self.assertEqual(a.checkpoint(),b.checkpoint())

    def test_only_final_champions_get_detached_model_copies(self):
        c=config();animals=[];brains={}
        for i in range(40):
            a=animal(i,'prey',i,0,health=i+1);a.age=i+1;a.kills=i+1;a.energy_gained=i+1
            animals.append(a);brains[i]=create_brain(c['brains']['small_q'],['rest','eat'],i,2)
        book=ChampionBook(c)
        with patch('ecosystem.library.model_snapshot',wraps=model_snapshot) as snapshot:
            book.consider(animals,brains,{'tick':40})
            snapshot.assert_called_once_with(brains[39])
        for entry in book.entries['prey'].values():self.assertEqual(entry['source']['animal_id'],39)
        before=copy.deepcopy(book.entries)
        brains[39].weights['b1'][0]+=1
        self.assertEqual(before,book.entries)

    def test_champion_selection_matches_former_intermediate_snapshot_path(self):
        c=config();rng=random.Random(13);animals=[];brains={}
        for i in range(30):
            species='prey' if i%2==0 else 'predator'
            a=animal(i,species,i,0);animals.append(a)
            brains[i]=create_brain(c['brains']['small_q'],['rest'],i,1)
        actual,expected=ChampionBook(c),ChampionBook(c)
        for tick in range(15):
            rng.shuffle(animals)
            for a in animals:
                a.age+=1;a.kills+=rng.randrange(2);a.energy_gained+=rng.random();a.health=rng.randrange(31)
                brains[a.id].weights['b2'][0]+=0.01
            actual.consider(animals,brains,{'tick':tick});old_consider(expected,animals,brains,{'tick':tick})
            self.assertEqual(actual.entries,expected.entries)

    def test_reproduction_window_settings_resolved_per_species_not_per_animal(self):
        c=config()
        for s in c['species'].values():s['reproduction'].update(enabled=True)
        w=World(c,42)
        with patch.object(w,'_has_reproduction_window',wraps=w._has_reproduction_window) as check:
            w.step({i:'rest' for i in w.animals})
            self.assertEqual(check.call_count,2)
        w.config['species']['prey']['reproduction']['age_deviation_pct']=10
        with patch.object(w,'_has_reproduction_window',wraps=w._has_reproduction_window) as check:
            w.step({i:'rest' for i in w.animals})
            self.assertEqual(check.call_count,2)

    def test_metrics_include_empty_species_and_current_manual_edits(self):
        w=empty_world();w.animals={0:animal(0,'prey',2,2),1:animal(1,'prey',3,2)}
        self.assertEqual(w.metrics()['populations'],{'prey':2,'predator':0})
        del w.animals[1]
        self.assertEqual(w.metrics()['populations'],{'prey':1,'predator':0})

    def test_all_algorithm_chains_match_the_former_champion_copy_path(self):
        for kind in ('recurrent_ppo','qrdqn','maskable_ppo','mappo'):
            a=Experiment(environment(kind));b=Experiment(environment(kind))
            while not a.done:
                a.step()
                with patch.object(ChampionBook,'consider',old_consider):b.step()
                self.assertEqual(a.checkpoint(),b.checkpoint(),kind)
