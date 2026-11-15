"""Champion snapshots, compatible warm starts and ancestry without replay copies."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from ecosystem.brains import load_model_weights, model_snapshot
from ecosystem.config import validate
from ecosystem.library import ChampionBook, catalog, export_champions
from ecosystem.runner import Experiment
from ecosystem.world import Animal, World
from tests.test_simulation import config
from tests.test_ppo import ppo_environment
from tests.test_policy_algorithms import environment
from tests.test_reproduction import breeding_experiment


def settings(kind):
    return config() if kind=='q_network' else ppo_environment() if kind=='ppo' else environment(kind)


class ChampionTests(unittest.TestCase):
    def test_independent_criteria_and_healthy_age_ties(self):
        e=Experiment(config())
        book=ChampionBook(e.config)
        animals=[Animal(0,'prey',1,1,0,60,30,age=5,energy_gained=2),
                 Animal(1,'prey',2,1,0,60,25,age=10,energy_gained=20),
                 Animal(2,'prey',3,1,0,60,30,age=8,energy_gained=1)]
        book.consider(animals,e.brains,dict(tick=10))
        self.assertEqual(book.entries['prey']['health']['source']['animal_id'],2)
        self.assertEqual(book.entries['prey']['lifespan']['source']['animal_id'],1)
        self.assertEqual(book.entries['prey']['food']['source']['animal_id'],1)
        saved=copy.deepcopy(book.entries)
        e.brains[1].weights['b2'][0]+=100
        animals[1].health=0
        self.assertEqual(book.entries,saved)
        self.assertNotIn('replay',json.dumps(book.entries))

    def test_terminal_descendant_is_saved_before_controller_retirement(self):
        e=breeding_experiment()
        e.step()
        child=next(a for a in e.world.animals.values() if a.parent_id==0)
        child.age=100;child.energy=0;child.health=.1;child.kills=7
        brain=e.brains[child.id]
        e.observations=e.world.observe_all();e.step()
        self.assertNotIn(child.id,e.brains)
        candidate=e.champions.entries['prey']['lifespan']
        self.assertEqual(candidate['source']['animal_id'],child.id)
        self.assertEqual(candidate['score'],101)
        self.assertFalse(candidate['source']['alive'])
        self.assertEqual(candidate['model'],model_snapshot(brain))
        self.assertEqual(e.champions.entries['prey']['kills']['score'],7)

    def test_lifetime_kills_and_energy_are_individual_and_children_start_at_zero(self):
        e=breeding_experiment()
        e.world.obstacles.clear()
        pred=e.world.animals[3];prey=e.world.animals[0]
        pred.x,pred.y,pred.heading=1,2,1;pred.energy=20
        prey.x,prey.y=2,2
        e.world.config['species']['predator']['body']['attack_damage']=30
        actions={i:'rest' for i in e.world.animals};actions[3]='attack'
        e.world.step(actions)
        self.assertEqual(pred.kills,1)
        self.assertEqual(pred.energy_gained,40)
        self.assertEqual(sum(a.kills for a in e.world.animals.values()),1)
        self.assertTrue(all(a.kills==0 and a.energy_gained==0 for a in e.world.animals.values() if a.parent_id>=0))

    def test_old_world_marks_partial_lifetime_statistics(self):
        e=Experiment(config());e.step()
        state=json.loads(json.dumps(e.world.checkpoint()))
        for a in state['snapshot']['animals']:
            for key in ('kills','energy_gained','statistics_since_age'):a.pop(key)
        w=World.restore(state)
        self.assertTrue(all(a.statistics_since_age==a.age==1 for a in w.animals.values()))


class LibraryTests(unittest.TestCase):
    def export(self,e,root,label='test'):
        return export_champions(e,root/'run'/'champions.json',root=root/'library',label=label)

    def test_all_algorithms_load_weights_with_fresh_training_state(self):
        for kind in ('q_network','ppo','a2c','reinforce','grpo'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);c=settings(kind);e=Experiment(c)
                for _ in range(4):e.step()
                archive=self.export(e,root)
                c['species']['prey']['initial_brain']={'path':str(archive/'prey'/'lifespan.json')}
                e2=Experiment(c,library_root=root/'library')
                expected=e.champions.entries['prey']['lifespan']['model']
                for i,a in e2.world.animals.items():
                    if a.species!='prey':continue
                    brain=e2.brains[i]
                    self.assertEqual(model_snapshot(brain),expected)
                    self.assertEqual((brain.steps,brain.updates),(0,0))
                    for buffer in ('replay','rollout','trajectory'):
                        if hasattr(brain,buffer):self.assertEqual(getattr(brain,buffer),[])
                    if kind=='q_network':
                        self.assertEqual(brain.target,brain.weights)
                        self.assertEqual(brain.epsilon,c['brains'][c['species']['prey']['brain']]['exploration']['initial'])
                    else:
                        self.assertTrue(all(v==0 for layer in brain.moments for values in layer.values() for v in values))
                    if kind=='grpo':
                        self.assertEqual(brain.reference_network.layers,brain.network.layers)
                self.assertIsNot(e2.brains[0].rng,e2.brains[1].rng)
                self.assertNotEqual(e2.brains[0].rng.getstate(),e2.brains[1].rng.getstate())
                original=model_snapshot(e2.brains[1])
                if kind=='q_network':e2.brains[0].weights['b2'][0]+=1
                else:e2.brains[0].network.layers[0]['bias'][0]+=1
                self.assertEqual(model_snapshot(e2.brains[1]),original)

    def test_best_and_latest_select_different_immutable_generations(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);e=Experiment(config())
            e.world.animals[0].age=100
            e.champions.consider(e.world.animals.values(),e.brains,e._champion_source())
            first=self.export(e,root)
            e=Experiment(config());e.step();second=self.export(e,root)
            self.assertNotEqual(first,second)
            self.assertTrue((first/'prey/lifespan.json').is_file())
            c=config();c['species']['prey']['initial_brain']={'selection':'best','metric':'lifespan'}
            best=Experiment(c,library_root=root/'library')
            self.assertEqual(best.initial_models['prey']['provenance']['score'],100)
            c['species']['prey']['initial_brain']['selection']='latest'
            latest=Experiment(c,library_root=root/'library')
            self.assertEqual(latest.initial_models['prey']['provenance']['score'],1)
            self.assertEqual(len(catalog(root/'library')),16)
            third=self.export(latest,root)
            artifact=json.loads((third/'prey/lifespan.json').read_text())
            self.assertEqual(artifact['model_generation'],2)
            self.assertEqual(artifact['parent_artifact']['artifact_id'],latest.initial_models['prey']['provenance']['artifact_id'])

    def test_checkpoint_embeds_initial_models_and_resumes_without_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);seed=Experiment(config());seed.step();archive=self.export(seed,root)
            c=config();c['experiment']['retain_brains_between_runs']=False
            c['species']['prey']['initial_brain']={'path':str(archive/'prey/lifespan.json')}
            a=Experiment(c);a.step()
            state=json.loads(json.dumps(a.checkpoint()))
            import shutil
            shutil.rmtree(root/'library')
            b=Experiment.restore(state)
            for _ in range(25):a.step();b.step()
            self.assertEqual(a.checkpoint(),b.checkpoint())

    def test_checkpoint_roundtrip_keeps_detached_champion_snapshots(self):
        a=breeding_experiment();a.step()
        b=Experiment.restore(json.loads(json.dumps(a.checkpoint())))
        for _ in range(6):a.step();b.step()
        self.assertEqual(a.checkpoint(),b.checkpoint())

    def test_weight_shape_nan_and_action_mismatches_are_rejected(self):
        e=Experiment(config());model=model_snapshot(e.brains[0])
        for mutate in (lambda m:m['parameters']['w1'].pop(),
                       lambda m:m['parameters']['w1'].__setitem__(0,float('nan')),
                       lambda m:m['signature']['actions'].reverse(),
                       lambda m:m['signature'].__setitem__('algorithm','ppo')):
            bad=copy.deepcopy(model);mutate(bad)
            with self.assertRaises(ValueError):load_model_weights(e.brains[0],bad)

    def test_training_hyperparameters_can_change_without_changing_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);e=Experiment(config());e.step();archive=self.export(e,root)
            c=config();c['brains']['small_q']['learning_rate']=.02
            c['species']['prey']['initial_brain']={'path':str(archive/'prey/lifespan.json')}
            warmed=Experiment(c)
            self.assertEqual(warmed.brains[0].config['learning_rate'],.02)
            self.assertEqual(model_snapshot(warmed.brains[0]),e.champions.entries['prey']['lifespan']['model'])

    def test_initialization_schema_is_valid_json(self):
        schema=json.loads((Path(__file__).resolve().parents[1]/'environment.schema.json').read_text())
        properties=schema['properties']['species']['additionalProperties']['properties']
        self.assertIn('initial_brain',properties)

    def test_config_rejects_ambiguous_or_unknown_selection(self):
        for request in ({'path':''},{'metric':'kills'}, {'selection':'x','metric':'food'},
                        {'selection':'best','metric':'damage'}, {'path':'x','metric':'food'}):
            c=config();c['species']['prey']['initial_brain']=request
            with self.subTest(request=request),self.assertRaises(ValueError):validate(c)

    def test_missing_and_incompatible_champions_fail_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c=config()
            c['species']['prey']['initial_brain']={'selection':'best','metric':'food'}
            with self.assertRaisesRegex(ValueError,'No compatible'):
                Experiment(c,library_root=root/'library')
            e=Experiment(config());archive=self.export(e,root)
            c['species']['prey']['initial_brain']={'path':str(archive/'predator/lifespan.json')}
            with self.assertRaisesRegex(ValueError,'species'):
                Experiment(c)
            c['species']['prey']['initial_brain']={'path':str(archive/'prey/lifespan.json')}
            c['brains']['small_q']['hidden_layers']=[5]
            with self.assertRaisesRegex(ValueError,'incompatible'):
                Experiment(c)

    def test_export_failure_does_not_publish_partial_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);e=Experiment(config())
            e.champions.entries['prey']['health']['model']['parameters']['w1'][0]=float('nan')
            with self.assertRaises(ValueError):self.export(e,root)
            self.assertEqual(catalog(root/'library'),[])
            self.assertEqual(list((root/'library').iterdir()),[])
