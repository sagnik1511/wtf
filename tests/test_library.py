"""Champion snapshots, compatible warm starts and ancestry without replay copies."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from ecosystem.brains import load_model_weights, model_snapshot
from ecosystem.config import validate
from ecosystem.library import ChampionBook, catalog, export_champions, apply_brain_overrides, localize_champions
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
        self.assertEqual(book.entries['prey']['female']['health']['source']['animal_id'],2)
        self.assertEqual(book.entries['prey']['female']['lifespan']['source']['animal_id'],1)
        self.assertEqual(book.entries['prey']['female']['food']['source']['animal_id'],1)
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
        candidate=e.champions.entries['prey'][child.sex]['lifespan']
        self.assertEqual(candidate['source']['animal_id'],child.id)
        self.assertEqual(candidate['score'],101)
        self.assertFalse(candidate['source']['alive'])
        self.assertEqual(candidate['model'],model_snapshot(brain))
        self.assertEqual(e.champions.entries['prey'][child.sex]['kills']['score'],7)

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
                c['species']['prey']['initial_brain']={'path':str(archive/'prey'/'female'/'lifespan.json')}
                e2=Experiment(c,library_root=root/'library')
                expected=e.champions.entries['prey']['female']['lifespan']['model']
                for i,a in e2.world.animals.items():
                    if a.species!='prey' or a.sex!='female':continue
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
            self.assertTrue((first/'prey/female/lifespan.json').is_file())
            c=config();c['species']['prey']['initial_brain']={'selection':'best','metric':'lifespan'}
            best=Experiment(c,library_root=root/'library')
            self.assertEqual(best.initial_models['prey']['female']['provenance']['score'],100)
            c['species']['prey']['initial_brain']['selection']='latest'
            latest=Experiment(c,library_root=root/'library')
            self.assertEqual(latest.initial_models['prey']['female']['provenance']['score'],1)
            self.assertEqual(len(catalog(root/'library')),26)
            third=self.export(latest,root)
            artifact=json.loads((third/'prey/female/lifespan.json').read_text())
            self.assertEqual(artifact['model_generation'],2)
            self.assertEqual(artifact['parent_artifact']['artifact_id'],latest.initial_models['prey']['female']['provenance']['artifact_id'])

    def test_checkpoint_embeds_initial_models_and_resumes_without_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);seed=Experiment(config());seed.step();archive=self.export(seed,root)
            c=config();c['experiment']['retain_brains_between_runs']=False
            c['species']['prey']['initial_brain']={'path':str(archive/'prey/female/lifespan.json')}
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
            c['species']['prey']['initial_brain']={'path':str(archive/'prey/female/lifespan.json')}
            warmed=Experiment(c)
            self.assertEqual(warmed.brains[0].config['learning_rate'],.02)
            self.assertEqual(model_snapshot(warmed.brains[0]),e.champions.entries['prey']['female']['lifespan']['model'])

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
            c['species']['prey']['initial_brain']={'path':str(archive/'predator/female/lifespan.json')}
            with self.assertRaisesRegex(ValueError,'species'):
                Experiment(c)
            c['species']['prey']['initial_brain']={'path':str(archive/'prey/female/lifespan.json')}
            c['brains']['small_q']['hidden_layers']=[5]
            with self.assertRaisesRegex(ValueError,'incompatible'):
                Experiment(c)

    def test_source_experiment_cli_selection_and_fresh_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);e=Experiment(config());e.step()
            index=root/'previous'/'champions'/'run_0001.json'
            archive=export_champions(e,index,root=root/'library')
            c=config()
            apply_brain_overrides(c,['prey=lifespan','predator=fresh'],root/'previous')
            self.assertEqual(Path(c['species']['prey']['initial_brains']['female']['path']),archive/'prey/female/lifespan.json')
            self.assertNotIn('initial_brain',c['species']['predator'])
            warmed=Experiment(c)
            self.assertEqual(model_snapshot(warmed.brains[0]),e.champions.entries['prey']['female']['lifespan']['model'])
            defaults=config();apply_brain_overrides(defaults,source_experiment=root/'previous')
            self.assertTrue(defaults['species']['prey']['initial_brains']['female']['path'].endswith('lifespan.json'))
            self.assertTrue(defaults['species']['predator']['initial_brains']['female']['path'].endswith('kills.json'))

    def test_cli_selections_validate_species_duplicates_and_criteria(self):
        for requests in (['prey'],['unknown=food'],['prey='],['prey=food','prey=kills'],
                         ['prey=food'],['prey=best:unknown']):
            with self.subTest(requests=requests),self.assertRaises(ValueError):
                apply_brain_overrides(config(),requests)
        with self.assertRaises(ValueError):apply_brain_overrides(config(),source_run=1)
        c=config();apply_brain_overrides(c,['prey=best:food','predator=latest:lifespan'])
        self.assertEqual(c['species']['prey']['initial_brain'],{'selection':'best','metric':'food'})
        self.assertEqual(c['species']['predator']['initial_brain'],{'selection':'latest','metric':'lifespan'})

    def test_cli_applies_source_models_without_editing_environment(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);e=Experiment(config())
            export_champions(e,root/'previous/champions/run_0001.json',root=root/'library')
            path=root/'environment.json';path.write_text(json.dumps(config()))
            original=path.read_bytes()
            output=root/'results'
            subprocess.run([sys.executable,'-m','ecosystem.cli',str(path),'--runs','1',
                '--steps','1','--brains-from',str(root/'previous'),
                '--brain','prey=food','--brain','predator=kills',
                '--no-library','--no-record','--no-progress','--output',str(output)],
                check=True,capture_output=True,text=True)
            folder=next(output.iterdir())
            metadata=json.loads((folder/'manifest.json').read_text())
            self.assertEqual(metadata['initialized_from']['prey']['female']['criterion'],'food')
            self.assertEqual(metadata['initialized_from']['predator']['female']['criterion'],'kills')
            self.assertEqual(path.read_bytes(),original)
            failed=subprocess.run([sys.executable,'-m','ecosystem.cli','--resume',
                str(folder/'checkpoints/final.json'),'--brain','prey=fresh'],
                capture_output=True,text=True)
            self.assertEqual(failed.returncode,2)
            self.assertIn('cannot be used with --resume',failed.stderr)

    def test_cli_can_pin_an_earlier_source_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);e=Experiment(config())
            first=export_champions(e,root/'previous/champions/run_0001.json',root=root/'library')
            e.step()
            second=export_champions(e,root/'previous/champions/run_0002.json',root=root/'library')
            c=config();apply_brain_overrides(c,source_experiment=root/'previous')
            self.assertEqual(Path(c['species']['prey']['initial_brains']['female']['path']),second/'prey/female/lifespan.json')
            apply_brain_overrides(c,source_experiment=root/'previous',source_run=1)
            self.assertEqual(Path(c['species']['prey']['initial_brains']['female']['path']),first/'prey/female/lifespan.json')

    def test_run_local_models_move_with_the_experiment_folder(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);e=Experiment(config())
            index=root/'original/champions/run_0001.json'
            export_champions(e,index,label='original')
            self.assertEqual(json.loads(index.read_text())['model_directory'],'run_0001')
            shutil.move(str(root/'original'),str(root/'moved'))
            c=config();apply_brain_overrides(c,source_experiment=root/'moved')
            loaded=Experiment(c)
            self.assertEqual(model_snapshot(loaded.brains[0]),e.champions.entries['prey']['female']['lifespan']['model'])
            self.assertEqual(model_snapshot(loaded.brains[3]),e.champions.entries['predator']['female']['kills']['model'])

    def test_legacy_models_are_localized_without_changing_artifact_identity(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);e=Experiment(config())
            index=root/'previous/champions/run_0001.json'
            archive=export_champions(e,index,root=root/'library')
            original=(archive/'prey/female/food.json').read_bytes()
            local=localize_champions(index)
            self.assertEqual((local/'prey/female/food.json').read_bytes(),original)
            self.assertEqual(localize_champions(index),local)
            shutil.rmtree(root/'library')
            c=config();apply_brain_overrides(c,source_experiment=root/'previous')
            loaded=Experiment(c)
            self.assertEqual(model_snapshot(loaded.brains[0]),e.champions.entries['prey']['female']['lifespan']['model'])

    def test_source_run_name_resolves_under_project_results(self):
        from unittest.mock import patch
        from ecosystem import library
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);e=Experiment(config())
            export_champions(e,root/'results/example/champions/run_0001.json')
            with patch.object(library,'PROJECT',root):
                c=config();apply_brain_overrides(c,source_experiment='example')
                loaded=Experiment(c)
                self.assertEqual(model_snapshot(loaded.brains[0]),e.champions.entries['prey']['female']['lifespan']['model'])

    def test_cli_saves_self_contained_models_by_default(self):
        import subprocess,sys
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'environment.json';path.write_text(json.dumps(config()))
            subprocess.run([sys.executable,'-m','ecosystem.cli',str(path),'--runs','1',
                '--steps','1','--no-record','--no-progress','--output',str(root/'results')],
                check=True,capture_output=True,text=True)
            folder=next((root/'results').iterdir())
            self.assertEqual(len(list((folder/'champions/run_0001').glob('*/*/*.json'))),13)
            c=config();apply_brain_overrides(c,source_experiment=folder)
            self.assertTrue(all(Path(request['path']).is_file() for spec in c['species'].values() for request in spec['initial_brains'].values() if request))

    def test_export_failure_does_not_publish_partial_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);e=Experiment(config())
            e.champions.entries['prey']['female']['health']['model']['parameters']['w1'][0]=float('nan')
            with self.assertRaises(ValueError):self.export(e,root)
            self.assertEqual(catalog(root/'library'),[])
            self.assertEqual(list((root/'library').iterdir()),[])
