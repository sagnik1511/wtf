from ecosystem.checkpoint_io import read_checkpoint
"""Run-to-run champion transfer and original-initialization controls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from ecosystem.brains import model_snapshot
from ecosystem.config import validate
from ecosystem.library import export_champions
from ecosystem.runner import Experiment
from tests.test_simulation import config
from tests.test_library import settings


class BrainChainingTests(unittest.TestCase):
    def finish(self, experiment):
        while not experiment.finished_run:
            experiment.step()

    def test_default_chains_same_criteria_for_all_algorithms(self):
        for kind in ('q_network','ppo','a2c','reinforce','grpo'):
            with self.subTest(kind=kind):
                c=settings(kind);c['experiment'].pop('chain_best_brains',None)
                e=Experiment(c);self.finish(e)
                expected={s:copy.deepcopy(e.champions.entries[s][metric]['model'])
                          for s,metric in [('prey','lifespan'),('predator','kills')]}
                count=e.total_updates
                e.next_run()
                for i,a in e.world.animals.items():
                    self.assertEqual(model_snapshot(e.brains[i]),expected[a.species])
                    self.assertEqual(e.brains[i].steps,0)
                    self.assertEqual(e.brains[i].updates,0)
                self.assertEqual(e.total_updates,count)
                self.assertIsNot(e.brains[0].rng,e.brains[1].rng)

    def test_disabled_chaining_restores_original_even_with_random_worlds(self):
        c=config();c['experiment'].update(chain_best_brains=False,world_seed_policy='random')
        e=Experiment(c);original=model_snapshot(e.brains[0]);self.finish(e)
        e.next_run()
        self.assertEqual(model_snapshot(e.brains[0]),original)
        self.assertEqual(e.brains[0].steps,0)

    def test_loaded_metric_and_archive_parent_follow_previous_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);e=Experiment(config());self.finish(e)
            archive=export_champions(e,root/'source/champions/run_0001.json')
            for chain in (True,False):
                c=config();c['experiment']['chain_best_brains']=chain
                c['species']['prey']['initial_brain']={'path':str(archive/'prey/health.json')}
                a=Experiment(c);original=copy.deepcopy(a.initial_models['prey']['model'])
                self.finish(a)
                saved=export_champions(a,root/f'new_{chain}/champions/run_0001.json')
                prior=json.loads((saved/'prey/health.json').read_text())
                expected=copy.deepcopy(a.champions.entries['prey']['health']['model']) if chain else original
                a.next_run()
                self.assertEqual(model_snapshot(a.brains[0]),expected)
                if chain:
                    self.assertEqual(a.run_initial_models['prey']['provenance']['artifact_id'],prior['artifact_id'])
                    self.finish(a)
                    next_archive=export_champions(a,root/'next/champions/run_0002.json')
                    exported=json.loads((next_archive/'prey/health.json').read_text())
                    self.assertEqual(exported['parent_artifact']['artifact_id'],prior['artifact_id'])
                    self.assertEqual(exported['model_generation'],prior['model_generation']+1)

    def test_checkpoints_resume_across_boundary_both_modes(self):
        for chain in (True,False):
            c=config();c['experiment']['chain_best_brains']=chain
            a=Experiment(c);a.step()
            b=Experiment.restore(json.loads(json.dumps(a.checkpoint())))
            while not a.done:
                a.step();b.step()
            self.assertEqual(a.checkpoint(),b.checkpoint())

    def test_old_checkpoints_keep_legacy_individual_retention(self):
        for retain in (True,False):
            c=config();c['experiment']['retain_brains_between_runs']=retain
            a=Experiment(c);self.finish(a)
            saved=a.checkpoint();saved.pop('legacy_brain_retention')
            saved['config']['experiment'].pop('chain_best_brains',None)
            b=Experiment.restore(saved);original=b.brains[0]
            b.next_run()
            self.assertEqual(b.legacy_brain_retention,True)
            self.assertEqual(b.brains[0] is original,retain)

    def test_cli_three_runs_chain_archives_or_reuse_original_source(self):
        import subprocess,sys
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=Experiment(config())
            archive=export_champions(source,root/'source/champions/run_0001.json')
            original=json.loads((archive/'prey/health.json').read_text())
            environment=root/'environment.json';environment.write_text(json.dumps(config()))
            for chain in (True,False):
                output=root/f'output_{chain}'
                command=[sys.executable,'-m','ecosystem.cli',str(environment),
                    '--runs','3','--steps','3','--brains-from',str(root/'source'),
                    '--brain','prey=health','--brain','predator=lifespan',
                    '--no-record','--no-progress','--output',str(output)]
                if not chain:command.append('--no-chain-brains')
                subprocess.run(command,check=True,capture_output=True,text=True)
                folder=next(output.iterdir());parent=original['artifact_id']
                for number in (1,2,3):
                    artifact=json.loads((folder/f'champions/run_{number:04d}/prey/health.json').read_text())
                    self.assertEqual(artifact['parent_artifact']['artifact_id'],parent)
                    if chain:parent=artifact['artifact_id']
                checkpoint=read_checkpoint(folder/'checkpoints/final.json.gz')
                self.assertEqual(checkpoint['config']['experiment']['chain_best_brains'],chain)

    def test_flag_requires_boolean(self):
        c=config();c['experiment']['chain_best_brains']='false'
        with self.assertRaisesRegex(ValueError,'chain_best_brains'):
            validate(c)
