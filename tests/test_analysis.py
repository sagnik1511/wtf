"""Analysis uses exact denominators, inheritance baselines and coverage flags."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ecosystem.analyze import analyze_recording, analyze_run
from tests.test_simulation import config


def animal(i, species, age=0, action='rest', parent_id=-1):
    return dict(id=i, species=species, x=i+1, y=2, heading=0,
                energy=60, health=30, age=age, last_action=action,
                parent_id=parent_id, generation=int(parent_id >= 0))


def frame(tick, animals, events=(), diagnostics=None):
    return dict(tick=tick, animals=animals, events=list(events),
        metrics=dict(tick=tick, populations={s: sum(a['species']==s for a in animals)
                      for s in ('prey','predator')}, food=20, kills={}, births={}, deaths={}),
        diagnostics=diagnostics or {})


def fixture():
    return [
        frame(0,[animal(0,'prey'),animal(1,'predator')]),
        frame(1,[animal(0,'prey',1,'eat'),animal(1,'predator',1,'attack'),animal(2,'prey',parent_id=0)], [
            dict(kind='energy_gained',actor=0,amount=4,cause='resource'),
            dict(kind='damage_dealt',actor=1,amount=5),
            dict(kind='damage_taken',actor=0,amount=5),
            dict(kind='born',actor=0,target=2)]),
        frame(2,[animal(1,'predator',2,'attack'),animal(2,'prey',1,'move_forward',0)], [
            dict(kind='damage_dealt',actor=1,amount=25),
            dict(kind='damage_taken',actor=0,amount=25),
            dict(kind='kill',actor=1,target=0),
            dict(kind='energy_gained',actor=1,amount=40,cause='prey'),
            dict(kind='died',actor=0,cause='predation'),
            dict(kind='moved',actor=2)]),
    ]


class AnalysisTests(unittest.TestCase):
    def test_exact_rates_terminal_rewards_and_newborn_denominators(self):
        result=analyze_run(config(),fixture(),window_steps=1)
        prey,predator=result['species']['prey'],result['species']['predator']
        self.assertEqual(prey['agent_steps'],3)
        self.assertEqual(predator['agent_steps'],2)
        self.assertEqual(prey['observed_actions'],2)  # No birth-step action.
        self.assertAlmostEqual(prey['action_coverage'],2/3)
        self.assertEqual(predator['action_coverage'],1)
        self.assertEqual(prey['feeding_success'],1)
        self.assertEqual(prey['movement_success'],1)
        self.assertEqual(predator['hunt_success'],.5)
        self.assertEqual(predator['hit_success'],1)
        self.assertAlmostEqual(prey['energy_demand'],20 + .3 + .15)
        self.assertAlmostEqual(prey['energy_coverage'],4/20.45)
        self.assertAlmostEqual(prey['reward_per_agent_step'],(.4-6-10)/3)
        self.assertAlmostEqual(predator['reward_per_agent_step'],(30*.05+4)/2)
        self.assertEqual(prey['births'],1)
        self.assertEqual(prey['energy_budget']['net_reproduction_energy'],40)
        self.assertAlmostEqual(prey['energy_budget']['food_input_share'],4/124)
        self.assertEqual(prey['mean_completed_lifespan'],2)
        self.assertEqual(prey['survivors'],1)
        self.assertEqual(prey['mean_survivor_age'],1)
        self.assertEqual(result['coverage']['step_fraction'],1)
        self.assertEqual(result['windows'][0]['species']['prey']['agent_steps'],1)
        self.assertEqual(result['windows'][1]['species']['prey']['agent_steps'],2)

    def test_deceased_successful_action_is_recoverable(self):
        frames=fixture()[:2]
        frames[1]['animals']=[animal(1,'predator',1,'rest')]
        frames[1]['events']=[dict(kind='energy_gained',actor=0,amount=4,cause='resource'),
                              dict(kind='died',actor=0,cause='starvation')]
        prey=analyze_run(config(),frames)['species']['prey']
        self.assertEqual(prey['observed_actions'],1)
        self.assertEqual(prey['action_coverage'],1)
        self.assertEqual(prey['feeding_success'],1)

    def test_gaps_are_not_interpolated_and_missing_metrics_are_null(self):
        frames=[fixture()[0],fixture()[2]]
        report=analyze_run(config(),frames)
        self.assertEqual(report['coverage']['step_fraction'],0)
        for s in report['species'].values():
            self.assertEqual(s['agent_steps'],0)
            self.assertIsNone(s['energy_coverage'])
            self.assertIsNone(s['reward_per_agent_step'])
            self.assertIsNone(s['action_coverage'])
            self.assertEqual(s['completed_lives'],0)
        json.dumps(report,allow_nan=False)

    def test_inherited_counters_are_not_counted_as_child_learning(self):
        def diagnostic(updates,groups,zero=True):
            return dict(algorithm='grpo',updates=updates,rollout_updates=groups,
                        action_probabilities={'attack':.5,'rest':.5},entropy=.6931471805599453,
                        training_metrics=dict(zero_variance=zero,group_returns=[0,0] if zero else [0,1]))
        frames=[
            frame(0,[animal(1,'predator')],diagnostics={'1':diagnostic(10,5)}),
            frame(1,[animal(1,'predator',1),animal(2,'predator',parent_id=1)],
                  diagnostics={'1':diagnostic(12,6,False),'2':diagnostic(12,6,False)}),
            frame(2,[animal(1,'predator',2),animal(2,'predator',1,parent_id=1)],
                  diagnostics={'1':diagnostic(12,7),'2':diagnostic(14,7,False)}),
        ]
        learning=analyze_run(config(),frames)['species']['predator']['learning']
        self.assertEqual(learning['observed_optimizer_updates'],4)
        self.assertEqual(learning['observed_grpo_groups'],3)
        self.assertEqual(learning['classified_grpo_groups'],3)
        self.assertAlmostEqual(learning['grpo_zero_variance_fraction'],1/3)
        self.assertEqual(learning['grpo_positive_return_groups'],2)
        self.assertAlmostEqual(learning['normalized_entropy'],1)

    def test_sampled_grpo_group_jumps_classify_only_the_last_group(self):
        frames=[frame(0,[animal(1,'predator')],diagnostics={'1':dict(algorithm='grpo',updates=0,rollout_updates=0)}),
                frame(1,[animal(1,'predator',1)],diagnostics={'1':dict(algorithm='grpo',updates=2,rollout_updates=3,
                       training_metrics=dict(zero_variance=True,group_returns=[0,0]))})]
        learning=analyze_run(config(),frames)['species']['predator']['learning']
        self.assertEqual(learning['observed_grpo_groups'],3)
        self.assertEqual(learning['classified_grpo_groups'],1)

    def test_zero_step_recording_has_valid_window_bounds(self):
        report=analyze_run(config(),fixture()[:1])
        self.assertEqual(report['windows'][0]['start_tick'],0)
        self.assertEqual(report['windows'][0]['end_tick'],0)
        self.assertIsNone(report['coverage']['step_fraction'])

    def test_invalid_window_and_nonincreasing_ticks(self):
        for size in (0,-1,True,1.5):
            with self.assertRaises(ValueError):
                analyze_run(config(),fixture(),size)
        with self.assertRaises(ValueError):
            analyze_run(config(),[fixture()[0],fixture()[0]])

    def save(self,folder,frames,run=1):
        playback=folder/'playback';playback.mkdir(parents=True)
        metadata=dict(recording_version=1,name='test',mode='train',config=config(),
            runs=[dict(run=run,seed=42,first_tick=frames[0]['tick'],last_tick=frames[-1]['tick'],
                       chunk_count=len(frames),frame_count=len(frames),complete=True)])
        (playback/'recording.json').write_text(json.dumps(metadata))
        for index,f in enumerate(frames):
            (playback/f'run_{run:04d}_chunk_{index:06d}.json').write_text(json.dumps([f]))
        return playback

    def test_chunks_stream_once_and_analysis_does_not_modify_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'example';self.save(folder,fixture())
            before={p:p.read_bytes() for p in folder.rglob('*') if p.is_file()}
            report=analyze_recording(folder)
            self.assertEqual(report['runs'][0]['frames'],3)
            self.assertEqual(report['runs'][0]['species']['prey']['births'],1)
            self.assertEqual(before,{p:p.read_bytes() for p in folder.rglob('*') if p.is_file()})
            self.assertFalse((folder/'analysis.json').exists())
            self.assertEqual(analyze_recording(folder/'playback'),report)
            with self.assertRaises(FileNotFoundError):
                analyze_recording(folder,run=2)

    def test_long_runs_bound_time_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'example'
            frames=[frame(i,[animal(0,'prey')]) for i in range(0,1000001,10000)]
            self.save(folder,frames)
            result=analyze_recording(folder,window_steps=1)['runs'][0]
            self.assertLessEqual(len(result['windows']),120)
            self.assertGreater(result['window_steps'],1)

    def test_offline_analysis_imports_no_simulation_or_learning_code(self):
        code="import sys; import ecosystem.analyze; assert not any(x in sys.modules for x in ('ecosystem.world','ecosystem.runner','ecosystem.brains','ecosystem.policy_gradients'))"
        subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)
