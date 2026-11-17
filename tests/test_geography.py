"""Optional geography: river vectors, swimming gates, slopes and exact learning."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ecosystem.analyze import analyze_recording
from ecosystem.brains import model_signature,load_model_weights,model_snapshot
from ecosystem.checkpoint_io import read_checkpoint
from ecosystem.config import validate,load
from ecosystem.contextual_ppo import action_mask
from ecosystem.geography import Geography
from ecosystem.grouped_rollouts import collect_group
from ecosystem.recording import Recorder,RecordingStore
from ecosystem.replay_io import load_chunk
from ecosystem.runner import Experiment
from ecosystem.telemetry import summarize
from ecosystem.world import World,Animal
from tests.test_mutation import KINDS,environment
from tests.reproduction_support import force_reproduction


def configuration(kind='q_network',rows=None,legend=None):
    c=environment(kind,1.0)
    rows=rows or ['....~....']*7
    c['world'].update(width=len(rows[0]),height=len(rows),max_steps=8)
    c['resources']['patch_count']=0
    c['terrain']=dict(map_version=1,name='test_map',width=len(rows[0]),height=len(rows),rows=rows,
        legend=legend or {'.':{},'#':{'blocked':True},'~':{'water_depth':1,'flow':[0,0]},'1':{'elevation':1},'3':{'elevation':3}})
    for b in c['brains'].values():b['observation_version']=2
    for name,s in c['species'].items():
        s['initial_population']=2 if s['diet']=='resource' else 0
        s['reproduction']['enabled']=False
        s['allowed_actions'].append('swim')
        s['locomotion']=dict(swim_min_health=10,swim_energy_cost=2,water_energy_cost=0,drowning_damage=2,uphill_energy_cost=3,max_climb=1)
    return c


def animal(i,x,y,heading=1,energy=40,health=30,species='prey',sex='female'):
    return Animal(i,species,x,y,heading,energy,health,age=10,sex=sex,last_action='reproduce')


def world(rows=None,legend=None):
    w=World(validate(configuration(rows=rows,legend=legend)),42)
    w.animals={0:animal(0,3,2)};w.next_animal_id=1
    return w


def step(w,**actions):
    chosen={i:'rest' for i in w.animals};chosen.update({int(i):a for i,a in actions.items()})
    return w.step(chosen)


class TerrainPhysicsTests(unittest.TestCase):
    def test_swim_required_to_enter_and_leave_water_and_only_near_water(self):
        w=world();step(w,**{'0':'move_forward'})
        self.assertEqual(w.animals[0].position,(3,2));self.assertEqual(w.animals[0].energy,40)
        self.assertEqual(w.events[0].cause,'requires_swim')
        step(w,**{'0':'swim'});self.assertEqual(w.animals[0].position,(4,2));self.assertEqual(w.animals[0].energy,38)
        step(w,**{'0':'move_forward'});self.assertEqual(w.animals[0].position,(4,2))
        step(w,**{'0':'swim'});self.assertEqual(w.animals[0].position,(5,2));self.assertEqual(w.animals[0].energy,36)
        step(w,**{'0':'swim'});self.assertEqual(w.animals[0].position,(5,2));self.assertEqual(w.events[0].cause,'swim:not_near_water')

    def test_health_depth_and_energy_gates_do_not_charge_failed_strokes(self):
        for health,energy,reason in ((9,40,'health'),(30,2,'energy')):
            w=world();a=w.animals[0];a.health=health;a.energy=energy
            step(w,**{'0':'swim'})
            self.assertEqual(a.position,(3,2));self.assertEqual(a.energy,energy);self.assertEqual(w.events[0].cause,'swim:'+reason)
        w=world();w.animals[0].health=10;step(w,**{'0':'swim'});self.assertEqual(w.animals[0].position,(4,2))
        c=configuration();c['terrain']['legend']['~']['water_depth']=2
        w=World(validate(c),42);w.animals={0:animal(0,3,2,health=19)}
        step(w,**{'0':'swim'});self.assertEqual(w.animals[0].position,(3,2))

    def test_fractional_vector_currents_accumulate_both_components_and_negative_flow(self):
        for flow,start,expected in (([.25,.5],(2,1),(3,3)),([-.5,-.25],(5,4),(3,3))):
            legend={'.':{},'~':{'water_depth':1,'flow':flow}}
            w=world(['.~~~~~~~.']*7,legend);w.animals[0].x,w.animals[0].y=start
            for _ in range(4):step(w)
            self.assertEqual(w.animals[0].position,expected)
            self.assertEqual((w.animals[0].drift_x,w.animals[0].drift_y),(0,0))
        w=world(['.~~~~~~~.']*7,{'.':{},'~':{'water_depth':1,'flow':[.1,0]}});w.animals[0].x=2
        for _ in range(10):step(w)
        self.assertEqual(w.animals[0].position,(3,2))

    def test_flow_pushes_to_opposite_bank_and_stops_on_land(self):
        w=world(legend={'.':{},'~':{'water_depth':1,'flow':[3,0]}})
        step(w,**{'0':'swim'})
        self.assertEqual(w.animals[0].position,(5,2));self.assertEqual(w.animals[0].drift_x,0)
        self.assertEqual(sum(e.kind=='drifted' for e in w.events),1)

    def test_fast_flow_cannot_tunnel_walls_or_cut_diagonal_corners(self):
        rows=['.~~~~~~~.']*7;rows[2]='.~~~#~~~.'
        legend={'.':{},'#':{'blocked':True},'~':{'water_depth':1,'flow':[3,0]}}
        w=world(rows,legend);w.animals[0].x=2;step(w)
        self.assertEqual(w.animals[0].position,(3,2));self.assertTrue(any(e.kind=='drift_blocked' for e in w.events))
        w=world(['.~~~~~~~.']*7,{'.':{},'~':{'water_depth':1,'flow':[1,1]}})
        w.animals[0].x=2;w.obstacles.add((3,2));step(w)
        self.assertEqual(w.animals[0].position,(2,2))

    def test_flow_occupancy_and_contention_have_no_swaps_or_overlaps(self):
        w=world(['.~~~~~~~.']*7,{'.':{},'~':{'water_depth':1,'flow':[1,0]}})
        w.animals={0:animal(0,2,2),1:animal(1,3,2)};step(w)
        self.assertEqual(w.animals[0].position,(2,2));self.assertEqual(w.animals[1].position,(4,2))
        rows=['.........']*7;rows[2]='..>~<....'
        legend={'.':{},'>':{'water_depth':1,'flow':[1,0]},'<':{'water_depth':1,'flow':[-1,0]},'~':{'water_depth':1}}
        w=world(rows,legend);w.animals={0:animal(0,2,2),1:animal(1,4,2)}
        clone=World.restore(json.loads(json.dumps(w.checkpoint())))
        step(w);step(clone)
        self.assertEqual(w.checkpoint(),clone.checkpoint());self.assertEqual(len({a.position for a in w.animals.values()}),2)
        self.assertEqual(sum(a.position==(3,2) for a in w.animals.values()),1)

    def test_uphill_charges_success_only_and_cliffs_require_a_ramp(self):
        rows=['.........']*7;rows[2]='..1......'
        w=world(rows);w.animals[0].x=1;step(w,**{'0':'move_forward'})
        self.assertEqual(w.animals[0].energy,37);self.assertEqual(w.animals[0].position,(2,2))
        self.assertTrue(any(e.kind=='energy_spent' and e.amount==3 for e in w.events))
        w.animals[0].heading=3;step(w,**{'0':'move_forward'});self.assertEqual(w.animals[0].energy,37)
        rows[2]='...3.....';w=world(rows);w.animals[0].x=2
        step(w,**{'0':'move_forward'});self.assertEqual(w.animals[0].position,(2,2));self.assertEqual(w.events[0].cause,'slope')
        w=world(['..1......']*7);w.animals[0].x=1;w.animals[0].energy=3
        step(w,**{'0':'move_forward'});self.assertEqual(w.animals[0].position,(1,2));self.assertEqual(w.animals[0].energy,3)

    def test_drowning_and_water_energy_are_separate_from_attack_kills(self):
        w=world();w.animals[0].x=4;w.animals[0].health=5
        w.config['species']['prey']['locomotion']['water_energy_cost']=.5
        step(w);self.assertEqual(w.animals[0].energy,39.5);self.assertEqual(w.animals[0].health,3)
        step(w);r=step(w);self.assertIn(0,r.dead_ids);self.assertEqual(w.deaths['drowning'],1)
        self.assertEqual(next(e.position for e in r.events if e.kind=='died'),[4,2])
        w=world();w.animals={0:animal(0,4,2,health=6),1:animal(1,3,2,species='predator')}
        w.config['species']['predator']['body']['attack_damage']=5
        r=step(w,**{'1':'attack'})
        self.assertIn(0,r.dead_ids);self.assertEqual(w.deaths['drowning'],1);self.assertFalse(any(e.kind=='kill' for e in r.events))

    def test_observations_and_masks_expose_water_flow_and_slope_locally(self):
        w=world();a=w.animals[0];o=w.observe(a)
        self.assertEqual(len(o),32);self.assertEqual(o[4],1);self.assertEqual(o[24],.1)
        self.assertEqual(action_mask(w,a,['rest','move_forward','swim']),(True,False,True))
        a.health=9;self.assertEqual(action_mask(w,a,['rest','swim']),(True,False))
        a.health=30;a.x=4
        self.assertEqual(w.observe(a)[20],.1)
        rows=['.........']*7;rows[2]='...3.....';w=world(rows);w.animals[0].x=2
        self.assertEqual(w.observe(w.animals[0])[25],.03)

    def test_births_cannot_cross_water_cliffs_or_secluded_wall_corners(self):
        c=configuration();c['species']['prey']['reproduction']['enabled']=True
        w=World(validate(c),42);w.animals={0:animal(0,3,2),1:animal(1,5,2,sex='male')};w.next_animal_id=2
        events=[];w._reproduce(events);self.assertFalse(events)
        rows=['.........']*7;rows[2]='...3.....';c=configuration(rows=rows);c['species']['prey']['reproduction']['enabled']=True
        w=World(validate(c),42);w.animals={0:animal(0,2,2),1:animal(1,3,2,sex='male')};w.next_animal_id=2
        w._reproduce(events);self.assertFalse(events)
        rows=['#####','#..##','#..##','###..','###..'];c=configuration(rows=rows)
        c['species']['prey']['reproduction'].update(enabled=True,mate_radius=3)
        w=World(validate(c),42);w.animals={0:animal(0,2,2),1:animal(1,3,3,sex='male')};w.next_animal_id=2
        w._reproduce(events);self.assertFalse(events)


class TerrainLearningAndStorageTests(unittest.TestCase):
    def test_every_algorithm_handles_mutated_births_and_resumes_32_inputs_exactly(self):
        for kind in KINDS:
            c=configuration(kind);c['species']['prey']['initial_population']=3;c['species']['predator']['initial_population']=2
            for s in c['species'].values():s['reproduction']['enabled']=True
            a=Experiment(c)
            positions=[(1,1),(2,1),(1,2),(6,2),(7,2)]
            for body,p in zip(a.world.animals.values(),positions):body.x,body.y=p
            a.observations=a.world.observe_all();a._refresh_context();force_reproduction(a);a.step()
            children=[body for body in a.world.animals.values() if body.parent_id>=0]
            self.assertTrue(children,kind);self.assertTrue(all(body.mutated for body in children))
            child=children[0];child.x,child.y=4,2;child.drift_x=.25;child.drift_y=.5
            a.observations=a.world.observe_all();a._refresh_context()
            b=Experiment.restore(json.loads(json.dumps(a.checkpoint())))
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'state.bin.xz';a.write_checkpoint(path);saved=read_checkpoint(path);z=Experiment.restore(saved)
            while not a.done:
                a.step();b.step();z.step()
                self.assertEqual(a.checkpoint(),b.checkpoint(),kind);self.assertEqual(a.checkpoint(),z.checkpoint(),kind)
            self.assertEqual(model_signature(next(iter(a.brains.values())).config,next(iter(a.brains.values())).actions)['inputs'],32)

    def test_last_terrain_feature_reaches_every_algorithm_network(self):
        for kind in KINDS:
            e=Experiment(configuration(kind));brain=e.brains[0]
            x=[0.0]*32;y=x.copy();y[-1]=1
            if kind=='q_network':a,b=brain.forward(x)[1],brain.forward(y)[1]
            elif kind=='qrdqn':a,b=brain.values(x),brain.values(y)
            else:a,b=brain.action_probabilities(x),brain.action_probabilities(y)
            self.assertNotEqual(a,b,kind)

    def test_all_algorithms_update_weights_connected_to_the_last_terrain_input(self):
        from ecosystem.brain import Experience
        from ecosystem.policy_gradients import PolicyStep
        from ecosystem.ppo import policy_distribution
        def input_weights(b):
            if b.config['type']=='q_network':return b.weights['w1']
            layer=b.network.layers[0]
            return layer.get('input_weights',layer.get('weights'))
        for kind in KINDS:
            e=Experiment(configuration(kind));b=e.brains[0];o=(0.0,)*31+(1.0,)
            before=input_weights(b)[31]
            if kind=='grpo':
                logs=policy_distribution(b.network.predict(o))[1]
                b.update_group([[PolicyStep(o,0,1,logs[0],True,False)], [PolicyStep(o,1,0,logs[1],True,False)]])
            else:
                for _ in range(4):
                    action=b.choose(o)
                    b.learn(Experience(o,action,1,None,True))
            self.assertGreater(b.updates,0,kind)
            self.assertNotEqual(input_weights(b)[31],before,kind)

    def test_saved_map_is_embedded_and_resume_does_not_read_deleted_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=configuration();path=root/'map.json';path.write_text(json.dumps(c['terrain']))
            c['terrain']={'map_file':str(path)};env=root/'env.json';env.write_text(json.dumps(c))
            a=Experiment(load(env));a.step();state=json.loads(json.dumps(a.checkpoint()));path.unlink()
            b=Experiment.restore(state)
            while not a.done:a.step();b.step()
            self.assertEqual(a.checkpoint(),b.checkpoint());self.assertIn('rows',a.config['terrain'])

    def test_grpo_trials_isolate_geography_mutation_and_current_carry(self):
        e=Experiment(configuration('grpo'));e.world.animals[0].x=4;e.world.animals[0].drift_x=.3
        e.observations=e.world.observe_all();before=e.world.checkpoint()
        others={i:copy.deepcopy(b.checkpoint()) for i,b in e.brains.items() if i!=0}
        group=collect_group(e.world,e.brains,0)
        self.assertTrue(group);self.assertEqual(e.world.checkpoint(),before)
        self.assertEqual({i:b.checkpoint() for i,b in e.brains.items() if i!=0},others)
        with self.assertRaises(TypeError):e.world.geography.water[(1,1)]=(1,0,0)
        shared=World.restore(e.world.checkpoint(),static_geography=e.world.geography)
        self.assertIs(shared.geography,e.world.geography)

    def test_old_20_feature_models_cannot_silently_initialize_terrain_brains(self):
        e=Experiment(configuration());model=model_snapshot(e.brains[0]);model['signature']['inputs']=20;model['signature']['observation_version']=1
        with self.assertRaisesRegex(ValueError,'incompatible'):load_model_weights(e.brains[0],model)

    def test_recording_stores_map_once_and_telemetry_counts_swimming_and_costs(self):
        c=configuration();c['recording'].update(encoding='binary',compression='xz',frames_per_chunk=2,diagnostics_every_steps=1)
        e=Experiment(c);e.world.animals[0].x,e.world.animals[0].y,e.world.animals[0].heading=3,2,1
        e.world.animals[1].x,e.world.animals[1].y,e.world.animals[1].heading=1,5,0
        e.observations=e.world.observe_all()
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'terrain';r=Recorder(folder,e.config,e.mode);r.begin_run(e)
            with patch.object(type(e.brains[0]),'choose',return_value='swim'):e.step()
            r.capture(e);r.close(False);store=RecordingStore(d);frames=load_chunk(store.chunk('terrain',1,0))
            self.assertNotIn('terrain',frames[-1]);self.assertIn('rows',store.metadata('terrain')['config']['terrain'])
            stat=summarize(c,frames[-1],frames[0])['species']['prey']
            self.assertEqual(stat['swim_attempts'],2);self.assertEqual(stat['terrain_energy'],2)
            report=analyze_recording(folder)['runs'][0]['species']['prey']
            self.assertEqual(report['swim_attempts'],2);self.assertEqual(report['terrain_energy'],2)
            self.assertEqual(report['swim_success'],.5)
            from ecosystem.replay_io import read_chunk_bytes
            import base64
            script="""import assert from 'node:assert/strict';
            import {decodeBinaryReplay} from './viewer/binary-codec.mjs';
            import {decodeChunk} from './viewer/replay-codec.mjs';
            let input='';for await(const x of process.stdin)input+=x;
            const d=JSON.parse(input),frames=decodeChunk(decodeBinaryReplay(Buffer.from(d.bytes,'base64')));
            assert.deepStrictEqual(frames,d.frames);
            assert.equal(frames.at(-1).diagnostics['0'].features.length,32);
            """
            payload=dict(bytes=base64.b64encode(read_chunk_bytes(store.chunk('terrain',1,0))).decode(),frames=frames)
            result=subprocess.run(['node','--input-type=module','-e',script],input=json.dumps(payload),text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)


    def test_cli_terrain_presets_capture_maps_and_match_environment_actions(self):
        signatures={}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ('river_crossing','highlands','enclosed_valleys'):
                c=load('environments/'+name+'.json')
                signatures[name]=[model_signature(c['brains'][s['brain']],s['allowed_actions']) for s in c['species'].values()]
                result=subprocess.run([sys.executable,'-m','ecosystem.cli','environments/'+name+'.json','--steps','4','--runs','1',
                    '--no-library','--no-progress','--output',str(root/name)],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                folder=next((root/name).iterdir())
                manifest=json.loads((folder/'manifest.json').read_text())
                saved=json.loads((folder/'environment.json').read_text())
                self.assertEqual(manifest['observation_version'],2)
                self.assertIn('rows',saved['terrain']);self.assertNotIn('map_file',saved['terrain'])
                header=(folder/'metrics.csv').read_text().splitlines()[0]
                self.assertIn('drowning',header)
            self.assertNotEqual(signatures['river_crossing'],signatures['highlands'])
            self.assertEqual(signatures['river_crossing'],signatures['enclosed_valleys'])

    def test_terrain_presets_and_invalid_geography_are_validated(self):
        for name in ('river_crossing','highlands','enclosed_valleys'):
            c=load('environments/'+name+'.json');self.assertEqual(c['terrain']['name'],name)
            for species in c['species'].values():
                self.assertEqual('swim' in species['allowed_actions'],name!='highlands')
            self.assertTrue(all(b.get('observation_version')==2 for b in c['brains'].values()))
            e=Experiment(c);self.assertTrue(all(not e.world.geography.depth(a.position) for a in e.world.animals.values()))
            self.assertTrue(all(not e.world.geography.depth(p) for p in e.world.resources))
        for change in (lambda c:c['terrain']['rows'].pop(),lambda c:c['terrain']['legend']['~'].update(flow=[5,0]),
            lambda c:c['terrain']['legend']['.'].update(flow=[1,0]),lambda c:c['terrain']['legend']['~'].update(water_depth=-1),
            lambda c:c['terrain'].update(width=99),lambda c:c['brains'][c['species']['prey']['brain']].update(observation_version=1),
            lambda c:c['species']['prey']['locomotion'].update(max_climb=-1)):
            c=configuration();change(c)
            with self.assertRaises(ValueError):validate(c)


if __name__=='__main__':unittest.main()
