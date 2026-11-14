"""Experiment lifecycle, training, evaluation and exact JSON checkpoints."""
import copy
import random
import secrets
from .brain import Brain, Experience
from .config import validate
from .world import World


class Experiment:
    def __init__(self, config, mode='train'):
        if mode not in ('train', 'evaluate', 'random'):
            raise ValueError('mode must be train, evaluate, or random')
        self.config = copy.deepcopy(validate(config))
        self.mode = mode
        randomized = self.config['experiment']['world_seed_policy'] == 'random'
        self.randomization_seed = secrets.randbits(64) if randomized else None
        self.seed_rng = random.Random(self.randomization_seed) if randomized else None
        self.run_index = 0
        self.brains = {}
        self.history = []
        self.summaries = []
        self.run_seeds = []
        self.last_record = None
        self._start_run()

    def _start_run(self):
        c = self.config
        policy = c['experiment']['world_seed_policy']
        if policy == 'random':
            used = set(self.run_seeds)
            seed = self.seed_rng.getrandbits(32)
            while seed in used:
                seed = self.seed_rng.getrandbits(32)
        else:
            seed = (c['seed'] + (self.run_index if policy == 'increment' else 0)) % 2**32
        # Retained brains keep their learning and individual RNG states.
        # Newly initialized random-mode brains use the actual run seed.
        brain_seed = seed if policy == 'random' else c['seed']
        self.run_seeds.append(seed)
        self.world = World(c, seed)
        if not self.brains or not c['experiment']['retain_brains_between_runs']:
            self.brains = {}
            for species_index, (species, spec) in enumerate(c['species'].items()):
                base_seed = brain_seed + 1000003 * (species_index + 1)
                for i, a in self.world.animals.items():
                    if a.species == species:
                        self.brains[i] = Brain(c['brains'][spec['brain']], spec['allowed_actions'], base_seed, brain_seed + i * 7919 + 13)
        self.observations = self.world.observe_all()
        self.finished_run = False
        self.history = [self.world.metrics()]
        self.last_record = None

    @property
    def done(self):
        return self.finished_run and self.run_index + 1 >= self.config['experiment']['runs']

    def step(self):
        if self.finished_run:
            if self.done:
                return self.snapshot()
            self.run_index += 1
            self._start_run()
        actions = {i: self.brains[i].choose(o, explore=self.mode == 'train', random_baseline=self.mode == 'random') for i, o in self.observations.items()}
        result = self.world.step(actions)
        limit = self.world.tick >= self.config['world']['max_steps']
        for i, observation in self.observations.items():
            if self.mode == 'train':
                self.brains[i].learn(Experience(observation, actions[i], result.rewards[i], result.observations.get(i), i in result.dead_ids, limit and i not in result.dead_ids))
        self.observations = result.observations
        self.last_record = None
        self.finished_run = limit or not self.world.animals
        if self.world.tick % self.config['experiment']['record_metrics_every_steps'] == 0 or self.finished_run:
            self.last_record = self.world.metrics()
            self.history.append(self.last_record)
            # Browser history is bounded; CLI streams all recorded rows to disk.
            self.history = self.history[-2000:]
        if self.finished_run:
            self.summaries.append(dict(run=self.run_index + 1, seed=self.world.seed, **self.world.metrics(), lifespans=copy.deepcopy(self.world.lifespans), survivors=[dict(id=a.id, species=a.species, age=a.age) for a in self.world.animals.values()]))
        return None

    def next_run(self):
        if not self.finished_run or self.done:
            return False
        self.run_index += 1
        self._start_run()
        return True

    def snapshot(self, selected=None):
        data = self.world.snapshot()
        data.update(run=self.run_index + 1, total_runs=self.config['experiment']['runs'], done=self.done, finished_run=self.finished_run, mode=self.mode, history=self.history, summaries=self.summaries, name=self.config['name'], species={k: {'diet': v['diet'], 'body': v['body']} for k, v in self.config['species'].items()}, total_updates=sum(b.updates for b in self.brains.values()))
        if selected in self.world.animals:
            brain = self.brains[selected]
            o = self.observations[selected]
            data['selected'] = dict(id=selected, features=o, action_values=dict(zip(brain.actions, brain.forward(o)[1])), epsilon=brain.epsilon, updates=brain.updates, replay_size=len(brain.replay))
        return data

    def checkpoint(self):
        return dict(checkpoint_version=1, randomization_seed=self.randomization_seed, seed_rng_state=self.seed_rng.getstate() if self.seed_rng else None, config=copy.deepcopy(self.config), mode=self.mode, run_index=self.run_index, world=self.world.checkpoint(), brains={str(i): b.checkpoint() for i, b in self.brains.items()}, history=copy.deepcopy(self.history), summaries=copy.deepcopy(self.summaries), run_seeds=self.run_seeds[:], finished_run=self.finished_run)

    @classmethod
    def restore(cls, state, mode=None):
        if state.get('checkpoint_version') != 1:
            raise ValueError('Unsupported checkpoint version')
        e = cls.__new__(cls)
        e.config = copy.deepcopy(validate(state['config']))
        e.mode = mode or state['mode']
        if e.mode not in ('train', 'evaluate', 'random'):
            raise ValueError('Invalid checkpoint mode')
        for key in ('run_index', 'history', 'summaries', 'run_seeds', 'finished_run'):
            setattr(e, key, copy.deepcopy(state[key]))
        e.randomization_seed = state.get('randomization_seed')
        e.seed_rng = None
        if e.config['experiment']['world_seed_policy'] == 'random':
            if state.get('seed_rng_state') is None:
                raise ValueError('Randomized checkpoint is missing the seed generator state')
            def tuples(value):
                return tuple(tuples(x) for x in value) if isinstance(value, (list, tuple)) else value
            e.seed_rng = random.Random(0)
            e.seed_rng.setstate(tuples(state['seed_rng_state']))
        e.world = World.restore(state['world'])
        e.brains = {int(i): Brain.restore(b) for i, b in state['brains'].items()}
        e.observations = e.world.observe_all()
        e.last_record = None
        return e
