"""Physical state and simultaneous action resolution; no learning dependencies."""
import copy
import math
import random
from collections import defaultdict
from dataclasses import dataclass

DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


@dataclass
class Animal:
    id: int
    species: str
    x: int
    y: int
    heading: int
    energy: float
    health: float
    age: int = 0
    last_action: str = 'rest'
    last_reward: float = 0.
    parent_id: int = -1
    generation: int = 0
    last_reproduced_age: int = -1

    @property
    def position(self):
        return self.x, self.y


@dataclass
class Resource:
    x: int
    y: int
    quantity: float


@dataclass(frozen=True)
class Event:
    kind: str
    actor: int
    target: int = -1
    amount: float = 0.
    cause: str = ''


@dataclass
class StepResult:
    observations: dict
    rewards: dict
    dead_ids: set
    events: list


class World:
    def __init__(self, config, seed):
        self.config = copy.deepcopy(config)
        self.seed = seed
        self.tick = 0
        self.rng = random.Random(seed ^ 0x12345678)
        layout = random.Random(seed)
        w = config['world']
        self.width, self.height = w['width'], w['height']
        cells = [(x, y) for y in range(self.height) for x in range(self.width)]
        layout.shuffle(cells)
        o = config['obstacles']
        self.obstacles = set(map(tuple, o['explicit_cells'])) if o['generation'] == 'explicit' else set(cells[:int(len(cells) * o['occupied_fraction'])])
        free = [p for p in cells if p not in self.obstacles]
        r = config['resources']
        resource_cells = layout.sample(free, r['patch_count'])
        self.resources = {p: Resource(*p, r['initial_quantity_per_patch']) for p in resource_cells}
        layout.shuffle(free)
        self.animals = {}
        for species, s in config['species'].items():
            for _ in range(s['initial_population']):
                x, y = free.pop()
                i = len(self.animals)
                self.animals[i] = Animal(i, species, x, y, layout.randrange(4), s['body']['initial_energy'], s['body']['max_health'])
        self.next_animal_id = len(self.animals)
        self.births = defaultdict(int)
        self.events = []
        self.deaths = defaultdict(int)
        self.total_energy_gained = defaultdict(float)
        self.kills = defaultdict(int)
        self.lifespans = []

    def body(self, animal):
        return self.config['species'][animal.species]['body']

    def diet(self, animal):
        return self.config['species'][animal.species]['diet']

    def front(self, animal):
        dx, dy = DIRECTIONS[animal.heading]
        return animal.x + dx, animal.y + dy

    def blocked(self, p):
        x, y = p
        return not (0 <= x < self.width and 0 <= y < self.height) or p in self.obstacles

    def visible(self, start, end):
        # Integer Bresenham ray. The start cell does not occlude itself.
        x, y = start
        tx, ty = end
        dx, dy = abs(tx - x), -abs(ty - y)
        sx, sy = (1 if x < tx else -1), (1 if y < ty else -1)
        error = dx + dy
        while (x, y) != (tx, ty):
            e2 = 2 * error
            if e2 >= dy:
                error += dy
                x += sx
            if e2 <= dx:
                error += dx
                y += sy
            if (x, y) in self.obstacles:
                return False
        return True

    def _observation_context(self):
        occupied, food, prey, predators = set(), [], [], []
        species = self.config['species']
        for i, animal in self.animals.items():
            position = animal.position
            occupied.add(position)
            (prey if species[animal.species]['diet'] == 'resource' else predators).append((i, position))
        for position, patch in self.resources.items():
            if patch.quantity > .001:
                food.append(position)
        return occupied, food, prey, predators

    def observe(self, a, context=None):
        body = self.body(a)
        radius = body['vision_radius']
        occupied, food, prey, predators = self._observation_context() if context is None else context
        features = [a.energy / body['max_energy'], a.health / body['max_health'], *DIRECTIONS[a.heading]]
        for offset in range(4):
            dx, dy = DIRECTIONS[(a.heading + offset) % 4]
            p = a.x + dx, a.y + dy
            features.append(float(self.blocked(p) or p in occupied))
        radius_squared = radius * radius
        def nearest(points):
            best, best_squared = None, float('inf')
            for p in points:
                dx, dy = p[0] - a.x, p[1] - a.y
                squared = dx * dx + dy * dy
                if squared <= radius_squared and squared < best_squared and self.visible(a.position, p):
                    best, best_squared = p, squared
            # Keep hypot and division order identical to the reference model.
            return [0., 0., 0., 0.] if best is None else [(best[0] - a.x) / radius, (best[1] - a.y) / radius, math.hypot(best[0] - a.x, best[1] - a.y) / radius, 1.]
        features += nearest(food)
        features += nearest(p for i, p in prey if i != a.id)
        features += nearest(p for i, p in predators if i != a.id)
        return tuple(features)

    def observe_all(self):
        # Shared only within this phase: manual edits and future steps cannot
        # accidentally reuse stale positions or resource quantities.
        context = self._observation_context()
        return {i: self.observe(a, context) for i, a in self.animals.items()}

    def step(self, actions):
        if set(actions) != set(self.animals):
            raise ValueError('Exactly one action required for every living animal')
        for i, action in actions.items():
            if action not in self.config['species'][self.animals[i].species]['allowed_actions']:
                raise ValueError(f'Animal {i}: unsupported action {action}')
        events = []
        # Destinations occupied at the beginning of this phase remain blocked.
        # Swaps and following into a just-vacated cell are deliberately disallowed.
        occupied = {a.position for a in self.animals.values()}
        claims = defaultdict(list)
        for i, a in self.animals.items():
            action = actions[i]
            a.last_action = action
            if action == 'turn_left':
                a.heading = (a.heading - 1) % 4
            elif action == 'turn_right':
                a.heading = (a.heading + 1) % 4
            elif action == 'move_forward':
                destination = self.front(a)
                if self.blocked(destination) or destination in occupied:
                    events.append(Event('movement_blocked', i))
                else:
                    claims[destination].append(i)
        for destination, ids in claims.items():
            winner = self.rng.choice(ids)
            a = self.animals[winner]
            a.x, a.y = destination
            events.append(Event('moved', winner))
            events.extend(Event('movement_blocked', i, cause='competing_claim') for i in ids if i != winner)
        # Feeding uses seeded contention order; resources may share animal cells.
        eaters = [i for i in self.animals if actions[i] == 'eat']
        self.rng.shuffle(eaters)
        for i in eaters:
            a = self.animals[i]
            b = self.body(a)
            candidates = [self.resources.get(p) for p in (a.position, self.front(a))]
            patch = next((r for r in candidates if r is not None and r.quantity > 0), None)
            conversion = b['energy_per_food_unit']
            if patch is not None and conversion > 0:
                amount = min(patch.quantity, b['bite_size'], (b['max_energy'] - a.energy) / conversion)
                patch.quantity -= amount
                gained = amount * conversion
                a.energy += gained
                if gained > 0:
                    events.append(Event('energy_gained', i, amount=gained, cause='resource'))
        # Attacks are simultaneous. Multiple attackers share actual damage;
        # one seeded attacker receives energy for a kill, never multiple rewards.
        positions = {a.position: a for a in self.animals.values()}
        attacks = defaultdict(list)
        for i, a in self.animals.items():
            if actions[i] == 'attack':
                target = positions.get(self.front(a))
                if target is not None and self.diet(target) == 'resource' and self.body(a)['attack_damage'] > 0:
                    attacks[target.id].append(i)
        for target_id, ids in attacks.items():
            target = self.animals[target_id]
            attempted_damage = sum(self.body(self.animals[i])['attack_damage'] for i in ids)
            damage = min(target.health, attempted_damage)
            # Attribute only actual damage, proportional to each attacker's
            # contribution. Overkill cannot multiply the hit reward.
            for i in ids:
                contribution = damage * (self.body(self.animals[i])['attack_damage'] / attempted_damage)
                events.append(Event('damage_dealt', i, target_id, contribution, 'attack'))
            target.health -= damage
            events.append(Event('damage_taken', target_id, amount=damage, cause='attack'))
            if target.health <= 0:
                killer = self.animals[self.rng.choice(ids)]
                b = self.body(killer)
                gained = min(b['energy_per_kill'], b['max_energy'] - killer.energy)
                killer.energy += gained
                events.append(Event('kill', killer.id, target_id))
                events.append(Event('energy_gained', killer.id, target_id, gained, 'prey'))
        for i, a in self.animals.items():
            b = self.body(a)
            cost = b['living_cost']
            if actions[i] == 'move_forward':
                cost += b['movement_cost']
            if actions[i] == 'attack':
                cost += b['attack_cost']
            a.energy = max(0., a.energy - cost)
            a.age += 1
            if a.energy <= 0 and a.health > 0:
                damage = min(a.health, b['starvation_damage'])
                a.health -= damage
                events.append(Event('damage_taken', i, amount=damage, cause='starvation'))
        dead = {i for i, a in self.animals.items() if a.health <= 0}
        attack_deaths = set(attacks) & dead
        for i in sorted(dead):
            a = self.animals[i]
            cause = 'predation' if i in attack_deaths else 'starvation'
            events.append(Event('died', i, cause=cause))
            self.deaths[cause] += 1
            self.lifespans.append({'id': i, 'species': a.species, 'age': a.age, 'cause': cause})
        rewards = {i: 0. for i in self.animals}
        for e in events:
            a = self.animals[e.actor]
            r = self.config['species'][a.species]['rewards']
            if e.kind == 'energy_gained':
                rewards[e.actor] += e.amount * r['per_energy_gained']
                self.total_energy_gained[a.species] += e.amount
            elif e.kind == 'damage_dealt':
                rewards[e.actor] += e.amount * r.get('per_damage_dealt', 0.)
            elif e.kind == 'damage_taken':
                rewards[e.actor] += e.amount * r['per_damage_taken']
            elif e.kind == 'died':
                rewards[e.actor] += r['death']
            elif e.kind == 'kill':
                self.kills[a.species] += 1
        for i, a in self.animals.items():
            a.last_reward = rewards[i]
        for i in dead:
            del self.animals[i]
        self._reproduce(events)
        r = self.config['resources']
        for patch in self.resources.values():
            patch.quantity = min(r['capacity_per_patch'], patch.quantity + r['growth_per_step'])
        self.tick += 1
        self.events = events
        return StepResult(self.observe_all(), rewards, dead, events)

    def _reproduce(self, events):
        # Resolve after combat/starvation. Only animals present before this
        # phase can parent; newborns first act on the next simulation step.
        parents = list(self.animals.values())
        eligible = []
        counts = defaultdict(int)
        for a in parents:
            counts[a.species] += 1
            r = self.config['species'][a.species].get('reproduction')
            if (r and r['enabled'] and a.health > r['health_threshold']
                    and a.energy >= r['min_energy'] and a.energy > r['energy_cost']
                    and a.age >= r['min_age_steps']
                    and (a.last_reproduced_age < 0 or a.age - a.last_reproduced_age >= r['cooldown_steps'])):
                eligible.append(a)
        if not eligible:
            return  # Preserve RNG sequence for environments without births.
        self.rng.shuffle(eligible)
        occupied = {a.position for a in parents}
        for parent in eligible:
            spec = self.config['species'][parent.species]
            r = spec['reproduction']
            if counts[parent.species] >= r['max_population']:
                continue
            free = [(parent.x + dx, parent.y + dy) for dx, dy in DIRECTIONS
                    if not self.blocked((parent.x + dx, parent.y + dy))
                    and (parent.x + dx, parent.y + dy) not in occupied]
            if not free:
                continue
            x, y = self.rng.choice(free)
            child_id = self.next_animal_id
            self.next_animal_id += 1
            child = Animal(child_id, parent.species, x, y, self.rng.randrange(4),
                           spec['body']['initial_energy'], spec['body']['max_health'],
                           parent_id=parent.id, generation=parent.generation + 1)
            self.animals[child_id] = child
            occupied.add(child.position)
            counts[parent.species] += 1
            parent.energy -= r['energy_cost']
            parent.last_reproduced_age = parent.age
            self.births[parent.species] += 1
            events.append(Event('born', parent.id, child_id, cause='reproduction'))

    def metrics(self):
        return dict(tick=self.tick, populations={s: sum(a.species == s for a in self.animals.values()) for s in self.config['species']}, food=sum(r.quantity for r in self.resources.values()), deaths=dict(self.deaths), energy_gained=dict(self.total_energy_gained), kills=dict(self.kills), births=dict(self.births))

    def snapshot(self):
        return dict(width=self.width, height=self.height, tick=self.tick, resource_capacity=self.config['resources']['capacity_per_patch'], obstacles=sorted(self.obstacles), resources=[vars(r).copy() for r in self.resources.values()], animals=[vars(a).copy() for a in self.animals.values()], events=[vars(e).copy() for e in self.events], metrics=self.metrics())

    def checkpoint(self):
        return dict(config=self.config, seed=self.seed, snapshot=self.snapshot(), rng_state=self.rng.getstate(), lifespans=self.lifespans, next_animal_id=self.next_animal_id)

    @classmethod
    def restore(cls, state):
        # Construct directly: restoration must not depend on a new layout draw.
        w = cls.__new__(cls)
        w.config = copy.deepcopy(state['config'])
        w.seed = state['seed']
        s = state['snapshot']
        w.width, w.height, w.tick = s['width'], s['height'], s['tick']
        w.obstacles = set(map(tuple, s['obstacles']))
        w.animals = {a['id']: Animal(**a) for a in s['animals']}
        w.resources = {(r['x'], r['y']): Resource(**r) for r in s['resources']}
        w.events = [Event(**e) for e in s['events']]
        w.deaths = defaultdict(int, s['metrics']['deaths'])
        w.total_energy_gained = defaultdict(float, s['metrics']['energy_gained'])
        w.kills = defaultdict(int, s['metrics']['kills'])
        w.lifespans = copy.deepcopy(state['lifespans'])
        w.births = defaultdict(int, s['metrics'].get('births', {}))
        # Legacy worlds have no descendants, so founding IDs remain reserved.
        founding_count = sum(spec['initial_population'] for spec in w.config['species'].values())
        highest_id = max(list(w.animals) + [a['id'] for a in w.lifespans], default=-1) + 1
        w.next_animal_id = state.get('next_animal_id', max(founding_count, highest_id))
        def tuples(v):
            return tuple(tuples(x) for x in v) if isinstance(v, (list, tuple)) else v
        w.rng = random.Random()
        w.rng.setstate(tuples(state['rng_state']))
        return w
