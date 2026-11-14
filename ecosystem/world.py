"""Physical state and simultaneous action resolution; no learning dependencies."""
import copy
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass

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

    def observe(self, a):
        body = self.body(a)
        radius = body['vision_radius']
        occupied = {other.position for other in self.animals.values() if other.id != a.id}
        features = [a.energy / body['max_energy'], a.health / body['max_health'], *DIRECTIONS[a.heading]]
        for offset in range(4):
            dx, dy = DIRECTIONS[(a.heading + offset) % 4]
            p = a.x + dx, a.y + dy
            features.append(float(self.blocked(p) or p in occupied))
        def nearest(points):
            best = None
            distance = float('inf')
            for p in points:
                d = math.hypot(p[0] - a.x, p[1] - a.y)
                if d <= radius and d < distance and self.visible(a.position, p):
                    best, distance = p, d
            return [0., 0., 0., 0.] if best is None else [(best[0] - a.x) / radius, (best[1] - a.y) / radius, distance / radius, 1.]
        features += nearest(p for p, r in self.resources.items() if r.quantity > .001)
        features += nearest(b.position for b in self.animals.values() if b.id != a.id and self.diet(b) == 'resource')
        features += nearest(b.position for b in self.animals.values() if b.id != a.id and self.diet(b) == 'prey')
        return tuple(features)

    def observe_all(self):
        return {i: self.observe(a) for i, a in self.animals.items()}

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
            damage = min(target.health, sum(self.body(self.animals[i])['attack_damage'] for i in ids))
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
        r = self.config['resources']
        for patch in self.resources.values():
            patch.quantity = min(r['capacity_per_patch'], patch.quantity + r['growth_per_step'])
        self.tick += 1
        self.events = events
        return StepResult(self.observe_all(), rewards, dead, events)

    def metrics(self):
        return dict(tick=self.tick, populations={s: sum(a.species == s for a in self.animals.values()) for s in self.config['species']}, food=sum(r.quantity for r in self.resources.values()), deaths=dict(self.deaths), energy_gained=dict(self.total_energy_gained), kills=dict(self.kills))

    def snapshot(self):
        return dict(width=self.width, height=self.height, tick=self.tick, resource_capacity=self.config['resources']['capacity_per_patch'], obstacles=sorted(self.obstacles), resources=[asdict(r) for r in self.resources.values()], animals=[asdict(a) for a in self.animals.values()], events=[asdict(e) for e in self.events], metrics=self.metrics())

    def checkpoint(self):
        return dict(config=self.config, seed=self.seed, snapshot=self.snapshot(), rng_state=self.rng.getstate(), lifespans=self.lifespans)

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
        def tuples(v):
            return tuple(tuples(x) for x in v) if isinstance(v, (list, tuple)) else v
        w.rng = random.Random()
        w.rng.setstate(tuples(state['rng_state']))
        return w
