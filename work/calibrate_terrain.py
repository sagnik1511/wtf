"""Small matched fresh-policy calibration trials; saves scalar results, never replays."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median

from tqdm import tqdm
from ecosystem.config import load, validate
from ecosystem.runner import Experiment

VARIANTS = ('baseline', 'energy', 'energy_radius', 'energy_breeding', 'energy_reserves', 'energy_maintenance')
DEFAULT_VARIANTS = ('baseline', 'energy', 'energy_breeding')


def configure(source, variant, seed, steps):
    c = deepcopy(source)
    c['seed'] = seed
    c['world']['max_steps'] = steps
    c['experiment'].update(runs=1, world_seed_policy='increment', chain_best_brains=False)
    c['recording']['enabled'] = False
    for s in c['species'].values():
        s.pop('initial_brain', None)
        if variant != 'baseline':
            predator = s['diet'] == 'prey'
            s['body']['living_cost'] = .10 if predator else .07
            s['body']['movement_cost'] = .14 if predator else .10
            if predator:
                s['body']['attack_cost'] = .35
        if variant == 'energy_breeding':
            s['reproduction']['min_energy'] = 35 if predator else 25
            s['reproduction']['energy_cost'] = 12 if predator else 8
        elif variant == 'energy_reserves':
            s['body']['initial_energy'] = 100 if predator else 75
        elif variant == 'energy_maintenance':
            s['body']['living_cost'] = .075 if predator else .05
            if predator:
                s['body']['attack_cost'] = .25
        if variant == 'energy_radius':
            s['reproduction']['mate_radius'] = 3
    return validate(c)


def trial(config):
    e = Experiment(config)
    names = list(config['species'])
    stats = {name: Counter() for name in names}
    actions = {name: Counter() for name in names}
    initial_ids = set(e.world.animals)
    for _ in range(config['world']['max_steps']):
        if e.done:
            break
        before = dict(e.world.animals)
        for a in before.values():
            s = stats[a.species]
            s['animal_steps'] += 1
            s['starving_steps'] += a.energy <= 0
        if e.world.tick % 10 == 0:
            occupied = {a.position: a for a in before.values()}
            populations = Counter(a.species for a in before.values())
            for a in before.values():
                s = stats[a.species]
                s['sampled_states'] += 1
                s['physical_mate_states'] += e.world.reproduction_available(a, occupied, populations)
        e.step()
        for a in before.values():
            s = stats[a.species]
            actions[a.species][a.last_action] += 1
            b = config['species'][a.species]['body']
            s['energy_demand'] += b['living_cost']
            if a.last_action in ('move_forward', 'swim'):
                s['energy_demand'] += b['movement_cost']
            elif a.last_action == 'attack':
                s['energy_demand'] += b['attack_cost']
        for event in e.world.events:
            a = before.get(event.actor)
            if a is None:
                continue
            s = stats[a.species]
            if event.kind == 'energy_gained':
                s['food_energy'] += event.amount
            elif event.kind == 'energy_spent':
                s['energy_demand'] += event.amount
                s['terrain_energy'] += event.amount
            elif event.kind == 'born':
                # Both parents belong to the same species.
                s['energy_demand'] += 2 * config['species'][a.species]['reproduction']['energy_cost']
            elif event.kind == 'died':
                s['deaths_' + event.cause] += 1
    m = e.world.metrics()
    for name in names:
        survivors = [a for a in e.world.animals.values() if a.species == name]
        stats[name].update(births=m['births'].get(name, 0), kills=m['kills'].get(name, 0),
                           survivors=len(survivors),
                           newborn_survivors=sum(a.id not in initial_ids for a in survivors),
                           max_surviving_generation=max((a.generation for a in survivors), default=0))
    return dict(tick=e.world.tick, species={n: dict(stats[n], actions=dict(actions[n])) for n in names})



def aggregate(report):
    groups = defaultdict(list)
    for episode in report['episodes']:
        groups[(episode['environment'], episode['variant'])].append(episode)
    summaries = []
    for (environment, variant), episodes in groups.items():
        species = {}
        for name in episodes[0]['species']:
            rows = [e['species'][name] for e in episodes]
            totals = Counter()
            for row in rows:
                totals.update({k:v for k,v in row.items() if k != 'actions'})
            body = report['configurations'][environment][variant]['species'][name]['body']
            founders = report['configurations'][environment][variant]['species'][name]['initial_population']
            opening_energy = len(episodes) * founders * body['initial_energy']
            offspring_energy = totals['births'] * body['initial_energy']
            inputs = opening_energy + offspring_energy + totals['food_energy']
            species[name] = dict(
                opening_energy=opening_energy, offspring_energy=offspring_energy,
                food_input_share=totals['food_energy']/inputs if inputs else None,
                total_births=totals['births'], median_births=median(r['births'] for r in rows),
                runs_with_births=sum(r['births'] > 0 for r in rows),
                total_survivors=totals['survivors'],
                extinction_runs=sum(r['survivors'] == 0 for r in rows),
                starvation_deaths=totals['deaths_starvation'], drowning_deaths=totals['deaths_drowning'],
                newborn_survivors=totals['newborn_survivors'], kills=totals['kills'],
                physical_mate_states=totals['physical_mate_states'], sampled_states=totals['sampled_states'],
                energy_coverage=totals['food_energy']/totals['energy_demand'] if totals['energy_demand'] else None,
                starving_fraction=totals['starving_steps']/totals['animal_steps'] if totals['animal_steps'] else None)
        summaries.append(dict(environment=environment, variant=variant, episodes=len(episodes), species=species))
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environments', nargs='+', default=[f'environments/{n}.json' for n in ('river_crossing', 'highlands', 'enclosed_valleys')])
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(DEFAULT_VARIANTS))
    p.add_argument('--seeds', nargs='+', type=int, default=[100,101,102,103,104])
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--reference-report', type=Path, help='Use captured baseline configurations from an earlier calibration report')
    p.add_argument('--output', type=Path, default=Path('results') / ('terrain_calibration_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')) / 'summary.json')
    args = p.parse_args()
    if args.steps < 1 or any(not 0 <= seed < 2**32 for seed in args.seeds):
        p.error('steps must be positive and seeds must be unsigned 32-bit integers')
    if args.reference_report:
        try:
            captured = json.loads(args.reference_report.read_text())['configurations']
            sources = {str(path): validate(deepcopy(captured[str(path)]['baseline'])) for path in args.environments}
        except (OSError, ValueError, KeyError) as error:
            p.error(f'Cannot load captured baseline configurations: {error}')
    else:
        sources = {str(path): load(path) for path in args.environments}
    report = dict(steps=args.steps, seeds=args.seeds, variants=args.variants,
                  reference_report=str(args.reference_report) if args.reference_report else None,
                  configurations={}, episodes=[])
    for path, c in sources.items():
        report['configurations'][path] = {v: configure(c,v,args.seeds[0],args.steps) for v in args.variants}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tasks = [(path,c,v,seed) for path,c in sources.items() for seed in args.seeds for v in args.variants]
    for path,c,v,seed in tqdm(tasks, desc='Calibration episodes', mininterval=5):
        result = dict(environment=path, variant=v, seed=seed, **trial(configure(c,v,seed,args.steps)))
        report['episodes'].append(result)
        report['aggregates'] = aggregate(report)
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2)+'\n')
        temp.replace(args.output)
        tqdm.write(f"{Path(path).stem} {v} seed={seed}: " + ' | '.join(
            f"{n} births={s['births']} survivors={s['survivors']}" for n,s in result['species'].items()))
    print(args.output)


if __name__ == '__main__':
    main()
