"""Short diagnostic: stop when predators die; do not save model checkpoints."""

import argparse
from collections import Counter
from ecosystem.config import load
from ecosystem.runner import Experiment

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--variant", choices=["current", "lethal", "exploratory"], default="lethal"
)
args = parser.parse_args()
c = load("environments/baseline.json")
c["seed"] = args.seed
c["experiment"]["runs"] = 1
if args.variant in ("lethal", "exploratory"):
    c["species"]["predator"]["body"]["attack_damage"] = 30
if args.variant == "exploratory":
    import copy

    c["brains"]["predator_q"] = copy.deepcopy(
        c["brains"][c["species"]["predator"]["brain"]]
    )
    c["brains"]["predator_q"]["exploration"].update(
        initial=0.8, minimum=0.25, decay_per_decision=0.9995
    )
    c["species"]["predator"]["brain"] = "predator_q"
e = Experiment(c)
actions = Counter()
hits = 0
while not e.done:
    e.step()
    for a in e.world.animals.values():
        if a.species == "predator":
            actions[a.last_action] += 1
    hits += sum(event.kind == "damage_dealt" for event in e.world.events)
    if not any(a.species == "predator" for a in e.world.animals.values()):
        break
print(
    dict(
        variant=args.variant,
        seed=args.seed,
        tick=e.world.tick,
        kills=dict(e.world.kills),
        hits=hits,
        actions=dict(actions),
    ),
    flush=True,
)
