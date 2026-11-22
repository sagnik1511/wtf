"""Exact per-recorded-step dashboard data, derived without running a simulation."""

from wtf.native import kernels
from wtf.simulation.actions import ATTEMPT_ACTIONS, movement_action
from collections import Counter, defaultdict
import math

PAGE_SIZE = 128
GROUP_RADIUS = 3


def mean(values):
    return sum(values) / len(values) if values else None


def groups(animals):
    """Same-species connected components within three cells, ignoring obstacles."""
    if kernels.available("groups"):
        result = kernels.native.groups(animals, GROUP_RADIUS)
        if result is not NotImplemented:
            return result
    parent = list(range(len(animals)))
    buckets = defaultdict(list)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(animals):
        bx, by = a["x"] // GROUP_RADIUS, a["y"] // GROUP_RADIUS
        for x in range(bx - 1, bx + 2):
            for y in range(by - 1, by + 2):
                for j in buckets[(x, y)]:
                    b = animals[j]
                    if (a["x"] - b["x"]) ** 2 + (
                        a["y"] - b["y"]
                    ) ** 2 <= GROUP_RADIUS**2:
                        parent[find(i)] = find(j)
        buckets[(bx, by)].append(i)
    sizes = Counter(find(i) for i in range(len(animals)))
    return dict(
        clusters=sum(n >= 2 for n in sizes.values()),
        solitary=sum(n == 1 for n in sizes.values()),
        largest=max(sizes.values(), default=0),
    )


def summarize(config, frame, previous=None):
    animals = {a["id"]: a for a in frame["animals"]}
    prior = {a["id"]: a for a in previous["animals"]} if previous else {}
    consecutive = previous is not None and frame["tick"] == previous["tick"] + 1
    actions, events = defaultdict(Counter), defaultdict(Counter)
    successful = set()
    if consecutive:
        observed = {i: a["last_action"] for i, a in animals.items() if i in prior}
        for event in frame.get("events", []):
            i, kind = event["actor"], event["kind"]
            if i in prior and i not in observed:
                if kind == "action_failed":
                    observed[i] = event["cause"]
                elif kind == "mating_failed":
                    observed[i] = "reproduce"
                elif kind in ("moved", "movement_blocked"):
                    observed[i] = movement_action(event.get("cause", ""))
                elif kind == "challenge_attempt":
                    observed[i] = "challenge"
                elif kind == "damage_dealt" and event.get("cause") != "competition":
                    observed[i] = "attack"
                elif kind == "energy_gained" and event.get("cause") == "resource":
                    observed[i] = "eat"
            if (
                kind in ("moved", "challenge_resolved")
                or kind == "damage_dealt"
                and event.get("cause") != "competition"
                and event.get("amount", 0) > 0
                or kind == "energy_gained"
                and event.get("cause") == "resource"
                and event.get("amount", 0) > 0
            ):
                successful.add(i)
            if kind == "born":
                successful.update((i, event.get("mate", -1)))
            actor = prior.get(i)
            if actor is not None:
                counter = events[actor["species"]]
                if kind in (
                    "born",
                    "kill",
                    "died",
                    "mutated",
                    "drifted",
                    "contest_won",
                    "contest_lost",
                ):
                    counter[kind] += 1
                if kind == "energy_spent":
                    counter[
                        (
                            "competition_energy"
                            if event.get("cause", "").startswith("competition")
                            else "terrain_energy"
                        )
                    ] += event.get("amount", 0)
                if kind == "mating_failed":
                    counter["mating_failed:" + event.get("cause", "unknown")] += 1
                if kind == "died":
                    counter[event.get("cause", "unknown")] += 1
                if (
                    kind == "energy_gained"
                    and event.get("cause") == "resource"
                    and event.get("amount", 0) > 0
                ):
                    counter["productive_feeds"] += 1
        for i, action in observed.items():
            actions[prior[i]["species"]][action] += 1
            if action in ATTEMPT_ACTIONS and i not in successful:
                events[prior[i]["species"]]["failed_actions"] += 1
    species = {}
    for name, spec in config["species"].items():
        living = [a for a in animals.values() if a["species"] == name]
        old = [a for a in prior.values() if a["species"] == name]
        body = spec["body"]
        predictions, q_values, entropies, values, epsilons = (
            defaultdict(list),
            defaultdict(list),
            [],
            [],
            [],
        )
        samples = 0
        for a in living:
            diagnostic = frame.get("diagnostics", {}).get(str(a["id"]))
            if diagnostic is None:
                continue
            scores = diagnostic.get(
                "action_probabilities", diagnostic.get("action_values", {})
            )
            if scores:
                samples += 1
            target = predictions if "action_probabilities" in diagnostic else q_values
            for action, value in scores.items():
                target[action].append(value)
            if "entropy" in diagnostic and len(scores) > 1:
                entropies.append(diagnostic["entropy"] / math.log(len(scores)))
            if diagnostic.get("state_value") is not None:
                values.append(diagnostic["state_value"])
            if diagnostic.get("epsilon") is not None:
                epsilons.append(diagnostic["epsilon"])
        event, action = events[name], actions[name]
        attempts = action["attack"] if spec["diet"] == "prey" else action["eat"]
        success = event["kill"] if spec["diet"] == "prey" else event["productive_feeds"]
        species[name] = dict(
            **(
                dict(
                    challenge_attempts=action["challenge"] if consecutive else None,
                    contest_wins=event["contest_won"] if consecutive else None,
                    contest_losses=event["contest_lost"] if consecutive else None,
                    competition_deaths=event["competition"] if consecutive else None,
                    competition_energy=(
                        event["competition_energy"] if consecutive else None
                    ),
                )
                if spec.get("competition", {}).get("enabled", False)
                else {}
            ),
            population=len(living),
            health=mean([a["health"] / body["max_health"] for a in living]),
            energy=mean([a["energy"] / body["max_energy"] for a in living]),
            mean_age=mean([a["age"] for a in living]),
            mean_generation=mean([a.get("generation", 0) for a in living]),
            max_generation=max((a.get("generation", 0) for a in living), default=None),
            starving=sum(a["energy"] <= 0 for a in living),
            births=event["born"] if consecutive else None,
            mutations=event["mutated"] if consecutive else None,
            deaths=event["died"] if consecutive else None,
            predation_deaths=event["predation"] if consecutive else None,
            starvation_deaths=event["starvation"] if consecutive else None,
            drowning_deaths=event["drowning"] if consecutive else None,
            swim_attempts=action["swim"] if consecutive else None,
            reproduction_attempts=action["reproduce"] if consecutive else None,
            mating_failures=(
                {
                    k.split(":", 1)[1]: v
                    for k, v in event.items()
                    if k.startswith("mating_failed:")
                }
                if consecutive
                else None
            ),
            failed_actions=event["failed_actions"] if consecutive else None,
            reproduction_success=(
                2 * event["born"] / action["reproduce"]
                if consecutive
                and action["reproduce"]
                and spec.get("reproduction", {}).get("intent_steps", 1) == 1
                else None
            ),
            drift_moves=event["drifted"] if consecutive else None,
            terrain_energy=event["terrain_energy"] if consecutive else None,
            kills=event["kill"] if consecutive else None,
            attempts=attempts if consecutive else None,
            successes=success if consecutive else None,
            success=success / attempts if consecutive and attempts else None,
            action_coverage=(
                sum(action.values()) / len(old) if consecutive and old else None
            ),
            reward=(
                mean(
                    [
                        a["last_reward"]
                        for a in living
                        if a["id"] in prior and a.get("last_reward") is not None
                    ]
                )
                if consecutive
                else None
            ),
            groups=groups(living),
            predictions={k: mean(v) for k, v in predictions.items()},
            q_values={k: mean(v) for k, v in q_values.items()},
            entropy=mean(entropies),
            state_value=mean(values),
            epsilon=mean(epsilons),
            diagnostic_samples=samples,
        )
    return dict(
        tick=frame["tick"],
        consecutive=consecutive,
        food=frame["metrics"]["food"],
        species=species,
    )
