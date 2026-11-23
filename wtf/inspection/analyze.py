"""Offline behavior statistics from bounded playback chunks; no learning imports."""

# Black owns code layout; long literal payloads, URLs and regression names retain
# their exact text.
# pylint: disable=line-too-long


import argparse
from collections import Counter
import json
import math
from pathlib import Path
from wtf.storage.replay_io import chunk_path, load_chunk
from wtf.inspection.movement import MovementCounts
from wtf.simulation.actions import (
    MOVE_ACTIONS,
    WALK_ACTIONS,
    ATTEMPT_ACTIONS,
    movement_action,
)

DEFINITIONS = {
    "agent_steps": "One living animal present at the start of one observed consecutive step.",
    "energy_budget": "Opening snapshot energy + realized food energy + configured newborn grants. Net reproduction energy is newborn grants minus parent reproduction costs; energy_coverage excludes grants.",
    "feeding_success": "Positive resource-energy events / observed eat attempts.",
    "hunt_success": "Kills / observed attack attempts; partial hits are reported separately.",
    "movement_heatmap": "Whole-run changed-position arrivals per destination cell, across all species/sexes. Only consecutive recorded frames and known terminal positions count; births, stationary frames and interpolated routes do not.",
    "movement_success": "Successful moves / observed move attempts, including blocked attempts.",
    "energy_coverage": "Food/prey energy gained / living + known action + terrain + reproduction energy demand. Initial and newborn energy are excluded from intake.",
    "occupied_grid_fraction": "Distinct recorded occupied cells / all grid cells, including obstacles in the denominator.",
    "starving_fraction": "Animal-steps starting at zero energy / observed animal-steps.",
    "mean_completed_lifespan": "Mean age at death only; survivors are censored and listed separately.",
    "reward_per_agent_step": "Event-derived reward including terminal penalties / observed animal-steps.",
    "births_per_1000_agent_steps": "Observed births * 1000 / observed animal-steps.",
    "grpo_zero_variance_fraction": "Sampled new GRPO groups with equal returns / classified new groups. Inherited counters are excluded.",
    "normalized_entropy": "Sampled policy entropy / log(number of actions); 1 is uniform.",
    "observed_optimizer_updates": "Positive diagnostic counter changes within each animal; the first sample establishes an inheritance baseline. A lower bound when samples miss updates before death.",
}
LIMITATIONS = [
    "Behavior changes in one training run do not establish policy improvement. Compare frozen trained and fresh/random policies on matched held-out world seeds.",
    "Population, age, inheritance and survivor selection affect early/late comparisons.",
    "Per-step behavior uses adjacent recorded frames only. Sampled recordings omit events; missing steps are not interpolated.",
    "Actions of animals dying that step are inferred only when an event identifies the action. Action coverage quantifies the remaining missing actions.",
    "Learning diagnostics are sampled; updates and group observations are lower bounds. Policy entropy is measured on changing observed states, not a fixed probe set.",
]


def ratio(a, b):
    """Compute a ratio with the configured empty-denominator behavior."""
    return a / b if b else None


class Statistics:
    """Aggregate observed behavior without loading or changing learning state."""

    def __init__(self, config):
        self.config = config
        self.counts = Counter()
        self.actions = Counter()
        self.deaths = Counter()
        self.visited = 0

    def position(self, animal):
        """Return the entity's grid coordinates."""
        index = animal["y"] * self.config["world"]["width"] + animal["x"]
        self.visited |= 1 << index

    def expose(self, animal):
        """Accumulate the observed animal-time exposure."""
        body = self.config["species"][animal["species"]]["body"]
        c = self.counts
        c["agent_steps"] += 1
        c["energy_fraction_sum"] += animal["energy"] / body["max_energy"]
        c["starving_steps"] += animal["energy"] <= 0
        c["energy_demand"] += body["living_cost"]
        c["reward"] += body["living_cost"] * self.config["species"][animal["species"]][
            "rewards"
        ].get("per_energy_cost", 0)
        self.position(animal)

    def action(self, animal, action):
        """Count an observed action in the appropriate species statistics."""
        self.actions[action] += 1
        spec = self.config["species"][animal["species"]]
        body = spec["body"]
        cost = (
            body["movement_cost"]
            if action in MOVE_ACTIONS
            else body["attack_cost"] if action == "attack" else 0
        )
        self.counts["energy_demand"] += cost
        self.counts["reward"] += cost * spec["rewards"].get("per_energy_cost", 0)

    def failed(self, animal):
        """Count a failed physical attempt and its recorded reason."""
        self.counts["failed_actions"] += 1
        self.counts["reward"] += self.config["species"][animal["species"]][
            "rewards"
        ].get("failed_action", 0)

    def event(self, event, animal):
        """Accumulate a saved physical or lifecycle event."""

        # Keep this ordered phase and its shared state together; splitting it only to
        # meet a count obscures the contract.
        # pylint: disable=too-many-branches

        c = self.counts
        spec = self.config["species"][animal["species"]]
        rewards = spec["rewards"]
        kind, amount = event["kind"], event.get("amount", 0)
        if kind == "energy_gained":
            c["energy_gained"] += amount
            c["reward"] += amount * rewards["per_energy_gained"]
            if event.get("cause") == "resource" and amount > 0:
                c["productive_feeds"] += 1
        elif kind == "damage_dealt":
            c["hits"] += amount > 0 and event.get("cause") != "competition"
            if event.get("cause") == "competition":
                c["competition_damage_dealt"] += amount
            c["reward"] += amount * rewards.get("per_damage_dealt", 0)
        elif kind == "damage_taken":
            c["reward"] += amount * rewards["per_damage_taken"]
        elif kind == "kill":
            c["kills"] += 1
        elif kind in ("contest_won", "contest_lost"):
            c[kind] += 1
        elif kind == "moved":
            c["swim_moves" if event.get("cause") == "swim" else "moves"] += 1
        elif kind == "drifted":
            c["drift_moves"] += 1
        elif kind == "energy_spent":
            c["energy_demand"] += amount
            c[
                (
                    "competition_energy"
                    if event.get("cause", "").startswith("competition")
                    else "terrain_energy"
                )
            ] += amount
            c["reward"] += amount * rewards.get("per_energy_cost", 0)
        elif kind == "born":
            c["births"] += 1
            cost = spec.get("reproduction", {}).get("energy_cost", 0) * (
                2 if event.get("mate", -1) >= 0 else 1
            )
            c["energy_demand"] += cost
            c["reproduction_energy"] += cost
            c["offspring_energy"] += spec["body"]["initial_energy"]
            c["reward"] += cost * rewards.get("per_energy_cost", 0)
            c["reward"] += (2 if event.get("mate", -1) >= 0 else 1) * rewards.get(
                "per_offspring", 0
            )
        elif kind == "mutated":
            c["mutations"] += 1
            c["mutated_weights"] += amount
        elif kind == "mating_failed":
            c["mating_failed:" + event.get("cause", "unknown")] += 1
        elif kind == "died":
            c["completed_lives"] += 1
            c["death_age_sum"] += animal["age"] + 1
            c["reward"] += rewards["death"]
            self.deaths[event.get("cause", "unknown")] += 1

    def diagnostic(self, diagnostic, previous):
        """Accumulate a saved brain prediction sample."""
        c = self.counts
        c["diagnostic_samples"] += 1
        if (
            "entropy" in diagnostic
            and len(diagnostic.get("action_probabilities", {})) > 1
        ):
            c["policy_samples"] += 1
            c["entropy_sum"] += diagnostic["entropy"] / math.log(
                len(diagnostic["action_probabilities"])
            )
            c["peak_probability_sum"] += max(
                diagnostic["action_probabilities"].values()
            )
        if previous is None:
            return
        c["optimizer_updates"] += max(0, diagnostic["updates"] - previous["updates"])
        if diagnostic.get("algorithm") == "grpo":
            delta = diagnostic["rollout_updates"] - previous["rollout_updates"]
            c["grpo_groups"] += max(0, delta)
            metrics = diagnostic.get("training_metrics", {})
            if delta > 0 and "zero_variance" in metrics:
                c["classified_groups"] += 1
                c["zero_variance_groups"] += bool(metrics["zero_variance"])
                c["positive_return_groups"] += any(
                    r > 0 for r in metrics.get("group_returns", [])
                )

    def report(self, species):
        """Return the accumulated behavioral statistics."""

        # Keyword dict construction deliberately detects duplicate keys in dynamic **
        # expansions; a literal would silently overwrite them.
        # pylint: disable=use-dict-literal

        c, a = self.counts, self.actions
        actions = sum(a.values())
        capacity = self.config["world"]["width"] * self.config["world"]["height"]
        return dict(
            **(
                {
                    "challenge_attempts": a["challenge"],
                    "contest_wins": c["contest_won"],
                    "contest_losses": c["contest_lost"],
                    "competition_damage_dealt": c["competition_damage_dealt"],
                    "competition_energy_cost": c["competition_energy"],
                }
                if self.config["species"][species]
                .get("competition", {})
                .get("enabled", False)
                else {}
            ),
            diet=self.config["species"][species]["diet"],
            agent_steps=c["agent_steps"],
            observed_actions=actions,
            action_coverage=ratio(actions, c["agent_steps"]),
            actions=dict(a),
            action_fractions={k: ratio(v, actions) for k, v in a.items()},
            mean_energy_fraction=ratio(c["energy_fraction_sum"], c["agent_steps"]),
            starving_fraction=ratio(c["starving_steps"], c["agent_steps"]),
            energy_coverage=ratio(c["energy_gained"], c["energy_demand"]),
            energy_gained=c["energy_gained"],
            energy_demand=c["energy_demand"],
            offspring_energy=c["offspring_energy"],
            reproduction_energy_cost=c["reproduction_energy"],
            feeding_success=ratio(c["productive_feeds"], a["eat"]),
            hunt_success=ratio(c["kills"], a["attack"]),
            hit_success=ratio(c["hits"], a["attack"]),
            movement_success=ratio(
                c["moves"], sum(a[action] for action in WALK_ACTIONS)
            ),
            swim_success=ratio(c["swim_moves"], a["swim"]),
            swim_attempts=a["swim"],
            failed_actions=c["failed_actions"],
            reproduction_attempts=a["reproduce"],
            reproduction_success=(
                ratio(2 * c["births"], a["reproduce"])
                if self.config["species"][species]
                .get("reproduction", {})
                .get("intent_steps", 1)
                == 1
                else None
            ),
            drift_moves=c["drift_moves"],
            terrain_energy=c["terrain_energy"],
            reward_per_agent_step=ratio(c["reward"], c["agent_steps"]),
            births=c["births"],
            mating_failures={
                k.split(":", 1)[1]: v
                for k, v in c.items()
                if k.startswith("mating_failed:")
            },
            kills=c["kills"],
            deaths=dict(self.deaths),
            mutations=c["mutations"],
            mutated_weights=c["mutated_weights"],
            births_per_1000_agent_steps=ratio(c["births"] * 1000, c["agent_steps"]),
            deaths_per_1000_agent_steps=ratio(
                c["completed_lives"] * 1000, c["agent_steps"]
            ),
            completed_lives=c["completed_lives"],
            mean_completed_lifespan=ratio(c["death_age_sum"], c["completed_lives"]),
            occupied_cells=bin(self.visited).count("1"),
            occupied_grid_fraction=ratio(bin(self.visited).count("1"), capacity),
            learning={
                "algorithm": self.config["brains"][
                    self.config["species"][species]["brain"]
                ]["type"],
                "diagnostic_samples": c["diagnostic_samples"],
                "observed_optimizer_updates": c["optimizer_updates"],
                "normalized_entropy": ratio(c["entropy_sum"], c["policy_samples"]),
                "mean_peak_action_probability": ratio(
                    c["peak_probability_sum"], c["policy_samples"]
                ),
                "observed_grpo_groups": c["grpo_groups"],
                "classified_grpo_groups": c["classified_groups"],
                "grpo_zero_variance_fraction": ratio(
                    c["zero_variance_groups"], c["classified_groups"]
                ),
                "grpo_positive_return_groups": c["positive_return_groups"],
            },
        )


def analyze_run(config, frames, window_steps=250, *, include_movement=True):
    """Consume ordered frames once, retaining only the prior frame and summaries."""

    # Keep independent numeric/compatibility or benchmark paths explicit to preserve
    # their exact contracts.
    # pylint: disable=duplicate-code

    # Keep this ordered phase and its shared state together; splitting it only to meet
    # a count obscures the contract.
    # pylint: disable=too-many-boolean-expressions
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    # Exact types intentionally reject bool/subclass substitutions in persisted
    # numeric contracts.
    # pylint: disable=unidiomatic-typecheck

    if type(window_steps) is not int or window_steps < 1:
        raise ValueError("window_steps must be a positive integer")
    overall = {s: Statistics(config) for s in config["species"]}
    windows, diagnostics = {}, {}
    previous = None
    first_tick = last_tick = 0
    frame_count = adjacent = 0
    initial_populations, opening_energy = {}, {}
    final_metrics = {}
    width, height = config["world"]["width"], config["world"]["height"]
    movement = MovementCounts(width, height) if include_movement else None

    for frame in frames:
        tick = frame["tick"]
        if previous is not None and tick <= last_tick:
            raise ValueError("Recorded frame ticks must increase")
        if previous is None:
            first_tick = tick
            initial_populations = {
                s: sum(a["species"] == s for a in frame["animals"]) for s in overall
            }
            opening_energy = {
                s: sum(a["energy"] for a in frame["animals"] if a["species"] == s)
                for s in overall
            }
        last_tick, frame_count = tick, frame_count + 1
        final_metrics = frame["metrics"]
        animals = {a["id"]: a for a in frame["animals"]}
        # At most 120 time windows for the recording; analyze_recording chooses
        # window_steps from the advertised duration before consuming chunks.
        window_id = max(0, (tick - first_tick - 1) // window_steps)
        if window_id not in windows:
            windows[window_id] = {s: Statistics(config) for s in overall}
        window = windows[window_id]

        def destinations(species):

            # The callback is consumed synchronously within this iteration; it never
            # escapes the loop.
            # pylint: disable=cell-var-from-loop

            return overall[species], window[species]

        # The explicit None guard keeps both dictionary accesses on valid frames.
        if (
            previous is not None
            and tick == previous["tick"] + 1  # pylint: disable=unsubscriptable-object
        ):
            adjacent += 1
            prior = {
                a["id"]: a
                for a in previous["animals"]  # pylint: disable=unsubscriptable-object
            }
            for animal in prior.values():
                for stat in destinations(animal["species"]):
                    stat.expose(animal)
            actions = {}
            for i, animal in animals.items():
                if i in prior:
                    actions[i] = animal["last_action"]
                    if movement is not None:
                        movement.position(prior[i], animal)

            # Successful terminal actions are recoverable from recorded events.
            for event in frame.get("events", []):
                i, kind = event["actor"], event["kind"]
                if movement is not None:
                    movement.terminal(prior, animals, event)
                if i in prior and i not in actions:
                    if kind == "action_failed":
                        actions[i] = event["cause"]
                    elif kind == "mating_failed":
                        actions[i] = "reproduce"
                    elif kind in ("moved", "movement_blocked"):
                        actions[i] = movement_action(event.get("cause", ""))
                    elif kind == "challenge_attempt":
                        actions[i] = "challenge"
                    elif kind == "damage_dealt" and event.get("cause") != "competition":
                        actions[i] = "attack"
                    elif kind == "energy_gained" and event.get("cause") == "resource":
                        actions[i] = "eat"
            successful = set()
            for event in frame.get("events", []):
                kind = event["kind"]
                if (
                    kind in ("moved", "challenge_resolved")
                    or kind == "damage_dealt"
                    and event.get("cause") != "competition"
                    and event.get("amount", 0) > 0
                    or kind == "energy_gained"
                    and event.get("cause") == "resource"
                    and event.get("amount", 0) > 0
                ):
                    successful.add(event["actor"])
                if kind == "born":
                    successful.update((event["actor"], event.get("mate", -1)))
            for i, action in actions.items():
                for stat in destinations(prior[i]["species"]):
                    stat.action(prior[i], action)
                    if action in ATTEMPT_ACTIONS and i not in successful:
                        stat.failed(prior[i])
            for event in frame.get("events", []):
                animal = prior.get(event["actor"])
                if animal is not None:
                    for stat in destinations(animal["species"]):
                        stat.event(event, animal)
        for animal in animals.values():
            for stat in destinations(animal["species"]):
                stat.position(animal)
        for raw_id, diagnostic in frame.get("diagnostics", {}).items():
            animal_id = int(raw_id)
            if animal_id not in animals:
                continue
            for stat in destinations(animals[animal_id]["species"]):
                stat.diagnostic(diagnostic, diagnostics.get(animal_id))
            # Store counters only, not duplicate feature vectors or group data.
            diagnostics[animal_id] = {
                key: diagnostic.get(key, 0) for key in ("updates", "rollout_updates")
            }
        for animal_id in diagnostics.keys() - animals.keys():
            del diagnostics[animal_id]
        previous = frame
    result = {
        "first_tick": first_tick,
        "last_tick": last_tick,
        "frames": frame_count,
        "window_steps": window_steps,
        "coverage": {
            "adjacent_steps": adjacent,
            "total_steps": last_tick - first_tick,
            "step_fraction": ratio(adjacent, last_tick - first_tick),
        },
        "initial_populations": initial_populations,
        "final_metrics": final_metrics,
        "species": {s: stat.report(s) for s, stat in overall.items()},
        "windows": [],
    }
    if movement is not None:
        result["movement_heatmap"] = movement.summary()
    for s, stat in result["species"].items():
        survivors = [
            a for a in (previous or {}).get("animals", []) if a["species"] == s
        ]
        stat["energy_budget"] = {
            "opening_energy": opening_energy.get(s, 0),
            "food_energy": stat["energy_gained"],
            "offspring_energy": stat["offspring_energy"],
            "reproduction_energy_cost": stat["reproduction_energy_cost"],
            "net_reproduction_energy": stat["offspring_energy"]
            - stat["reproduction_energy_cost"],
            "food_input_share": ratio(
                stat["energy_gained"],
                opening_energy.get(s, 0)
                + stat["offspring_energy"]
                + stat["energy_gained"],
            ),
        }
        stat["survivors"] = len(survivors)
        stat["mean_survivor_age"] = ratio(
            sum(a["age"] for a in survivors), len(survivors)
        )
    for i, window in sorted(windows.items()):
        result["windows"].append(
            {
                "start_tick": (
                    first_tick if i == 0 else first_tick + i * window_steps + 1
                ),
                "end_tick": min(last_tick, first_tick + (i + 1) * window_steps),
                "species": {s: stat.report(s) for s, stat in window.items()},
            }
        )
    return result


def analyze_recording(folder, run=None, window_steps=250, *, include_movement=True):
    """Analyze a saved recording without running or updating its brains."""

    # Exact types intentionally reject bool/subclass substitutions in persisted
    # numeric contracts.
    # pylint: disable=unidiomatic-typecheck

    folder = Path(folder)
    playback = folder if (folder / "recording.json").is_file() else folder / "playback"
    playback = playback.resolve()
    manifest = json.loads((playback / "recording.json").read_text())
    if manifest.get("recording_version") != 1:
        raise ValueError("Unsupported recording version")
    if type(window_steps) is not int or window_steps < 1:
        raise ValueError("window_steps must be a positive integer")
    selected = [
        entry for entry in manifest["runs"] if run is None or entry["run"] == run
    ]
    if not selected or not any(entry["chunk_count"] for entry in selected):
        raise FileNotFoundError("Recorded run not found")
    report = {
        "analysis_version": 1,
        "recording": manifest["name"],
        "mode": manifest["mode"],
        "definitions": DEFINITIONS,
        "limitations": LIMITATIONS,
        "runs": [],
    }
    for entry in selected:
        if not entry["chunk_count"]:
            continue
        size = max(
            window_steps, math.ceil((entry["last_tick"] - entry["first_tick"]) / 120)
        )

        def frames():

            # The callback is consumed synchronously within this iteration; it never
            # escapes the loop.
            # pylint: disable=cell-var-from-loop

            for index in range(entry["chunk_count"]):
                path = chunk_path(playback, entry["run"], index)
                chunk = load_chunk(path)
                yield from chunk
                del chunk

        result = analyze_run(
            manifest["config"], frames(), size, include_movement=include_movement
        )
        result.update(run=entry["run"], seed=entry["seed"], complete=entry["complete"])
        report["runs"].append(result)
    return report


def main():
    """Parse command-line options and run the requested operation."""
    parser = argparse.ArgumentParser(
        description="Analyze saved WTF behavior without loading brains or simulating"
    )
    parser.add_argument("recording", type=Path, help="Experiment folder under results/")
    parser.add_argument("--run", type=int)
    parser.add_argument("--window-steps", type=int, default=250)
    parser.add_argument(
        "--output", type=Path, help="Default: <recording>/analysis.json"
    )
    args = parser.parse_args()
    report = analyze_recording(args.recording, args.run, args.window_steps)
    output = args.output or args.recording / "analysis.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for run in report["runs"]:
        print(
            f"Run {run['run']} · steps {run['first_tick']}–{run['last_tick']} · observed coverage {run['coverage']['step_fraction']}"
        )
        for species, stats in run["species"].items():
            print(species, json.dumps(stats, allow_nan=False))
    print(f"Analysis: {output.resolve()}")


if __name__ == "__main__":
    main()
