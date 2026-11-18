"""Arbitrary species identities with multiple prey and predator populations."""

import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ecosystem.analyze import analyze_recording
from ecosystem.checkpoint_io import read_checkpoint
from ecosystem.config import validate, load
from ecosystem.contextual_ppo import action_mask, central_contexts
from ecosystem.library import apply_brain_overrides, export_champions
from ecosystem.recording import Recorder, RecordingStore
from ecosystem.replay_io import load_chunk
from ecosystem.runner import Experiment
from ecosystem.telemetry import summarize
from ecosystem.world import Animal, World
from tests.test_efficiency import reference_observation
from tests.test_mutation import KINDS, environment

NAMES = ("rabbits", "deer", "foxes", "wolves")


def configuration(kind="q_network"):
    c = environment(kind, 0)
    herbivore, carnivore = c["species"]["prey"], c["species"]["predator"]
    c["species"] = {
        name: copy.deepcopy(herbivore if index < 2 else carnivore)
        for index, name in enumerate(NAMES)
    }
    for spec in c["species"].values():
        spec["initial_population"] = 2
        spec["reproduction"]["enabled"] = False
    c["species"]["deer"]["display"] = {"color": "#b4e580", "male_color": "#ca94ff"}
    c["world"]["max_steps"] = 6
    c["experiment"]["runs"] = 2
    return c


def body(i, species, x, y, sex="female", heading=0):
    return Animal(i, species, x, y, heading, 40, 30, sex=sex, last_action="reproduce")


class MultipleSpeciesTests(unittest.TestCase):
    def test_renamed_species_independent_private_controllers_and_mixed_algorithms(self):
        c = configuration()
        c["species"]["deer"]["body"]["max_health"] = 45
        for name, kind in (("deer", "ppo"), ("wolves", "grpo")):
            template = environment(kind, 0)
            key = name + "_brain"
            c["brains"][key] = copy.deepcopy(
                template["brains"][template["species"]["prey"]["brain"]]
            )
            c["species"][name]["brain"] = key
        e = Experiment(c)
        for i, a in e.world.animals.items():
            self.assertEqual(
                e.brains[i].config["type"],
                {"deer": "ppo", "wolves": "grpo"}.get(a.species, "q_network"),
            )
        self.assertEqual(set(e.world.metrics()["populations"]), set(NAMES))
        self.assertEqual(e.world.metrics()["populations"], dict.fromkeys(NAMES, 2))
        for a in e.world.animals.values():
            self.assertEqual(
                a.health,
                (
                    45
                    if a.species == "deer"
                    else c["species"][a.species]["body"]["max_health"]
                ),
            )
        ids = [i for i, a in e.world.animals.items() if a.sex == "female"]
        original = [copy.deepcopy(e.brains[i].checkpoint()) for i in ids[1:]]
        e.brains[ids[0]].weights["w1"][0] += 1
        self.assertEqual(original, [e.brains[i].checkpoint() for i in ids[1:]])
        for _ in range(4):
            e.step()

    def test_all_algorithms_resume_with_multiple_species_across_run_boundaries(self):
        for kind in KINDS:
            a = Experiment(configuration(kind))
            a.step()
            b = Experiment.restore(json.loads(json.dumps(a.checkpoint())))
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "state.bin.xz"
                a.write_checkpoint(path)
                c = Experiment.restore(read_checkpoint(path))
            while not a.done:
                a.step()
                b.step()
                c.step()
                self.assertEqual(a.checkpoint(), b.checkpoint(), kind)
                self.assertEqual(a.checkpoint(), c.checkpoint(), kind)

    def test_perception_and_masks_use_diets_not_species_names(self):
        w = World(configuration(), 42)
        w.obstacles.clear()
        w.animals = {
            0: body(0, "rabbits", 1, 2),
            1: body(1, "deer", 3, 2),
            2: body(2, "foxes", 2, 2, heading=1),
            3: body(3, "wolves", 4, 2),
        }
        for _ in range(2):
            observations = w.observe_all()
            for i, a in w.animals.items():
                self.assertEqual(observations[i], reference_observation(w, a))
            w.animals[1].x = 2
            w.animals[1].y = 3
        w.animals[1].x = 3
        w.animals[1].y = 2
        w.config["species"]["foxes"]["body"]["attack_damage"] = 30
        self.assertEqual(action_mask(w, w.animals[2], ["rest", "attack"]), (True, True))
        w.animals[1].x = 0
        w.animals[3].x = 3
        self.assertEqual(
            action_mask(w, w.animals[2], ["rest", "attack"]), (True, False)
        )
        self.assertEqual(set(central_contexts(w)), set(NAMES))

    def test_different_predators_can_hunt_different_prey_species_but_not_each_other(
        self,
    ):
        w = World(configuration(), 42)
        w.obstacles.clear()
        w.animals = {
            0: body(0, "rabbits", 1, 1),
            1: body(1, "deer", 3, 1),
            2: body(2, "foxes", 0, 1, heading=1),
            3: body(3, "wolves", 4, 1, heading=3),
        }
        for name in ("foxes", "wolves"):
            w.config["species"][name]["body"]["attack_damage"] = 30
        result = w.step({0: "rest", 1: "rest", 2: "attack", 3: "attack"})
        self.assertEqual(result.dead_ids, {0, 1})
        self.assertEqual(dict(w.kills), {"foxes": 1, "wolves": 1})
        w.animals[2].x, w.animals[2].y = 1, 1
        w.animals[3].x, w.animals[3].y = 2, 1
        before = w.animals[3].health
        w.step({2: "attack", 3: "rest"})
        self.assertEqual(w.animals[3].health, before)

    def test_pairing_is_same_species_even_for_same_diet(self):
        w = World(configuration(), 42)
        w.obstacles.clear()
        for spec in w.config["species"].values():
            spec["reproduction"]["enabled"] = True
        w.animals = {0: body(0, "rabbits", 1, 1), 1: body(1, "deer", 2, 1, sex="male")}
        for a in w.animals.values():
            a.age = 100
        events = []
        w._reproduce(events)
        self.assertEqual(events, [])
        w.animals[1].species = "rabbits"
        w.next_animal_id = 2
        w._reproduce(events)
        self.assertEqual(len(events), 1)
        child = w.animals[events[0].target]
        self.assertEqual(child.species, "rabbits")
        self.assertEqual((child.parent_id, child.second_parent_id), (0, 1))

    def test_archives_and_selectors_are_separate_for_eight_species_sex_groups(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            e = Experiment(configuration())
            archive = export_champions(e, root / "source/champions/run_0001.json")
            entries = json.loads((archive / "manifest.json").read_text())["entries"]
            self.assertEqual(len(entries), 36)
            self.assertEqual(
                {(x["species"], x["sex"]) for x in entries},
                {(s, sex) for s in NAMES for sex in ("female", "male")},
            )
            c = configuration()
            apply_brain_overrides(
                c, ["deer=fresh", "wolves:male=lifespan"], root / "source"
            )
            loaded = Experiment(c)
            self.assertNotIn("deer", loaded.initial_models)
            self.assertEqual(
                loaded.initial_models["wolves"]["male"]["provenance"]["criterion"],
                "lifespan",
            )
            self.assertEqual(set(loaded.initial_models), {"rabbits", "foxes", "wolves"})

    def test_binary_recording_analysis_and_telemetry_keep_every_species(self):
        c = configuration()
        c["recording"].update(encoding="binary", compression="xz", frames_per_chunk=2)
        e = Experiment(c)
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "run"
            r = Recorder(folder, c, e.mode)
            r.begin_run(e)
            for _ in range(3):
                e.step()
                r.capture(e)
            r.close(False)
            store = RecordingStore(d)
            first = load_chunk(store.chunk("run", 1, 0))
            last = load_chunk(store.chunk("run", 1, 1))
            stats = summarize(c, last[-1], last[-2])
            self.assertEqual(set(stats["species"]), set(NAMES))
            self.assertEqual(
                set(analyze_recording(folder)["runs"][0]["species"]), set(NAMES)
            )
            self.assertEqual({a["species"] for a in first[0]["animals"]}, set(NAMES))
            metadata = store.metadata("run")
            self.assertEqual(
                metadata["config"]["species"]["deer"]["display"],
                c["species"]["deer"]["display"],
            )

    def test_cli_metrics_columns_and_presets_stay_configurable(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "environment.json"
            path.write_text(json.dumps(configuration()))
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ecosystem.cli",
                    str(path),
                    "--runs",
                    "1",
                    "--steps",
                    "2",
                    "--no-progress",
                    "--no-library",
                    "--output",
                    str(root / "results"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            folder = next((root / "results").iterdir())
            with (folder / "metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            for name in NAMES:
                self.assertIn(name + "_population", rows[0])
        for p in (Path(__file__).resolve().parents[1] / "environments").glob("*.json"):
            self.assertEqual(set(load(p)["species"]), {"prey", "predator"})

    def test_display_and_species_identifiers_are_validated(self):
        for display in (
            {"color": "red"},
            {"male_color": "#123"},
            {"color": None},
            {"unknown": "#123456"},
        ):
            c = configuration()
            c["species"]["deer"]["display"] = display
            with self.assertRaisesRegex(ValueError, "display"):
                validate(c)
        for name in ("", "   ", 12):
            c = configuration()
            c["species"][name] = c["species"].pop("deer")
            with self.assertRaisesRegex(ValueError, "species names"):
                validate(c)


if __name__ == "__main__":
    unittest.main()
