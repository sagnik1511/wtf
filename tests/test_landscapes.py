"""Maintained landscapes must be navigable, spawn safely and round-trip to replay."""

# Test/fixture names and module narratives document the expected behavior.
# pylint: disable=missing-class-docstring,missing-function-docstring

from collections import deque
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from wtf.config import load
from wtf.experiment.runner import Experiment
from wtf.simulation.geography import Geography
from wtf.storage.recording import RecordingStore
from wtf.storage.replay_io import load_chunk
from work.build_geography_previews import preview

NAMES = (
    "grasslands",
    "seaside",
    "rolling_hills",
    "wetlands",
    "canyon",
    "oasis",
    "land_bridge",
)
DRY = {"grasslands", "rolling_hills"}


class LandscapeTests(unittest.TestCase):
    def test_spawn_actions_and_physical_connectivity(self):
        for name in NAMES:
            with self.subTest(name=name):
                c = load(f"environments/{name}.json")
                g = Geography(c["terrain"], 40, 28)
                self.assertEqual(bool(g.water), name not in DRY)
                self.assertEqual(c["species"]["prey"]["reproduction"]["mate_radius"], 2)
                self.assertEqual(
                    c["species"]["predator"]["reproduction"]["mate_radius"], 3
                )
                for s in c["species"].values():
                    self.assertEqual("swim" in s["allowed_actions"], name not in DRY)
                    self.assertEqual(c["brains"][s["brain"]]["type"], "ppo")
                free = {(x, y) for y in range(28) for x in range(40)} - g.obstacles
                seen = {min(free)}
                todo = deque(seen)
                while todo:
                    x, y = todo.popleft()
                    for q in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                        if (
                            q in free
                            and q not in seen
                            and abs(g.elevation(q) - g.elevation((x, y))) <= 1
                        ):
                            seen.add(q)
                            todo.append(q)
                self.assertEqual(seen, free, "stranded cells beyond cliffs/obstacles")
                e = Experiment(c)
                self.assertEqual(len(e.world.animals), 70)
                self.assertTrue(
                    all(
                        a.position in free and not g.depth(a.position)
                        for a in e.world.animals.values()
                    )
                )
                self.assertTrue(
                    all(p in free and not g.depth(p) for p in e.world.resources)
                )

    def test_rolling_hills_climbing_cost_and_feedback_track_actual_energy(self):

        # Keep setup, action and assertions together as one regression scenario.
        # pylint: disable=too-many-locals

        c = load("environments/rolling_hills.json")
        for species, uphill in (("prey", 3.0), ("predator", 4.0)):
            spec = c["species"][species]
            self.assertEqual(spec["locomotion"]["uphill_energy_cost"], uphill)
            for gain in (0, 1, -1):
                with self.subTest(species=species, gain=gain):
                    world = Experiment(c).world
                    geography = world.geography
                    start, destination = next(
                        ((x, y), (x + 1, y))
                        for y in range(28)
                        for x in range(39)
                        if (x, y) not in world.obstacles
                        and (x + 1, y) not in world.obstacles
                        and geography.elevation((x + 1, y))
                        - geography.elevation((x, y))
                        == gain
                    )
                    animal = next(
                        a for a in world.animals.values() if a.species == species
                    )
                    animal.x, animal.y = start
                    animal.heading = 1
                    animal.energy = 50
                    health = animal.health
                    world.animals = {animal.id: animal}
                    world.resources = {}
                    result = world.step({animal.id: "move_forward"})
                    self.assertEqual(animal.position, destination)
                    extra = uphill * max(0, gain)
                    cost = (
                        spec["body"]["living_cost"]
                        + spec["body"]["movement_cost"]
                        + extra
                    )
                    self.assertAlmostEqual(animal.energy, 50 - cost)
                    self.assertAlmostEqual(
                        result.rewards[animal.id],
                        cost * spec["rewards"]["per_energy_cost"],
                    )
                    self.assertEqual(animal.health, health)
                    charges = [
                        e.amount
                        for e in result.events
                        if e.kind == "energy_spent" and e.cause == "uphill"
                    ]
                    self.assertEqual(charges, [extra] if extra else [])

    def test_map_landmarks_and_themed_preview_are_not_cosmetic_aliases(self):
        signatures = set()
        for name in NAMES:
            c = load(f"environments/{name}.json")
            t = c["terrain"]
            signatures.add(tuple(t["rows"]))
            svg = preview(c)
            self.assertIn("prefers-color-scheme: light", svg)
            self.assertIn("fill: #d8e6cc", svg)
            self.assertEqual(svg.count("<svg "), 1)
            self.assertEqual(svg.count("</svg>"), 1)
            self.assertEqual(svg.count('width="16" height="16"'), 40 * 28)
            if name in ("oasis",):
                self.assertTrue(
                    all(
                        not any(cell.get("flow", [0, 0]))
                        for cell in t["legend"].values()
                    )
                )
        self.assertEqual(len(signatures), len(NAMES))

    def test_cli_presets_embed_exact_maps_without_source_tree_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name in NAMES:
                with self.subTest(name=name):
                    output = Path(temporary) / name
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "wtf",
                            "run",
                            name,
                            "--steps",
                            "4",
                            "--runs",
                            "1",
                            "--seed",
                            "100",
                            "--no-library",
                            "--no-progress",
                            "--output",
                            str(output),
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    folder = next(output.iterdir())
                    saved = json.loads((folder / "environment.json").read_text())
                    source = load(f"environments/{name}.json")
                    self.assertEqual(saved["terrain"], source["terrain"])
                    self.assertNotIn("map_file", saved["terrain"])
                    manifest = json.loads((folder / "manifest.json").read_text())
                    self.assertEqual(manifest["observation_version"], 4)
                    store = RecordingStore(output)
                    metadata = store.metadata(folder.name)
                    self.assertEqual(metadata["config"]["terrain"], source["terrain"])
                    self.assertGreater(metadata["runs"][0]["frame_count"], 0)
                    self.assertTrue(load_chunk(store.chunk(folder.name, 1, 0)))
