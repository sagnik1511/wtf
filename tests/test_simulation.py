import copy
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ecosystem.brain import Brain, Experience, INPUTS
from ecosystem.config import load, validate
from ecosystem.runner import Experiment
from ecosystem.server import State, make_handler
from ecosystem.world import Animal, Resource, World

ROOT = Path(__file__).resolve().parents[1]


def config():
    c = load(ROOT / "environments/baseline.json")
    c["recording"].update(encoding="delta", compression="gzip")
    for spec in c["species"].values():
        spec.pop('mutation',None)
        spec["reproduction"].update(enabled=False, age_deviation_pct=0, cooldown_deviation_pct=0)
    c["species"]["prey"]["initial_population"] = 3
    c["species"]["predator"]["initial_population"] = 1
    # Mechanics fixtures intentionally exercise multi-hit combat, independently
    # of the baseline scenario's stronger one-hit predators.
    c["species"]["predator"]["body"]["attack_damage"] = 10
    c["resources"]["patch_count"] = 8
    c["world"].update(width=10, height=8, max_steps=20)
    c["experiment"].update(
        runs=2, record_metrics_every_steps=2, world_seed_policy="increment"
    )
    b = c["brains"]["small_q"]
    b.update(
        hidden_layers=[4],
        batch_size=2,
        learning_starts_after=2,
        train_every_steps=1,
        target_update_every_training_steps=3,
    )
    return c


def empty_world():
    w = World(config(), 42)
    w.obstacles = set()
    w.animals = {}
    w.resources = {}
    return w


def animal(i, species, x, y, heading=0, energy=20.0, health=30.0):
    return Animal(i, species, x, y, heading, energy, health)


class ConfigTests(unittest.TestCase):
    def test_all_examples(self):
        for path in (ROOT / "environments").glob("*.json"):
            validate(load(path))

    def test_invalid_values_and_unknown_fields(self):
        cases = [
            (["world", "width"], True),
            (["resources", "growth_per_step"], -1),
            (["world", "max_steps"], 0),
            (["species", "prey", "body", "initial_energy"], 1000),
            (["brains", "small_q", "batch_size"], 999999),
        ]
        for keys, value in cases:
            c = config()
            target = c
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
            with (
                self.subTest(keys=keys),
                self.assertRaisesRegex(
                    ValueError, keys[-1] if keys[-1] != "batch_size" else "batch size"
                ),
            ):
                validate(c)
        c = config()
        c["world"]["widht"] = 40
        with self.assertRaisesRegex(ValueError, "widht: unknown field"):
            validate(c)

    def test_invalid_capacity_and_diet(self):
        c = config()
        c["species"]["prey"]["allowed_actions"].append("attack")
        with self.assertRaises(ValueError):
            validate(c)
        c = config()
        c["species"]["prey"]["initial_population"] = 500
        with self.assertRaisesRegex(ValueError, "insufficient free"):
            validate(c)


class WorldTests(unittest.TestCase):
    def test_resources_regrow_and_cap(self):
        w = empty_world()
        w.resources[(1, 1)] = Resource(1, 1, 19.99)
        w.step({})
        self.assertEqual(w.resources[(1, 1)].quantity, 20)

    def test_contested_movement_never_overlaps(self):
        w = empty_world()
        w.animals = {0: animal(0, "prey", 1, 2, 1), 1: animal(1, "prey", 3, 2, 3)}
        result = w.step({0: "move_forward", 1: "move_forward"})
        self.assertEqual(sum(a.position == (2, 2) for a in w.animals.values()), 1)
        self.assertEqual(sum(e.kind == "movement_blocked" for e in result.events), 1)

    def test_obstacle_and_occupied_cells_block(self):
        w = empty_world()
        w.animals = {0: animal(0, "prey", 1, 2, 1), 1: animal(1, "prey", 2, 2, 1)}
        w.obstacles = {(3, 2)}
        w.step({0: "move_forward", 1: "move_forward"})
        self.assertEqual(w.animals[0].position, (1, 2))
        self.assertEqual(w.animals[1].position, (2, 2))

    def test_feeding_rewards_only_actual_energy(self):
        w = empty_world()
        w.animals = {0: animal(0, "prey", 2, 2, energy=99.0)}
        w.resources[(2, 2)] = Resource(2, 2, 10)
        result = w.step({0: "eat"})
        self.assertAlmostEqual(result.rewards[0], 0.1)
        self.assertAlmostEqual(w.animals[0].energy, 99.9)
        self.assertAlmostEqual(w.resources[(2, 2)].quantity, 9.6)

    def test_simultaneous_attack_single_kill(self):
        w = empty_world()
        w.animals = {
            0: animal(0, "prey", 2, 2, health=15),
            1: animal(1, "predator", 1, 2, 1, energy=20),
            2: animal(2, "predator", 3, 2, 3, energy=20),
        }
        result = w.step({0: "rest", 1: "attack", 2: "attack"})
        self.assertEqual(result.dead_ids, {0})
        self.assertEqual(sum(e.kind == "kill" for e in result.events), 1)
        self.assertEqual(
            sum(e.amount for e in result.events if e.kind == "energy_gained"), 40
        )
        self.assertAlmostEqual(result.rewards[0], -13)
        self.assertEqual(w.deaths["predation"], 1)

    def test_starvation_and_terminal_reward(self):
        w = empty_world()
        w.animals = {0: animal(0, "prey", 1, 1, energy=0.0, health=0.5)}
        result = w.step({0: "rest"})
        self.assertEqual(result.dead_ids, {0})
        self.assertAlmostEqual(result.rewards[0], -10.1)
        self.assertNotIn(0, result.observations)

    def test_visibility_and_observation_shape(self):
        w = empty_world()
        a = animal(0, "prey", 1, 1)
        w.animals = {0: a}
        w.resources[(3, 1)] = Resource(3, 1, 10)
        self.assertEqual(len(w.observe(a)), INPUTS)
        self.assertEqual(w.observe(a)[11], 1.0)
        w.obstacles.add((2, 1))
        self.assertEqual(w.observe(a)[11], 0.0)


class LearningTests(unittest.TestCase):
    def test_positive_feedback_improves_action_value(self):
        c = config()["brains"]["small_q"]
        c.update(
            batch_size=1,
            learning_starts_after=1,
            learning_rate=0.1,
            discount_factor=0.0,
        )
        b = Brain(c, ["rest", "eat"], 4, 9)
        observation = (0.0,) * INPUTS
        before = b.forward(observation)[1][1]
        for _ in range(40):
            b.learn(Experience(observation, "eat", 1.0, None, True))
        self.assertGreater(b.forward(observation)[1][1], before + 0.5)
        self.assertEqual(b.choose(observation, explore=False), "eat")

    def test_independent_brains(self):
        e = Experiment(config())
        a, b = e.brains[0], e.brains[1]
        self.assertEqual(a.weights, b.weights)
        saved = copy.deepcopy(b.checkpoint())
        for _ in range(4):
            a.learn(Experience((0.0,) * INPUTS, "eat", 1.0, None, True))
        self.assertEqual(b.checkpoint(), saved)
        self.assertNotEqual(a.weights, b.weights)

    def test_time_limit_is_truncation(self):
        c = config()
        c["world"]["max_steps"] = 1
        e = Experiment(c)
        e.step()
        experience = e.brains[0].replay[-1]
        self.assertTrue(experience.truncated)
        self.assertFalse(experience.terminated)
        self.assertIsNotNone(experience.next_observation)

    def test_death_updates_controller(self):
        e = Experiment(config())
        e.world.animals[0].energy = 0
        e.world.animals[0].health = 0.1
        e.observations = e.world.observe_all()
        brain = e.brains[0]
        e.step()
        self.assertTrue(brain.replay[-1].terminated)
        self.assertLess(brain.replay[-1].reward, -10)
        self.assertNotIn(0, e.brains)


class ExperimentTests(unittest.TestCase):
    def test_same_seed_reproduces_learning(self):
        a, b = Experiment(config()), Experiment(config())
        for _ in range(12):
            a.step()
            b.step()
        self.assertEqual(a.checkpoint(), b.checkpoint())

    def test_json_checkpoint_resumes_exactly(self):
        a = Experiment(config())
        for _ in range(7):
            a.step()
        b = Experiment.restore(json.loads(json.dumps(a.checkpoint())))
        for _ in range(22):
            a.step()
            b.step()
        self.assertEqual(a.checkpoint(), b.checkpoint())

    def test_next_run_uses_champion_or_original_weights(self):
        from ecosystem.brains import model_snapshot
        for chain in (True, False):
            c = config()
            c["experiment"]["chain_best_brains"] = chain
            c["world"]["max_steps"] = 3
            e = Experiment(c)
            original = model_snapshot(e.brains[0])
            for _ in range(3):
                e.step()
            expected = e.champions.entries["prey"]["food"]["model"] if chain else original
            e.next_run()
            self.assertEqual(model_snapshot(e.brains[0]), expected)
            self.assertEqual(e.brains[0].steps, 0)
            self.assertEqual(e.world.tick, 0)
            self.assertEqual(e.world.seed, 43)

    def test_evaluation_does_not_change_brains(self):
        e = Experiment(config(), "evaluate")
        before = {i: b.checkpoint() for i, b in e.brains.items()}
        for _ in range(5):
            e.step()
        self.assertEqual(before, {i: b.checkpoint() for i, b in e.brains.items()})

    def test_random_mode_does_not_train(self):
        e = Experiment(config(), "random")
        for _ in range(5):
            e.step()
        self.assertTrue(all(b.steps == 0 for b in e.brains.values()))


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ecosystem.recording import Recorder

        cls.directory = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.directory.name) / "recorded_test"
        e = Experiment(config())
        r = Recorder(cls.folder, e.config, e.mode)
        r.begin_run(e)
        for _ in range(3):
            e.step()
            r.capture(e)
        r.close(completed=False)
        cls.state = State(cls.directory.name)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.state))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.directory.cleanup()

    def request(self, path, data=None):
        req = Request(
            self.url + path,
            data=None if data is None else json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urlopen(req, timeout=10)

    def test_viewer_and_recording_api(self):
        with self.request("/") as response:
            self.assertIn(b"Wildlife Trial Framework", response.read())
        with self.request("/api/recordings") as response:
            self.assertEqual(json.load(response)[0]["id"], "recorded_test")
        with self.request("/api/recordings/recorded_test") as response:
            self.assertEqual(json.load(response)["runs"][0]["frame_count"], 4)
        with self.request("/api/recordings/recorded_test/runs/1/chunks/0") as response:
            frames = json.load(response)
            self.assertEqual([f["tick"] for f in frames], [0, 1, 2, 3])
        with self.request("/api/recordings/recorded_test/runs/1/metrics") as response:
            self.assertEqual([m["tick"] for m in json.load(response)], [0, 1, 2, 3])
        self.assertFalse(hasattr(self.state, "experiment"))

    def test_behavior_analysis_is_read_only_and_matches_recorded_steps(self):
        with self.request("/api/recordings/recorded_test/runs/1/analysis") as response:
            report = json.load(response)
        self.assertEqual(report["analysis_version"], 1)
        self.assertEqual(report["runs"][0]["frames"], 4)
        self.assertEqual(report["runs"][0]["coverage"]["step_fraction"], 1)
        self.assertEqual(set(report["runs"][0]["species"]), {"prey", "predator"})
        self.assertFalse((self.folder / "analysis.json").exists())
        with self.assertRaises(HTTPError) as raised:
            self.request("/api/recordings/recorded_test/runs/99/analysis")
        self.assertEqual(raised.exception.code, 404)

    def test_replay_gzip_negotiation_and_plain_fallback(self):
        import gzip
        path = self.url + "/api/recordings/recorded_test/runs/1/chunks/0"
        req = Request(path, headers={"Accept-Encoding": "gzip"})
        with urlopen(req, timeout=10) as response:
            self.assertEqual(response.headers.get("Content-Encoding"), "gzip")
            self.assertEqual(set(response.headers.get("Vary").split(", ")), {"Accept-Encoding", "X-WTF-Replay"})
            frames = json.loads(gzip.decompress(response.read()))
        self.assertEqual(len(frames), 4)
        from ecosystem.replay_codec import decode_chunk
        req = Request(path, headers={"Accept-Encoding": "gzip", "X-WTF-Replay": "delta-v1"})
        with urlopen(req, timeout=10) as response:
            payload = json.loads(gzip.decompress(response.read()))
        self.assertEqual(payload["codec"], "wtf-delta-v1")
        self.assertEqual(decode_chunk(payload), frames)
        req = Request(path, headers={"Accept-Encoding": "identity", "X-WTF-Replay": "delta-v1"})
        with urlopen(req, timeout=10) as response:
            self.assertIsNone(response.headers.get("Content-Encoding"))
            self.assertEqual(json.load(response), payload)
        req = Request(path, headers={"Accept-Encoding": "gzip;q=0"})
        with urlopen(req, timeout=10) as response:
            self.assertIsNone(response.headers.get("Content-Encoding"))
            self.assertEqual(json.load(response), frames)

    def test_per_step_telemetry_endpoint_and_validation(self):
        with self.request("/api/recordings/recorded_test/runs/1/telemetry/0") as response:
            page=json.load(response)
        self.assertEqual(page["page_size"],128)
        self.assertEqual([p["tick"] for p in page["points"]],[0,1,2,3])
        self.assertEqual(set(page["points"][0]["species"]),{"prey","predator"})
        for path,code in [("/api/recordings/recorded_test/runs/1/telemetry/1",404),
                          ("/api/recordings/recorded_test/runs/1/telemetry/-1",400)]:
            with self.assertRaises(HTTPError) as raised:self.request(path)
            self.assertEqual(raised.exception.code,code)

    def test_all_live_simulation_endpoints_removed(self):
        for path in ("/api/step", "/api/load", "/api/restore"):
            with self.subTest(path=path), self.assertRaises(HTTPError) as raised:
                self.request(path, {"count": 1})
            self.assertEqual(raised.exception.code, 405)
        with self.assertRaises(HTTPError) as raised:
            self.request("/api/checkpoint")
        self.assertEqual(raised.exception.code, 404)

    def test_traversal_and_missing_chunks(self):
        for path in (
            "/../ecosystem/config.py",
            "/api/recordings/recorded_test/runs/1/chunks/9999",
        ):
            with self.subTest(path=path), self.assertRaises(HTTPError) as raised:
                self.request(path)
            self.assertEqual(raised.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
