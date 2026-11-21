"""Executable contributor recipes and complete schema-field documentation."""

import ast
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from ecosystem.config import load, validate
from ecosystem.brain import Experience
from work.build_config_reference import render_reference

ROOT = Path(__file__).resolve().parents[1]


def excerpt(path, marker):
    text = (ROOT / path).read_text()
    return (
        text.split(f"<!-- {marker}:start -->")[1]
        .split(f"<!-- {marker}:end -->")[0]
        .split("```python\n")[1]
        .rsplit("```", 1)[0]
    )


class DocumentationContractsTests(unittest.TestCase):
    def test_every_field_is_defined_and_reference_is_current(self):
        schema = json.loads((ROOT / "environment.schema.json").read_text())
        self.assertEqual(
            render_reference(schema),
            (ROOT / "docs/configuration-reference.md").read_text(),
        )
        schema["properties"]["world"]["properties"]["new_undocumented_setting"] = {
            "type": "number"
        }
        with self.assertRaisesRegex(ValueError, "Missing field definition"):
            render_reference(schema)

    def test_command_reference_covers_every_declared_core_flag(self):
        reference = (ROOT / "docs/command-reference.md").read_text()
        sections = {
            "cli.py": reference.split("## Simulator")[1].split("## Playback server")[0],
            "server.py": reference.split("## Playback server")[1].split(
                "## Saved-run analyzer"
            )[0],
            "analyze.py": reference.split("## Saved-run analyzer")[1],
        }
        for filename, section in sections.items():
            tree = ast.parse((ROOT / "ecosystem" / filename).read_text())
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                ):
                    for argument in node.args:
                        if (
                            isinstance(argument, ast.Constant)
                            and isinstance(argument.value, str)
                            and argument.value.startswith("--")
                        ):
                            with self.subTest(command=filename, flag=argument.value):
                                self.assertIn(argument.value, section)

    def test_species_recipe_writes_valid_local_json_without_modifying_preset(self):
        original = (ROOT / "environments/maskable_ppo.json").read_text()
        code = excerpt("docs/species.md", "species-example")
        # Intercept writes and directory creation: execute the real recipe without
        # leaving tutorial experiments or changing an existing local configuration.
        with (
            patch("pathlib.Path.write_text", autospec=True) as write,
            patch("pathlib.Path.mkdir", autospec=True),
            patch("builtins.print"),
        ):
            namespace = {}
            exec(compile(code, "<species-recipe>", "exec"), namespace)
        config = namespace["config"]
        validate(config)
        self.assertEqual(set(config["species"]), {"prey", "deer", "predator"})
        self.assertEqual(
            sum(s["initial_population"] for s in config["species"].values()), 54
        )
        self.assertIsNot(
            config["species"]["prey"]["body"], config["species"]["deer"]["body"]
        )
        self.assertEqual(json.loads(write.call_args.args[1]), config)
        self.assertEqual(
            (ROOT / "environments/maskable_ppo.json").read_text(), original
        )

    def test_illustrative_policy_class_learns_and_restores_its_schedule(self):
        namespace = {}
        exec(
            compile(
                excerpt("docs/extending-policies.md", "policy-example"),
                "<policy-recipe>",
                "exec",
            ),
            namespace,
        )
        cls = namespace["ScheduledPPOBrain"]
        config = load(ROOT / "environments/ppo.json")
        template = copy.deepcopy(next(iter(config["brains"].values())))
        template.update(type="scheduled_ppo", rollout_steps=2, epochs=1, batch_size=2)
        brain = cls(template, ["rest", "turn_left"], 31, 47)
        obs = (0.0,) * 20
        before = copy.deepcopy(brain.config)
        for _ in range(2):
            action = brain.choose(obs)
            brain.learn(Experience(obs, action, 0.1, obs, False, False))
        self.assertGreater(brain.rollout_updates, 0)
        self.assertEqual(brain.config, before)
        restored = cls.restore(brain.checkpoint())
        self.assertEqual(restored.checkpoint(), brain.checkpoint())
        self.assertEqual(restored.choose(obs), brain.choose(obs))
        # The recipe is not advertised as an already-registered algorithm.
        next(iter(config["brains"].values()))["type"] = "scheduled_ppo"
        with self.assertRaises(ValueError):
            validate(config)
