"""Strict environment validation. No third-party packages required."""

import json
import math
import re
from pathlib import Path

ACTIONS = ("move_forward", "turn_left", "turn_right", "rest", "eat", "attack")


def fail(path, message):
    raise ValueError(f"{path}: {message}")


def obj(value, path, fields, optional=()):
    if not isinstance(value, dict):
        fail(path, "must be an object")
    for key in value:
        if key not in fields and key not in optional:
            fail(f"{path}.{key}", "unknown field")
    for key in fields:
        if key not in value:
            fail(f"{path}.{key}", "required field")


def num(value, path, low=0, high=float("inf"), integer=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
        or (integer and not isinstance(value, int))
    ):
        fail(
            path,
            f'must be {"an integer" if integer else "a number"} between {low} and {high}',
        )


def validate(c):
    obj(
        c,
        "config",
        (
            "schema_version",
            "name",
            "description",
            "seed",
            "world",
            "obstacles",
            "resources",
            "species",
            "brains",
            "experiment",
        ),
        optional=("recording",),
    )
    if type(c["schema_version"]) is not int or c["schema_version"] != 1:
        fail("schema_version", "only version 1 is supported")
    if not isinstance(c["name"], str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", c["name"]):
        fail("name", "use letters, digits, underscores or hyphens")
    if not isinstance(c["description"], str):
        fail("description", "must be a string")
    num(c["seed"], "seed", 0, 2**32 - 1, True)
    w = c["world"]
    obj(w, "world", ("width", "height", "boundary", "max_steps"))
    for k in ("width", "height"):
        num(w[k], f"world.{k}", 3, 256, True)
    num(w["max_steps"], "world.max_steps", 1, 10000000, True)
    if w["boundary"] != "closed":
        fail("world.boundary", "must be closed")
    o = c["obstacles"]
    obj(o, "obstacles", ("generation", "occupied_fraction", "explicit_cells"))
    if o["generation"] not in ("scattered", "explicit"):
        fail("obstacles.generation", "must be scattered or explicit")
    num(o["occupied_fraction"], "obstacles.occupied_fraction", 0, 0.8)
    if not isinstance(o["explicit_cells"], list):
        fail("obstacles.explicit_cells", "must be an array")
    cells = set()
    for i, p in enumerate(o["explicit_cells"]):
        if not isinstance(p, list) or len(p) != 2:
            fail(f"obstacles.explicit_cells.{i}", "must be [x, y]")
        num(p[0], f"obstacles.explicit_cells.{i}.x", 0, w["width"] - 1, True)
        num(p[1], f"obstacles.explicit_cells.{i}.y", 0, w["height"] - 1, True)
        if tuple(p) in cells:
            fail("obstacles.explicit_cells", "duplicate cell")
        cells.add(tuple(p))
    if o["generation"] == "scattered" and cells:
        fail("obstacles.explicit_cells", "must be empty for scattered generation")
    r = c["resources"]
    obj(
        r,
        "resources",
        (
            "patch_count",
            "capacity_per_patch",
            "initial_quantity_per_patch",
            "growth_per_step",
        ),
    )
    num(r["patch_count"], "resources.patch_count", 0, w["width"] * w["height"], True)
    num(r["capacity_per_patch"], "resources.capacity_per_patch", 0.001)
    num(
        r["initial_quantity_per_patch"],
        "resources.initial_quantity_per_patch",
        0,
        r["capacity_per_patch"],
    )
    num(r["growth_per_step"], "resources.growth_per_step")
    if not isinstance(c["brains"], dict) or not c["brains"]:
        fail("brains", "at least one template required")
    for name, b in c["brains"].items():
        p = f"brains.{name}"
        if not isinstance(b, dict) or "type" not in b:
            fail(p + ".type", "required brain type")
        if b["type"] == "q_network":
            obj(
                b,
                p,
                (
                    "type",
                    "hidden_layers",
                    "learning_rate",
                    "discount_factor",
                    "batch_size",
                    "replay_capacity",
                    "learning_starts_after",
                    "train_every_steps",
                    "target_update_every_training_steps",
                    "exploration",
                ),
            )
            if not isinstance(b["hidden_layers"], list) or len(b["hidden_layers"]) != 1:
                fail(
                    p + ".hidden_layers",
                    "exactly one hidden layer required for q_network",
                )
            num(b["hidden_layers"][0], p + ".hidden_layers.0", 1, 256, True)
            num(b["learning_rate"], p + ".learning_rate", 0.000001, 1)
            num(b["discount_factor"], p + ".discount_factor", 0, 1)
            for k in (
                "batch_size",
                "replay_capacity",
                "learning_starts_after",
                "train_every_steps",
                "target_update_every_training_steps",
            ):
                num(b[k], f"{p}.{k}", 1, 1000000, True)
            if max(b["batch_size"], b["learning_starts_after"]) > b["replay_capacity"]:
                fail(p, "batch size and learning start must not exceed replay capacity")
            e = b["exploration"]
            obj(e, p + ".exploration", ("initial", "minimum", "decay_per_decision"))
            for k in e:
                num(e[k], f"{p}.exploration.{k}", 0, 1)
            if e["minimum"] > e["initial"]:
                fail(p + ".exploration.minimum", "must not exceed initial")
        elif b["type"] == "ppo":
            obj(
                b,
                p,
                (
                    "type",
                    "network",
                    "learning_rate",
                    "discount_factor",
                    "gae_lambda",
                    "clip_ratio",
                    "entropy_coefficient",
                    "value_coefficient",
                    "rollout_steps",
                    "epochs",
                    "batch_size",
                    "max_grad_norm",
                    "normalize_advantages",
                ),
                optional=("target_kl",),
            )
            network = b["network"]
            obj(
                network,
                p + ".network",
                ("type", "hidden_layers"),
                optional=("activation",),
            )
            if network["type"] != "mlp":
                fail(p + ".network.type", "must be mlp")
            if network.get("activation", "tanh") != "tanh":
                fail(p + ".network.activation", "must be tanh")
            layers = network["hidden_layers"]
            if not isinstance(layers, list) or not 1 <= len(layers) <= 4:
                fail(p + ".network.hidden_layers", "one to four hidden layers required")
            for i, size in enumerate(layers):
                num(size, f"{p}.network.hidden_layers.{i}", 1, 256, True)
            num(b["learning_rate"], p + ".learning_rate", 0.000001, 0.1)
            for key in ("discount_factor", "gae_lambda", "entropy_coefficient"):
                num(b[key], f"{p}.{key}", 0, 1)
            num(b["clip_ratio"], p + ".clip_ratio", 0.001, 1)
            num(b["value_coefficient"], p + ".value_coefficient", 0, 10)
            num(b["rollout_steps"], p + ".rollout_steps", 1, 4096, True)
            num(b["epochs"], p + ".epochs", 1, 20, True)
            num(b["batch_size"], p + ".batch_size", 1, b["rollout_steps"], True)
            num(b["max_grad_norm"], p + ".max_grad_norm", 0.00001, 100)
            if type(b["normalize_advantages"]) is not bool:
                fail(p + ".normalize_advantages", "must be boolean")
            if "target_kl" in b:
                num(b["target_kl"], p + ".target_kl", 0, 1)
        elif b["type"] in ("a2c", "reinforce", "grpo"):
            common = (
                "type",
                "network",
                "learning_rate",
                "discount_factor",
                "entropy_coefficient",
                "max_grad_norm",
            )
            extra = {
                "a2c": (
                    "gae_lambda",
                    "value_coefficient",
                    "rollout_steps",
                    "normalize_advantages",
                ),
                "reinforce": (
                    "max_episode_steps",
                    "baseline_decay",
                    "normalize_advantages",
                ),
                "grpo": (
                    "clip_ratio",
                    "kl_coefficient",
                    "group_size",
                    "rollout_horizon",
                    "train_every_steps",
                    "epochs",
                    "batch_size",
                ),
            }[b["type"]]
            obj(
                b,
                p,
                common + extra,
                optional=("target_kl",) if b["type"] == "grpo" else (),
            )
            network = b["network"]
            obj(
                network,
                p + ".network",
                ("type", "hidden_layers"),
                optional=("activation",),
            )
            if network["type"] != "mlp" or network.get("activation", "tanh") != "tanh":
                fail(p + ".network", "must use an mlp with tanh activation")
            layers = network["hidden_layers"]
            if not isinstance(layers, list) or not 1 <= len(layers) <= 4:
                fail(p + ".network.hidden_layers", "one to four hidden layers required")
            for i, size in enumerate(layers):
                num(size, f"{p}.network.hidden_layers.{i}", 1, 256, True)
            num(b["learning_rate"], p + ".learning_rate", 0.000001, 0.1)
            num(b["discount_factor"], p + ".discount_factor", 0, 1)
            num(b["entropy_coefficient"], p + ".entropy_coefficient", 0, 1)
            num(b["max_grad_norm"], p + ".max_grad_norm", 0.00001, 100)
            if b["type"] == "a2c":
                num(b["gae_lambda"], p + ".gae_lambda", 0, 1)
                num(b["value_coefficient"], p + ".value_coefficient", 0, 10)
                num(b["rollout_steps"], p + ".rollout_steps", 1, 4096, True)
            elif b["type"] == "reinforce":
                num(b["max_episode_steps"], p + ".max_episode_steps", 1, 10000000, True)
                num(b["baseline_decay"], p + ".baseline_decay", 0, 1)
            else:
                num(b["clip_ratio"], p + ".clip_ratio", 0.001, 1)
                num(b["kl_coefficient"], p + ".kl_coefficient", 0, 10)
                num(b["group_size"], p + ".group_size", 2, 8, True)
                num(b["rollout_horizon"], p + ".rollout_horizon", 1, 16, True)
                num(b["train_every_steps"], p + ".train_every_steps", 1, 1000000, True)
                num(b["epochs"], p + ".epochs", 1, 20, True)
                num(
                    b["batch_size"],
                    p + ".batch_size",
                    1,
                    b["group_size"] * b["rollout_horizon"],
                    True,
                )
                if "target_kl" in b:
                    num(b["target_kl"], p + ".target_kl", 0, 1)
            if (
                "normalize_advantages" in b
                and type(b["normalize_advantages"]) is not bool
            ):
                fail(p + ".normalize_advantages", "must be boolean")
        else:
            fail(
                p + ".type",
                "supported types are q_network, ppo, a2c, reinforce and grpo",
            )
    if not isinstance(c["species"], dict) or not c["species"]:
        fail("species", "at least one species required")
    population = 0
    for name, s in c["species"].items():
        p = f"species.{name}"
        obj(
            s,
            p,
            (
                "initial_population",
                "diet",
                "allowed_actions",
                "body",
                "brain",
                "rewards",
            ),
            optional=("reproduction", "initial_brain", "learning_rate_multiplier", "champion_selection"),
        )
        num(s["initial_population"], p + ".initial_population", 0, 10000, True)
        population += s["initial_population"]
        if s["diet"] not in ("resource", "prey"):
            fail(p + ".diet", "must be resource or prey")
        actions = s["allowed_actions"]
        if (
            not isinstance(actions, list)
            or not actions
            or any(not isinstance(a, str) or a not in ACTIONS for a in actions)
            or len(set(actions)) != len(actions)
        ):
            fail(p + ".allowed_actions", "must contain unique supported actions")
        required, forbidden = (
            ("eat", "attack") if s["diet"] == "resource" else ("attack", "eat")
        )
        if required not in actions or forbidden in actions:
            fail(p + ".allowed_actions", "feeding action incompatible with diet")
        if not isinstance(s["brain"], str) or s["brain"] not in c["brains"]:
            fail(p + ".brain", "unknown brain template")
        if "initial_brain" in s:
            initial = s["initial_brain"]
            ip = p + ".initial_brain"
            if isinstance(initial, dict) and "path" in initial:
                obj(initial, ip, ("path",))
                if not isinstance(initial["path"], str) or not initial["path"].strip():
                    fail(ip + ".path", "must be a nonempty saved-brain JSON path")
            else:
                obj(initial, ip, ("selection", "metric"))
                if initial["selection"] not in ("best", "latest"):
                    fail(ip + ".selection", "must be best or latest")
                if initial["metric"] not in ("health", "kills", "lifespan", "food", "balanced"):
                    fail(ip + ".metric", "must be health, kills, lifespan, food or balanced")
        if "champion_selection" in s:
            cp = p + ".champion_selection"
            policy = s['champion_selection']
            obj(policy,cp,('criterion',),optional=('survival_steps_per_kill','max_survival_bonus'))
            if policy['criterion'] not in ('health','kills','lifespan','food','balanced'):
                fail(cp+'.criterion','unknown champion criterion')
            if policy['criterion']=='balanced' and s['diet']!='prey':
                fail(cp+'.criterion','balanced selection requires a predator species')
            num(policy.get('survival_steps_per_kill',300),cp+'.survival_steps_per_kill',1,10000000)
            num(policy.get('max_survival_bonus',2),cp+'.max_survival_bonus',0,100)
        if s.get('initial_brain',{}).get('metric')=='balanced' and s['diet']!='prey':
            fail(p+'.initial_brain.metric','balanced selection requires a predator species')
        template = c["brains"][s["brain"]]
        multiplier = s.get("learning_rate_multiplier", 1.0)
        num(multiplier, p + ".learning_rate_multiplier", 0.01, 10)
        num(
            template["learning_rate"] * multiplier,
            p + ".effective_learning_rate",
            0.000001,
            1 if template["type"] == "q_network" else 0.1,
        )
        if (
            template["type"] == "reinforce"
            and template["max_episode_steps"] < w["max_steps"]
        ):
            fail(
                p + ".brain",
                "REINFORCE max_episode_steps must cover world.max_steps to retain complete episodes",
            )
        fields = [
            "initial_energy",
            "max_energy",
            "max_health",
            "vision_radius",
            "living_cost",
            "movement_cost",
            "starvation_damage",
        ]
        fields += (
            ["bite_size", "energy_per_food_unit"]
            if s["diet"] == "resource"
            else ["attack_damage", "attack_cost", "energy_per_kill"]
        )
        obj(s["body"], p + ".body", fields)
        for k in fields:
            num(s["body"][k], f"{p}.body.{k}")
        for k in ("max_energy", "max_health"):
            num(s["body"][k], f"{p}.body.{k}", 0.001)
        num(s["body"]["vision_radius"], p + ".body.vision_radius", 1, 256, True)
        if s["body"]["initial_energy"] > s["body"]["max_energy"]:
            fail(p + ".body.initial_energy", "exceeds maximum")
        if "reproduction" in s:
            r = s["reproduction"]
            rp = p + ".reproduction"
            obj(
                r,
                rp,
                (
                    "enabled",
                    "health_threshold",
                    "min_age_steps",
                    "cooldown_steps",
                    "min_energy",
                    "energy_cost",
                    "max_population",
                ),
            )
            if type(r["enabled"]) is not bool:
                fail(rp + ".enabled", "must be boolean")
            num(
                r["health_threshold"],
                rp + ".health_threshold",
                0,
                s["body"]["max_health"],
            )
            num(r["min_age_steps"], rp + ".min_age_steps", 1, 10000000, True)
            num(r["cooldown_steps"], rp + ".cooldown_steps", 1, 10000000, True)
            num(r["energy_cost"], rp + ".energy_cost", 0.001, s["body"]["max_energy"])
            num(
                r["min_energy"],
                rp + ".min_energy",
                r["energy_cost"],
                s["body"]["max_energy"],
            )
            num(r["max_population"], rp + ".max_population", 0, 10000, True)
        obj(
            s["rewards"],
            p + ".rewards",
            ("per_energy_gained", "per_damage_taken", "death"),
            optional=("per_damage_dealt",),
        )
        for k, value in s["rewards"].items():
            num(
                value,
                f"{p}.rewards.{k}",
                0 if k == "per_damage_dealt" else -100000,
                100000,
            )
    if not population:
        fail("species", "total population must be positive")
    blocked = (
        len(cells)
        if o["generation"] == "explicit"
        else int(w["width"] * w["height"] * o["occupied_fraction"])
    )
    free = w["width"] * w["height"] - blocked
    if max(population, c["resources"]["patch_count"]) > free:
        fail("world", "insufficient free cells for animals or resources")
    e = c["experiment"]
    obj(
        e,
        "experiment",
        (
            "runs",
            "retain_brains_between_runs",
            "world_seed_policy",
            "record_metrics_every_steps",
            "checkpoint_every_runs",
        ),
        optional=("grpo_updates_per_step", "chain_best_brains"),
    )
    for k in ("runs", "record_metrics_every_steps", "checkpoint_every_runs"):
        num(e[k], f"experiment.{k}", 1, 1000000, True)
    if "grpo_updates_per_step" in e:
        num(e["grpo_updates_per_step"], "experiment.grpo_updates_per_step", 1, 8, True)
    if "chain_best_brains" in e and type(e["chain_best_brains"]) is not bool:
        fail("experiment.chain_best_brains", "must be boolean")
    if type(e["retain_brains_between_runs"]) is not bool:
        fail("experiment.retain_brains_between_runs", "must be boolean")
    if e["world_seed_policy"] not in ("random", "fixed", "increment"):
        fail("experiment.world_seed_policy", "must be random, fixed, or increment")
    if "recording" in c:
        r = c["recording"]
        obj(
            r,
            "recording",
            (
                "enabled",
                "frame_every_steps",
                "frames_per_chunk",
                "diagnostics_every_steps",
            ),
            optional=("compression", "compression_level", "encoding"),
        )
        if r.get("encoding", "delta") not in ("delta", "snapshot"):
            fail("recording.encoding", "must be delta or snapshot")
        if r.get("compression", "gzip") not in ("gzip", "none"):
            fail("recording.compression", "must be gzip or none")
        if "compression_level" in r:
            num(r["compression_level"], "recording.compression_level", 1, 9, True)
        if type(r["enabled"]) is not bool:
            fail("recording.enabled", "must be boolean")
        num(r["frame_every_steps"], "recording.frame_every_steps", 1, 1000000, True)
        num(r["frames_per_chunk"], "recording.frames_per_chunk", 1, 256, True)
        num(
            r["diagnostics_every_steps"],
            "recording.diagnostics_every_steps",
            0,
            1000000,
            True,
        )
    return c


def load(path):
    return validate(json.loads(Path(path).read_text()))
