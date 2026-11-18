"""Run-local founder families; retain counters, never deceased controllers."""

import copy


def lineage_policy(config, species):
    spec = config["species"][species]
    return dict(
        version=1,
        family="founder_descendants",
        diet=spec["diet"],
        min_age_steps=spec.get("champion_selection", {}).get(
            "lineage_min_age_steps",
            max(1, spec.get("reproduction", {}).get("min_age_steps", 1)),
        ),
    )


class LineageBook:
    """A descendant contributes once to each distinct founding ancestor's family.

    Founder membership is cached at birth. Incremental family statistics cost
    O(living animals * distinct founding ancestors), not a full pedigree walk
    per animal per step. Checkpoints retain the compact ledger and aggregates
    to preserve exact floating-point accumulation across continuation.
    """

    def __init__(self, config, state=None):
        self.policies = {s: lineage_policy(config, s) for s in config["species"]}
        self.partial = (state or {}).get("partial", False)
        self.nodes = copy.deepcopy((state or {}).get("nodes", {}))
        self.families = {
            int(key): copy.deepcopy(value)
            for key, value in (state or {}).get("families", {}).items()
        }
        if self.nodes and not self.families:
            for key, node in self.nodes.items():
                self._add(int(key), node, new=True)

    def _add(self, animal_id, node, previous=None, new=False):
        previous = previous or {}
        for root in node["roots"]:
            family = self.families.setdefault(
                root,
                dict(
                    members=0,
                    viable_descendants=0,
                    kills=0,
                    energy_gained=0.0,
                    animal_steps=0,
                ),
            )
            family["members"] += int(new)
            family["viable_descendants"] += int(
                node["viable"]
                and not previous.get("viable", False)
                and root != animal_id
            )
            for field in ("kills", "energy_gained", "animal_steps"):
                family[field] += node[field] - previous.get(field, 0)

    def observe(self, animals):
        # Parents always have lower IDs; sort also supports unordered callers.
        for animal in sorted(animals, key=lambda a: a.id):
            key = str(animal.id)
            old = self.nodes.get(key)
            if old is None:
                parents = [
                    self.nodes[str(p)]
                    for p in (animal.parent_id, animal.second_parent_id)
                    if str(p) in self.nodes
                    and self.nodes[str(p)]["species"] == animal.species
                ]
                roots = sorted({r for p in parents for r in p["roots"]})
                node = dict(
                    species=animal.species,
                    roots=roots or [animal.id],
                    generation=(
                        max(p["generation"] for p in parents) + 1 if parents else 0
                    ),
                )
                if len(parents) < 2 and animal.parent_id >= 0:
                    self.partial = True
            else:
                node = dict(old)
            node.update(
                animal_steps=animal.age,
                kills=animal.kills,
                energy_gained=animal.energy_gained,
                viable=animal.age >= self.policies[animal.species]["min_age_steps"],
            )
            self._add(animal.id, node, old, new=old is None)
            self.nodes[key] = node

    def rank(self, animal):
        node = self.nodes[str(animal.id)]
        predator = self.policies[animal.species]["diet"] == "prey"
        qualified = node["viable"] and (
            animal.kills > 0 if predator else animal.energy_gained > 0
        )
        own = (
            (animal.kills, animal.age)
            if predator
            else (animal.age, animal.energy_gained)
        )
        best = None
        for root in node["roots"]:
            family = self.families[root]
            time = family["animal_steps"]
            primary = family["kills"] if predator else time
            secondary = time if predator else family["energy_gained"]
            rate = (family["kills"] if predator else family["energy_gained"]) / max(
                1, time
            )
            rank = (
                node["generation"] if qualified else -1,
                primary if qualified else 0,
                family["viable_descendants"] if qualified else 0,
                secondary if qualified else 0,
                rate if qualified else 0,
                *own,
                animal.health,
            )
            if best is None or rank > best[0]:
                best = rank, dict(
                    founder_id=root,
                    generation=node["generation"],
                    qualified=qualified,
                    partial_history=self.partial,
                    **family,
                    kills_per_1000_animal_steps=1000 * family["kills"] / max(1, time),
                    energy_per_animal_step=family["energy_gained"] / max(1, time),
                )
        return best

    def checkpoint(self):
        return dict(
            version=1,
            partial=self.partial,
            nodes=copy.deepcopy(self.nodes),
            families={
                str(key): copy.deepcopy(value) for key, value in self.families.items()
            },
        )
