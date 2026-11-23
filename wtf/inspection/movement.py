"""Movement-arrival grids and shared prior positions; no replay history."""

from array import array


class MovementCounts:
    """Accumulate visited cells and movement counts from saved frames."""

    def __init__(self, width, height, species=None, species_names=()):
        self.width, self.height = width, height
        self.species = species
        self.counts = array("I", [0]) * (width * height)
        self.by_species = {
            name: MovementCounts(width, height, name) for name in species_names
        }
        self.previous = {}
        self.tick = None

    def arrive(self, x, y, species=None):
        """Count an arrival at the supplied grid location."""

        # Exact types intentionally reject bool/subclass substitutions in persisted
        # numeric contracts.
        # pylint: disable=unidiomatic-typecheck

        if (
            type(x) is int
            and type(y) is int
            and 0 <= x < self.width
            and 0 <= y < self.height
        ):
            self.counts[y * self.width + x] += 1
            if species in self.by_species:
                self.by_species[species].arrive(x, y)

    def position(self, previous, animal):
        """Return the entity's grid coordinates."""
        if (animal["x"], animal["y"]) != (previous["x"], previous["y"]):
            self.arrive(animal["x"], animal["y"], animal.get("species"))

    def terminal(self, previous, animals, event):
        """Count the final location of an animal leaving the replay population."""
        i = event["actor"]
        position = event.get("position")
        if (
            event["kind"] == "died"
            and i in previous
            and i not in animals
            and position
            and tuple(position) != (previous[i]["x"], previous[i]["y"])
        ):
            self.arrive(*position, species=previous[i].get("species"))

    def consume(self, frame):
        """Consume one saved frame into the movement aggregates."""
        animals = {
            a["id"]: {
                "x": a["x"],
                "y": a["y"],
                **({"species": a.get("species")} if self.by_species else {}),
            }
            for a in frame["animals"]
            if self.species is None or a.get("species") == self.species
        }
        living = (
            animals if self.species is None else {a["id"] for a in frame["animals"]}
        )
        if self.tick is not None and frame["tick"] == self.tick + 1:
            for i, animal in animals.items():
                if i in self.previous:
                    self.position(self.previous[i], animal)
            for event in frame.get("events", []):
                self.terminal(self.previous, living, event)
        self.previous, self.tick = animals, frame["tick"]

    def summary(self):
        """Return the accumulated movement statistics."""
        return {
            **({"species": self.species} if self.species is not None else {}),
            **(
                {
                    "by_species": {
                        name: group.summary() for name, group in self.by_species.items()
                    }
                }
                if self.by_species
                else {}
            ),
            "width": self.width,
            "height": self.height,
            "counts": self.counts.tolist(),
            "arrivals": sum(self.counts),
            "maximum": max(self.counts, default=0),
            "metric": "changed_position_arrivals",
        }
