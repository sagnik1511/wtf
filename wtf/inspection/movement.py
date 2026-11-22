"""One movement-arrival grid and prior positions; no replay history or snapshots."""

from array import array


class MovementCounts:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.counts = array("I", [0]) * (width * height)
        self.previous = {}
        self.tick = None

    def arrive(self, x, y):
        if (
            type(x) is int
            and type(y) is int
            and 0 <= x < self.width
            and 0 <= y < self.height
        ):
            self.counts[y * self.width + x] += 1

    def position(self, previous, animal):
        if (animal["x"], animal["y"]) != (previous["x"], previous["y"]):
            self.arrive(animal["x"], animal["y"])

    def terminal(self, previous, animals, event):
        i = event["actor"]
        position = event.get("position")
        if (
            event["kind"] == "died"
            and i in previous
            and i not in animals
            and position
            and tuple(position) != (previous[i]["x"], previous[i]["y"])
        ):
            self.arrive(*position)

    def consume(self, frame):
        animals = {a["id"]: {"x": a["x"], "y": a["y"]} for a in frame["animals"]}
        if self.tick is not None and frame["tick"] == self.tick + 1:
            for i, animal in animals.items():
                if i in self.previous:
                    self.position(self.previous[i], animal)
            for event in frame.get("events", []):
                self.terminal(self.previous, animals, event)
        self.previous, self.tick = animals, frame["tick"]

    def summary(self):
        return dict(
            width=self.width,
            height=self.height,
            counts=self.counts.tolist(),
            arrivals=sum(self.counts),
            maximum=max(self.counts, default=0),
            metric="changed_position_arrivals",
        )
