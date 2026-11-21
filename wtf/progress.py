"""Terminal progress only; never changes world or learning state."""

import sys
from tqdm import tqdm


class RunProgress:
    def __init__(self, mode="train", enabled=None, stream=None):
        self.mode = mode
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = (
            (mode == "train" and self.stream.isatty()) if enabled is None else enabled
        )
        self.bar = None

    def start(self, experiment, metrics=None):
        self.close()
        if not self.enabled or experiment.finished_run:
            return
        label = {"train": "Train", "evaluate": "Evaluate", "random": "Random"}[
            self.mode
        ]
        self.label = f'{label} run {experiment.run_index + 1}/{experiment.config["experiment"]["runs"]}'
        self.bar = tqdm(
            total=experiment.config["world"]["max_steps"],
            initial=experiment.world.tick,
            desc=self.label,
            unit="step",
            dynamic_ncols=True,
            mininterval=0.2,
            file=self.stream,
        )
        self._metrics(experiment.world.metrics() if metrics is None else metrics)

    def _metrics(self, metrics):
        self.bar.set_postfix(
            dict(
                metrics["populations"],
                kills=sum(metrics["kills"].values()),
                food=round(metrics["food"]),
            ),
            refresh=False,
        )

    def tick(self, experiment):
        if self.bar is None:
            return
        self.bar.update(experiment.world.tick - self.bar.n)
        if experiment.last_record is not None:
            self._metrics(experiment.last_record)
        if experiment.finished_run:
            if not experiment.world.animals:
                self.bar.total = experiment.world.tick
                self.bar.set_description_str(self.label + " (extinct)", refresh=False)
            self.close()

    def close(self):
        if self.bar is not None:
            self.bar.close()
            self.bar = None
