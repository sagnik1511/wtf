"""Terminal progress only; never changes world or learning state."""

# Black owns code layout; long literal payloads, URLs and regression names retain
# their exact text.
# pylint: disable=line-too-long


import sys
from tqdm import tqdm


class RunProgress:
    """Manage an optional progress bar across runs and simulation steps."""

    def __init__(self, mode="train", enabled=None, stream=None):

        # bar names the injected progress-bar factory, not placeholder application
        # code.
        # pylint: disable=disallowed-name

        self.mode = mode
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = (
            (mode == "train" and self.stream.isatty()) if enabled is None else enabled
        )
        self.bar = None

    def start(self, experiment, metrics=None):
        """Start progress reporting for the current episode."""

        # This cache or restored state is rebuilt at its owning lifecycle boundary,
        # not eagerly.
        # pylint: disable=attribute-defined-outside-init

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
        """Advance the optional progress display to the saved-world step."""
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
        """Release this component's open output or reporting resources."""
        if self.bar is not None:
            self.bar.close()
            self.bar = None
