"""Run-owned rolling recovery files and bounded periodic warm-start exports."""

import json
from pathlib import Path
import shutil

from wtf.storage.model_io import ModelEncodings
from wtf.storage.library import export_champions
from wtf.storage.recording import write_json


class ArtifactRetention:
    """Own only files published by this instance, never scan/prune older artifacts."""

    # This record/owner deliberately keeps its related fields and lifecycle operations
    # together without artificial wrappers.
    # pylint: disable=too-many-instance-attributes

    def __init__(self, folder, options, checkpoint_format="binary"):
        self.folder = Path(folder)
        self.options = options
        self.suffix = {"binary": ".bin.xz", "compact": ".json.gz", "json": ".json"}[
            checkpoint_format
        ]
        self.shared = checkpoint_format == "binary" and options.get(
            "checkpoint_shared_metadata", False
        )
        self.keep = options.get("checkpoint_keep_last", 2)
        self.checkpoints = []
        self.saved = {}
        self.digests = {}
        self.step_exports = []
        self.owned_models = set()

    def checkpoint(self, experiment, name):
        """Write an experiment checkpoint and prune older owned recovery files."""
        folder = self.folder / "checkpoints"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (name + self.suffix)
        if path.exists():
            raise FileExistsError(f"Refusing to replace existing checkpoint: {path}")
        experiment.write_checkpoint(path, shared_metadata=self.shared)
        self.checkpoints.append(path)
        # Only prune after successful atomic publication.
        while self.keep and len(self.checkpoints) > self.keep:
            self.checkpoints.pop(0).unlink()
        return path

    def step_brains(self, experiment):
        """Save any due strictly improved brain milestones."""

        # Keep this ordered phase and its shared state together; splitting it only to
        # meet a count obscures the contract.
        # pylint: disable=too-many-locals

        interval = self.options.get("brain_save_every_steps", 250)
        if not interval or experiment.world.tick % interval:
            return False
        margin = self.options.get("brain_save_min_score_delta", 0)
        saved = {
            s: {sex: dict(criteria) for sex, criteria in groups.items()}
            for s, groups in self.saved.items()
        }
        changed = {}
        encoding = ModelEncodings()
        for species, groups in experiment.champions.entries.items():
            for sex, criteria in groups.items():
                for criterion, entry in criteria.items():
                    key = species, sex, criterion
                    previous = self.saved.get(species, {}).get(sex, {}).get(criterion)
                    if previous is not None and (
                        tuple(entry["rank"]) <= tuple(previous["rank"])
                        or entry["rank"][0] - previous["rank"][0] < margin
                    ):
                        continue
                    digest = encoding.digest(entry["model"])
                    if self.digests.get(key) == digest:
                        continue
                    saved.setdefault(species, {}).setdefault(sex, {})[criterion] = entry
                    changed[key] = digest
        if not changed:
            return False
        root = self.folder / "champions" / "steps"
        index = (
            root
            / f"step_{experiment.run_index + 1:04d}_{experiment.world.tick:09d}.json"
        )
        if not self.step_exports and (root / "latest.json").exists():
            raise FileExistsError(
                "Refusing to replace an existing periodic brain archive"
            )
        previous_models = set((root / "models").glob("*.bin.xz"))
        generation = export_champions(
            experiment,
            index,
            label=self.folder.name,
            entries=saved,
            update_artifacts=False,
            _encoding=encoding,
        )
        # Stable discovery pointer. Numbered run archives and chaining provenance stay untouched.
        write_json(root / "latest.json", json.loads(index.read_text()))
        self.saved = saved
        self.digests.update(changed)
        self.owned_models.update(
            set((root / "models").glob("*.bin.xz")) - previous_models
        )
        self.step_exports.append(
            (
                index,
                generation,
                {digest.hex() + ".bin.xz" for digest in self.digests.values()},
            )
        )
        keep = self.options.get("brain_keep_last", 2)
        while keep and len(self.step_exports) > keep:
            old_index, old_generation, _ = self.step_exports.pop(0)
            old_index.unlink()
            shutil.rmtree(old_generation)
        retained = set().union(*(models for _, _, models in self.step_exports))
        for path in tuple(self.owned_models):
            if path.name not in retained:
                path.unlink()
                self.owned_models.remove(path)
        return True
