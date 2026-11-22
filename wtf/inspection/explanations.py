"""Bounded on-demand archived champion analysis for the playback server."""

from collections import OrderedDict
import json
import math
from pathlib import Path
from threading import RLock

from wtf.simulation.tracking import scent_contract
from wtf.inspection.explainability import MAX_PARAMETERS, explain_model, prepare_model
from wtf.storage.model_io import model_path, read_model
from wtf.storage.replay_io import chunk_path, load_chunk

CRITERIA = ("health", "kills", "lifespan", "food", "balanced", "lineage")


def _inside(root, path):
    result = Path(path).resolve()
    if root not in result.parents:
        raise ValueError("Explanation artifact outside recording folder")
    return result


def _stamp(path):
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class ExplanationStore:
    """Two packed models / one million parameters total; two tiny frame contexts.

    Cache keys include recording/run/frame/animal and artifact/blob/chunk stat
    identities. Never retain a decoded replay chunk or cache mutable inference.
    """

    def __init__(self, recordings):
        self.recordings = recordings
        self.models = OrderedDict()
        self.contexts = OrderedDict()
        self.lock = RLock()

    def _context(self, recording_id, run, frame, animal):
        manifest = self.recordings.metadata(recording_id)
        entry = next((r for r in manifest["runs"] if r["run"] == run), None)
        if entry is None or not 0 <= frame < entry["frame_count"]:
            raise ValueError("Explanation frame outside recorded run")
        size = manifest["recording"]["frames_per_chunk"]
        index = frame // size
        if not 0 <= index < entry["chunk_count"]:
            raise ValueError("Explanation frame has no committed chunk")
        chunk = chunk_path(self.recordings._folder(recording_id), run, index)
        key = (recording_id, run, frame, animal, _stamp(chunk))
        with self.lock:
            if key in self.contexts:
                self.contexts.move_to_end(key)
                return manifest, self.contexts[key]
        frames = load_chunk(chunk)
        local = frame % size
        body = next((a for a in frames[local]["animals"] if a["id"] == animal), None)
        if body is None:
            raise ValueError("Selected animal is absent at this frame")
        diagnostic = None
        for offset in range(local, -1, -1):
            previous = frames[offset]
            if "diagnostics" in previous:
                diagnostic = previous["diagnostics"].get(str(animal))
                break
        if diagnostic is None or not diagnostic.get("features"):
            raise ValueError(
                "No recorded observation for this animal; enable recording diagnostics"
            )
        context = dict(
            body={k: body.get(k) for k in ("id", "species", "sex")},
            diagnostic=diagnostic,
            tick=frames[local]["tick"],
        )
        del frames
        with self.lock:
            self.contexts[key] = context
            while len(self.contexts) > 2:
                self.contexts.popitem(last=False)
        return manifest, context

    def _model(self, artifact_path, artifact):
        reference = artifact["model"]
        blob = (
            model_path(artifact_path, reference)
            if "weights_file" in reference
            else None
        )
        key = (
            (_stamp(blob), json.dumps(reference, sort_keys=True))
            if blob
            else (_stamp(artifact_path),)
        )
        with self.lock:
            if key in self.models:
                self.models.move_to_end(key)
                return self.models[key]
            # Serialize cache misses to bound decompression/preparation peak RAM.
            model = read_model(artifact_path, reference) if blob else reference
            prepared = prepare_model(model)
            while self.models and (
                len(self.models) >= 2
                or sum(m["parameters"] for m in self.models.values())
                + prepared["parameters"]
                > MAX_PARAMETERS
            ):
                self.models.popitem(last=False)
            self.models[key] = prepared
            return prepared

    def explain(
        self,
        recording_id,
        run,
        frame,
        animal,
        criterion=None,
        action=None,
        feature=None,
        value=None,
    ):
        if any(type(v) is not int or v < 0 for v in (run, frame, animal)) or run < 1:
            raise ValueError("Invalid explanation run/frame/animal")
        if criterion is not None and criterion not in CRITERIA:
            raise ValueError("Unknown champion criterion")
        manifest, context = self._context(recording_id, run, frame, animal)
        body, diagnostic = context["body"], context["diagnostic"]
        species, sex = body["species"], body["sex"]
        if sex not in ("male", "female"):
            raise ValueError(
                "Animal sex was not recorded; a matching archive cannot be selected"
            )
        root = self.recordings._folder(recording_id).parent
        index_path = _inside(root, root / "champions" / f"run_{run:04d}.json")
        if not index_path.is_file():
            raise ValueError(
                "No saved champions for this run; record with brain saving enabled"
            )
        index = json.loads(index_path.read_text())
        entries = [
            e
            for e in index.get("entries", [])
            if e.get("species") == species and e.get("sex") == sex
        ]
        available = [
            c for c in CRITERIA if any(e.get("criterion") == c for e in entries)
        ]
        spec = manifest["config"]["species"][species]
        requested = (
            criterion
            or spec.get("champion_selection_by_sex", {}).get(sex)
            or spec.get("champion_selection", {}).get("criterion")
            or ("lifespan" if spec["diet"] == "resource" else "kills")
        )
        if not criterion and requested not in available and available:
            requested = available[0]
        entry = next((e for e in entries if e.get("criterion") == requested), None)
        if entry is None:
            raise ValueError("No matching species/sex champion for this criterion")
        # Prefer the portable run-local directory, never an arbitrary legacy absolute root.
        directory = _inside(
            root, index_path.parent / index.get("model_directory", index_path.stem)
        )
        path = _inside(directory, directory / entry["file"])
        artifact = json.loads(path.read_text())
        if (
            artifact.get("source", {}).get("species") != species
            or artifact.get("source", {}).get("sex") != sex
        ):
            raise ValueError("Archived champion species/sex mismatch")
        if artifact.get("criterion") != requested:
            raise ValueError("Archived champion criterion mismatch")
        model = self._model(path, artifact)
        sig = model["signature"]
        template = manifest["config"]["brains"][spec["brain"]]
        if (
            sig["algorithm"] != template["type"]
            or sig.get("observation_version", 1)
            != template.get("observation_version", 1)
            or sig.get("observation_top_k", 1) != template.get("observation_top_k", 1)
            or sig.get("movement_history_steps", 0)
            != template.get("movement_history_steps", 0)
            or scent_contract(sig) != scent_contract(template)
            or sig["actions"] != spec["allowed_actions"]
        ):
            raise ValueError(
                "Archived model and recorded observation/action contract differ"
            )
        if sig["algorithm"] == "maskable_ppo" and (
            not isinstance(diagnostic.get("action_mask"), dict)
            or set(diagnostic["action_mask"]) != set(sig["actions"])
            or any(type(v) is not bool for v in diagnostic["action_mask"].values())
        ):
            raise ValueError("Recorded action mask is missing or incompatible")
        features = list(diagnostic["features"])
        intervention = None
        if feature is not None or value is not None:
            if (
                type(feature) is not int
                or not 0 <= feature < len(features)
                or type(value) not in (float, int)
                or not math.isfinite(value)
                or not -1 <= value <= 1
            ):
                raise ValueError(
                    "What-if needs a valid feature index and a finite value between -1 and 1"
                )
            intervention = dict(index=feature, original=features[feature], value=value)
            features[feature] = value
        result = explain_model(model, features, action, diagnostic.get("action_mask"))
        result.update(
            species=species,
            sex=sex,
            animal=animal,
            run=run,
            frame=frame,
            tick=context["tick"],
            diagnostic_tick=diagnostic["tick"],
            criterion=requested,
            available_criteria=available,
            source=artifact.get("source", entry.get("source", {})),
            model_generation=artifact.get("model_generation"),
            recorded_predictions=diagnostic.get(
                "action_probabilities", diagnostic.get("action_values", {})
            ),
            recorded_policy="action_probabilities" in diagnostic,
            intervention=intervention,
        )
        return result
