"""Out-of-fold fusion and postprocessing search.

Neural probabilities are produced by ``hatchmatch.predict.predict_probability``
and classical probabilities by ``hatchmatch.baseline.baseline_probability``.
This module only consumes caller-supplied maps for examples held out of the
model that produced them. It reads labels while fitting and scoring. Inference
must not import those label loaders. The search objective is an out-of-fold
document-macro IoU, never an official public-validation score.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import optuna
import yaml
from sklearn.linear_model import LogisticRegression

CHANNEL_NAMES = (
    "neural",
    "classical",
    "foreground",
    "local_variance",
    "query_compatibility",
)
RECALL_FLOOR = 0.95
SEARCH_SCHEMA = "hatchmatch-calibration-search/v1"
OOF_SCHEMA = "hatchmatch-oof-maps/v1"
FOLD_SCHEMA = "hatchmatch-group-folds/v1"
_LABEL_ARRAYS = frozenset(
    {"target", "known", "label", "labels", "positive", "negative", "blank"}
)


class MissingOutOfFoldMaps(FileNotFoundError):
    """Real held-out prediction maps are not available to score."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(
            "real out-of-fold prediction maps are absent: "
            f"{path}. Refusing to fabricate calibration metrics. "
            "Provide maps from hatchmatch.predict.predict_probability and "
            "hatchmatch.baseline.baseline_probability for held-out folds."
        )


class FusionChannels:
    """One query's score maps and the fold that produced the neural map."""

    def __init__(
        self,
        neural: np.ndarray,
        classical: np.ndarray,
        foreground: np.ndarray,
        local_variance: np.ndarray,
        query_compatibility: np.ndarray,
        document_id: str,
        example_id: str,
        source_fold: int,
        trained_example_ids: frozenset[str] | Sequence[str],
    ) -> None:
        arrays = [
            np.asarray(neural, dtype=np.float32),
            np.asarray(classical, dtype=np.float32),
            np.asarray(foreground, dtype=np.float32),
            np.asarray(local_variance, dtype=np.float32),
            np.asarray(query_compatibility, dtype=np.float32),
        ]
        if any(array.ndim != 2 for array in arrays):
            raise ValueError("fusion channels must be two-dimensional")
        shape = arrays[0].shape
        if any(array.shape != shape for array in arrays):
            raise ValueError("fusion channels must share a shape")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("document_id must be non-empty text")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("example_id must be non-empty text")
        if type(source_fold) is not int:
            raise ValueError("source_fold must be an integer")
        trained = frozenset(trained_example_ids)
        if any(type(item) is not str or not item for item in trained):
            raise ValueError("trained_example_ids must contain non-empty text")
        self.neural = arrays[0]
        self.classical = arrays[1]
        self.foreground = arrays[2]
        self.local_variance = arrays[3]
        self.query_compatibility = arrays[4]
        self.document_id = document_id
        self.example_id = example_id
        self.source_fold = source_fold
        self.trained_example_ids = trained

    @property
    def _planes(self) -> tuple[np.ndarray, ...]:
        return tuple(getattr(self, name) for name in CHANNEL_NAMES)


class FusionCalibrator:
    """L2 logistic fusion fit on balanced, document-capped known pixels."""

    def __init__(
        self,
        *,
        max_pixels_per_document: int = 4096,
        seed: int = 0,
    ) -> None:
        if (
            type(max_pixels_per_document) is not int
            or max_pixels_per_document < 2
        ):
            raise ValueError(
                "max_pixels_per_document must be an integer of at least 2"
            )
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.max_pixels_per_document = max_pixels_per_document
        self.seed = seed
        self._coef: np.ndarray | None = None
        self._intercept: float | None = None

    def fit(
        self,
        channels: Sequence[FusionChannels],
        target: Sequence[np.ndarray],
        known: Sequence[np.ndarray],
    ) -> FusionCalibrator:
        """Fit on known pixels from out-of-fold channels and return self."""

        if not channels:
            raise ValueError("channels must not be empty")
        if not (len(channels) == len(target) == len(known)):
            raise ValueError("channels, target, and known must align")
        for channel in channels:
            if not isinstance(channel, FusionChannels):
                raise TypeError("channels must contain FusionChannels")
            if channel.example_id in channel.trained_example_ids:
                raise ValueError(
                    "refusing to fit on in-fold labels for example "
                    f"{channel.example_id!r}"
                )

        features, labels, documents = _known_rows(channels, target, known)
        sampled_x, sampled_y, counts = _balanced_sample(
            features,
            labels,
            documents,
            document_ids=[channel.document_id for channel in channels],
            max_pixels_per_document=self.max_pixels_per_document,
            seed=self.seed,
        )
        model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=self.seed,
        )
        model.fit(sampled_x, sampled_y)
        self.pixels_per_document = counts
        self.sampled_positive_count = int(np.count_nonzero(sampled_y == 1))
        self.sampled_negative_count = int(np.count_nonzero(sampled_y == 0))
        self._coef = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
        self._intercept = float(model.intercept_.reshape(-1)[0])
        return self

    def predict(
        self, channels: Sequence[FusionChannels]
    ) -> list[np.ndarray]:
        """Return fused float32 probability maps."""

        if self._coef is None or self._intercept is None:
            raise RuntimeError("FusionCalibrator must be fit before predict")
        probabilities: list[np.ndarray] = []
        for channel in channels:
            cube = np.stack(channel._planes, axis=-1)
            scores = cube.reshape(-1, len(CHANNEL_NAMES)) @ self._coef
            scores += self._intercept
            probabilities.append(_sigmoid(scores).reshape(cube.shape[:2]))
        return probabilities

    def to_dict(self) -> dict[str, Any]:
        if self._coef is None or self._intercept is None:
            raise RuntimeError("FusionCalibrator must be fit before export")
        return {
            "features": list(CHANNEL_NAMES),
            "coefficients": [float(value) for value in self._coef],
            "intercept": float(self._intercept),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FusionCalibrator:
        features = tuple(payload["features"])
        if features != CHANNEL_NAMES:
            raise ValueError("fusion features do not match the calibrator")
        calibrator = cls()
        calibrator._coef = np.asarray(payload["coefficients"], dtype=np.float64)
        calibrator._intercept = float(payload["intercept"])
        if calibrator._coef.shape != (len(CHANNEL_NAMES),):
            raise ValueError("fusion coefficients must contain five values")
        return calibrator


def postprocess(
    probability: np.ndarray,
    foreground: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    """Threshold, close, and drop components below the minimum area."""

    if not isinstance(probability, np.ndarray) or probability.ndim != 2:
        raise ValueError("probability must be a two-dimensional array")
    if not isinstance(foreground, np.ndarray) or foreground.shape != probability.shape:
        raise ValueError("foreground must match the probability shape")
    threshold = float(config["threshold"])
    min_component = int(config["min_component"])
    close_radius = int(config["close_radius"])
    foreground_floor = float(config.get("foreground_floor", 0.0))
    if min_component < 1:
        raise ValueError("min_component must be at least 1")
    if close_radius < 0:
        raise ValueError("close_radius must be non-negative")

    active = (probability >= threshold) & (foreground >= foreground_floor)
    mask = active.astype(np.uint8)
    if close_radius > 0:
        diameter = 2 * close_radius + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (diameter, diameter)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = ((mask != 0) & (foreground >= foreground_floor)).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    keep = np.zeros(mask.shape, dtype=bool)
    for label in range(1, int(count)):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_component:
            keep[labels == label] = True
    return keep


def document_macro_iou(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    known: Sequence[np.ndarray],
    document_ids: Sequence[str],
) -> float | None:
    """Average nonempty known-pixel query IoUs within each document, then across documents."""

    _require_alignment(predictions, targets, known, document_ids)
    grouped: dict[str, list[float]] = defaultdict(list)
    for prediction, target, known_mask, document_id in zip(
        predictions, targets, known, document_ids, strict=True
    ):
        score = _query_iou(prediction, target, known_mask)
        if score is None:
            continue
        grouped[str(document_id)].append(score)
    if not grouped:
        return None
    means = [
        math.fsum(values) / len(values)
        for _, values in sorted(grouped.items())
    ]
    return math.fsum(means) / len(means)


def pooled_recall(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    known: Sequence[np.ndarray],
) -> float:
    """Return true positives divided by known positive pixels."""

    if not (len(predictions) == len(targets) == len(known)):
        raise ValueError("predictions, targets, and known must align")
    true_positive = 0
    positive = 0
    for prediction, target, known_mask in zip(
        predictions, targets, known, strict=True
    ):
        eligible_target = _binary(target) & _binary(known_mask)
        eligible_prediction = _binary(prediction) & _binary(known_mask)
        positive += int(np.count_nonzero(eligible_target))
        true_positive += int(
            np.count_nonzero(eligible_prediction & eligible_target)
        )
    if positive == 0:
        return 0.0
    return true_positive / positive


def search_objective(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    known: Sequence[np.ndarray],
    document_ids: Sequence[str],
    *,
    recall_floor: float = RECALL_FLOOR,
) -> tuple[float, float]:
    """Return document-macro IoU, or a hard penalty below the recall floor.

    Feasible objectives lie in ``[0, 1]``. Pooled recall below ``recall_floor``
    yields ``recall - 1``, which is strictly below every feasible IoU.
    """

    recall = pooled_recall(predictions, targets, known)
    macro = document_macro_iou(predictions, targets, known, document_ids)
    if macro is None or recall < recall_floor:
        return float(recall) - 1.0, float(recall)
    return float(macro), float(recall)


def search_postprocess(
    channels: Sequence[FusionChannels],
    target: Sequence[np.ndarray],
    known: Sequence[np.ndarray],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Search postprocessing on one fitted out-of-fold calibrator."""

    output_dir = Path(str(config["output_dir"]))
    seed = int(config["seed"])
    n_trials = int(config["n_trials"])
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    recall_floor = float(config["recall_floor"])
    calibrator = FusionCalibrator(
        max_pixels_per_document=int(config["max_pixels_per_document"]),
        seed=seed,
    )
    calibrator.fit(channels, target, known)
    probabilities = calibrator.predict(channels)
    document_ids = [channel.document_id for channel in channels]
    space = config["parameters"]
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "study.db"
    if database_path.exists():
        raise FileExistsError(f"study database already exists: {database_path}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=f"sqlite:///{database_path.resolve()}",
        study_name="hatchmatch-postprocess",
        load_if_exists=False,
    )

    def _objective(trial: optuna.Trial) -> float:
        parameters = _suggest_parameters(trial, space)
        masks = [
            postprocess(probability, channel.foreground, parameters)
            for probability, channel in zip(
                probabilities, channels, strict=True
            )
        ]
        score, recall = search_objective(
            masks,
            target,
            known,
            document_ids,
            recall_floor=recall_floor,
        )
        trial.set_user_attr("recall", float(recall))
        return float(score)

    study.optimize(_objective, n_trials=n_trials)
    best = study.best_trial
    parameters = _json_parameters(best.params)
    masks = [
        postprocess(probability, channel.foreground, parameters)
        for probability, channel in zip(probabilities, channels, strict=True)
    ]
    objective, recall = search_objective(
        masks,
        target,
        known,
        document_ids,
        recall_floor=recall_floor,
    )
    payload = {
        "schema": "hatchmatch-calibration/v1",
        "objective": float(objective),
        "objective_name": "out_of_fold_document_macro_iou",
        "recall": float(recall),
        "recall_floor": recall_floor,
        "study_seed": seed,
        "parameters": parameters,
        "fusion": calibrator.to_dict(),
        "official_public_validation": False,
    }
    trials_payload = {
        "study_seed": seed,
        "trials": [
            {
                "number": int(trial.number),
                "parameters": _json_parameters(trial.params),
                "objective": None if trial.value is None else float(trial.value),
                "recall": float(trial.user_attrs["recall"]),
            }
            for trial in study.trials
        ],
    }
    provenance = {
        "schema": "hatchmatch-calibration-provenance/v1",
        "examples": [
            {
                "example_id": channel.example_id,
                "document_id": channel.document_id,
                "source_fold": channel.source_fold,
                "trained_example_ids": sorted(channel.trained_example_ids),
                "out_of_fold": channel.example_id
                not in channel.trained_example_ids,
            }
            for channel in channels
        ],
    }
    _write_json(output_dir / "trials.json", trials_payload)
    _write_json(output_dir / "fold-provenance.json", provenance)
    _write_json(output_dir / "best.json", payload)
    return json.loads((output_dir / "best.json").read_text(encoding="utf-8"))


def load_out_of_fold_examples(
    config: Mapping[str, Any],
    base_dir: str | Path,
) -> tuple[list[FusionChannels], list[np.ndarray], list[np.ndarray]]:
    """Load held-out maps and labels. Prediction files must not contain labels."""

    base = Path(base_dir)
    manifest_path = (base / str(config["oof_manifest"])).resolve()
    if not manifest_path.is_file():
        raise MissingOutOfFoldMaps(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != OOF_SCHEMA:
        raise ValueError("unsupported out-of-fold map schema")
    entries = manifest.get("examples")
    if not isinstance(entries, list) or not entries:
        raise MissingOutOfFoldMaps(manifest_path)

    fold_path = (base / str(config["fold_manifest"])).resolve()
    fold_payload = json.loads(fold_path.read_text(encoding="utf-8"))
    if (
        not isinstance(fold_payload, dict)
        or fold_payload.get("schema") != FOLD_SCHEMA
    ):
        raise ValueError("unsupported fold manifest schema")
    folds = {
        int(record["fold"]): record
        for record in fold_payload["folds"]
        if isinstance(record, dict)
    }
    selected: list[tuple[dict[str, Any], frozenset[str], int]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("out-of-fold example entries must be objects")
        example_id = entry["id"]
        fold_index = int(entry["fold"])
        record = folds[fold_index]
        train_ids = frozenset(str(item) for item in record["train_ids"])
        valid_ids = frozenset(str(item) for item in record["valid_ids"])
        if example_id in train_ids or example_id not in valid_ids:
            raise ValueError(
                "refusing to fit on in-fold labels for example "
                f"{example_id!r}"
            )
        map_path = (manifest_path.parent / str(entry["path"])).resolve()
        if not map_path.is_file():
            raise MissingOutOfFoldMaps(map_path)
        selected.append((entry, train_ids, fold_index))

    from hatchmatch.data import HatchTileDataset, load_training_examples

    training_manifest = (base / str(config["training_manifest"])).resolve()
    data_root = (base / str(config["data_root"])).resolve()
    training = load_training_examples(training_manifest, data_root)
    by_id = {example.id: example for example in training}
    supervised = []
    for entry, _, _ in selected:
        example = by_id.get(str(entry["id"]))
        if example is None:
            raise ValueError(
                f"out-of-fold example {entry['id']!r} is missing from training labels"
            )
        supervised.append(example)
    dataset = HatchTileDataset(
        supervised,
        tile_size=8,
        query_size=4,
        samples_per_epoch=1,
        augment=False,
        cache_size=0,
    )

    channels: list[FusionChannels] = []
    targets: list[np.ndarray] = []
    knowns: list[np.ndarray] = []
    for index, (entry, train_ids, fold_index) in enumerate(selected):
        example = supervised[index]
        arrays = dataset._load_arrays(index)
        maps = _read_prediction_maps(
            (manifest_path.parent / str(entry["path"])).resolve()
        )
        if maps["neural"].shape != arrays.target.shape:
            raise ValueError(
                f"{example.id}: prediction map shape does not match labels"
            )
        channels.append(
            FusionChannels(
                neural=maps["neural"],
                classical=maps["classical"],
                foreground=maps["foreground"],
                local_variance=maps["local_variance"],
                query_compatibility=maps["query_compatibility"],
                document_id=example.document_id,
                example_id=example.id,
                source_fold=fold_index,
                trained_example_ids=train_ids,
            )
        )
        targets.append(arrays.target)
        knowns.append(arrays.known)
    return channels, targets, knowns


def main(argv: Sequence[str] | None = None) -> int:
    """Search postprocessing, or fail when real out-of-fold maps are absent."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    config_path = arguments.config
    if not config_path.is_file():
        print(f"error: config is missing: {config_path}", file=sys.stderr)
        return 2
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SEARCH_SCHEMA:
        raise ValueError("unsupported calibration search schema")
    base = config_path.parent
    try:
        channels, targets, knowns = load_out_of_fold_examples(config, base)
    except MissingOutOfFoldMaps as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    resolved = dict(config)
    resolved["output_dir"] = str((base / str(config["output_dir"])).resolve())
    search_postprocess(channels, targets, knowns, resolved)
    return 0


def _known_rows(
    channels: Sequence[FusionChannels],
    target: Sequence[np.ndarray],
    known: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    documents: list[np.ndarray] = []
    for channel, target_mask, known_mask in zip(
        channels, target, known, strict=True
    ):
        cube = np.stack(channel._planes, axis=-1)
        known_bool = _binary(known_mask)
        target_bool = _binary(target_mask)
        if known_bool.shape != cube.shape[:2] or target_bool.shape != cube.shape[:2]:
            raise ValueError(
                f"{channel.example_id}: target and known must match channel shape"
            )
        flat_known = known_bool.reshape(-1)
        if not flat_known.any():
            continue
        features.append(cube.reshape(-1, len(CHANNEL_NAMES))[flat_known])
        labels.append(target_bool.reshape(-1)[flat_known].astype(np.int8))
        documents.append(np.full(int(flat_known.sum()), channel.document_id))
    if not features:
        raise ValueError("no known pixels are available for fusion")
    return (
        np.concatenate(features, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(documents, axis=0),
    )


def _balanced_sample(
    features: np.ndarray,
    labels: np.ndarray,
    documents: np.ndarray,
    *,
    document_ids: Sequence[str],
    max_pixels_per_document: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    rng = np.random.default_rng(seed)
    per_class_cap = max_pixels_per_document // 2
    chosen_x: list[np.ndarray] = []
    chosen_y: list[np.ndarray] = []
    counts = {document_id: 0 for document_id in document_ids}
    for document_id in sorted(counts):
        index = np.flatnonzero(documents == document_id)
        if index.size == 0:
            continue
        group_labels = labels[index]
        positive = index[group_labels == 1]
        negative = index[group_labels == 0]
        per_class = min(int(positive.size), int(negative.size), per_class_cap)
        if per_class <= 0:
            continue
        take_positive = rng.choice(positive, size=per_class, replace=False)
        take_negative = rng.choice(negative, size=per_class, replace=False)
        chosen_x.append(features[take_positive])
        chosen_x.append(features[take_negative])
        chosen_y.append(np.ones(per_class, dtype=np.int8))
        chosen_y.append(np.zeros(per_class, dtype=np.int8))
        counts[document_id] = per_class * 2
    if not chosen_x:
        raise ValueError("no balanced known pixels are available for fusion")
    return np.concatenate(chosen_x), np.concatenate(chosen_y), counts


def _binary(array: np.ndarray) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise ValueError("masks must be NumPy arrays")
    if array.ndim != 2:
        raise ValueError("masks must be two-dimensional")
    if array.dtype == np.bool_:
        return array
    return array != 0


def _query_iou(
    prediction: np.ndarray,
    target: np.ndarray,
    known: np.ndarray,
) -> float | None:
    eligible_target = _binary(target) & _binary(known)
    if not bool(eligible_target.any()):
        return None
    eligible_prediction = _binary(prediction) & _binary(known)
    intersection = int(np.count_nonzero(eligible_prediction & eligible_target))
    union = int(np.count_nonzero(eligible_prediction | eligible_target))
    return intersection / union


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, -50.0, 50.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def _suggest_parameters(
    trial: optuna.Trial, space: Mapping[str, Any]
) -> dict[str, float | int]:
    return {
        "threshold": trial.suggest_float(
            "threshold",
            float(space["threshold"]["low"]),
            float(space["threshold"]["high"]),
        ),
        "min_component": trial.suggest_int(
            "min_component",
            int(space["min_component"]["low"]),
            int(space["min_component"]["high"]),
        ),
        "close_radius": trial.suggest_int(
            "close_radius",
            int(space["close_radius"]["low"]),
            int(space["close_radius"]["high"]),
        ),
        "foreground_floor": trial.suggest_float(
            "foreground_floor",
            float(space["foreground_floor"]["low"]),
            float(space["foreground_floor"]["high"]),
        ),
    }


def _json_parameters(params: Mapping[str, Any]) -> dict[str, float | int]:
    return {
        "threshold": float(params["threshold"]),
        "min_component": int(params["min_component"]),
        "close_radius": int(params["close_radius"]),
        "foreground_floor": float(params["foreground_floor"]),
    }


def _read_prediction_maps(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as stored:
        names = set(stored.files)
        if names & _LABEL_ARRAYS:
            raise ValueError(
                "out-of-fold prediction maps must not contain labels"
            )
        missing = [name for name in CHANNEL_NAMES if name not in names]
        if missing:
            rendered = ", ".join(missing)
            raise ValueError(f"out-of-fold map is missing channels: {rendered}")
        return {
            name: np.asarray(stored[name], dtype=np.float32)
            for name in CHANNEL_NAMES
        }


def _require_alignment(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    known: Sequence[np.ndarray],
    document_ids: Sequence[str],
) -> None:
    if not (
        len(predictions) == len(targets) == len(known) == len(document_ids)
    ):
        raise ValueError(
            "predictions, targets, known, and document ids must align"
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
