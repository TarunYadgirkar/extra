"""Experiment runner for baseline, ablation, and final-ensemble commands.

Completed runs keep the evaluator's ``metrics.json`` bytes unchanged. Ablation
promotion reads caller-supplied real-document scores and does not invent them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image
import yaml

from hatchmatch.baseline import baseline_probability
from hatchmatch.contracts import Box, Request, load_requests
from hatchmatch.output import write_binary_png

CONFIG_KEYS = {
    "challenge_dir",
    "challenge_revision",
    "data_root",
    "evaluation_manifest",
    "inputs",
    "max_dimension",
    "run_dir",
    "threshold",
}
REQUIRED_RUN_FILES = ("config.json", "metrics.json", "provenance.json")
SHARP_DROP = 0.05
DEFAULT_OOF_MANIFEST = "../runs/oof/manifest.json"
ABLATION_SCORE_SCHEMA = "hatchmatch-ablation-scores/v1"
_ABLATION_NAMES = (
    "texture_channels",
    "neural_model",
    "calibration",
    "morphology",
    "backbone_size",
    "fold_count",
    "seeds",
)
_TTA_MODES = ("identity", "horizontal", "vertical", "rot90")


def _config_path(value: Any, base: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"config field {field!r} must be a non-empty path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("baseline config must be a mapping")
    missing = CONFIG_KEYS - config.keys()
    unexpected = config.keys() - CONFIG_KEYS
    if missing or unexpected:
        raise ValueError(
            f"baseline config fields differ: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    threshold = config["threshold"]
    if (
        type(threshold) not in (int, float)
        or not np.isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("config threshold must be a finite number in [0, 1]")
    config["threshold"] = float(threshold)
    max_dimension = config["max_dimension"]
    if max_dimension is not None and (
        type(max_dimension) is not int or max_dimension <= 0
    ):
        raise ValueError(
            "config max_dimension must be null or a positive integer"
        )
    revision = config["challenge_revision"]
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError(
            "config challenge_revision must be 40 lowercase hexadecimal characters"
        )
    for field in CONFIG_KEYS - {
        "challenge_revision",
        "threshold",
        "max_dimension",
    }:
        config[field] = _config_path(config[field], path.parent, field)
    return config


def _predict(
    requests: list[Request],
    predictions: Path,
    threshold: float,
    max_dimension: int | None,
) -> None:
    predictions.mkdir(parents=True, exist_ok=True)
    for stale in predictions.glob("*.png"):
        stale.unlink()

    for request in requests:
        with Image.open(request.image) as image:
            gray = np.asarray(image.convert("L"))

        if (
            max_dimension is not None
            and max(request.width, request.height) > max_dimension
        ):
            nominal_scale = max_dimension / max(request.width, request.height)
            working_width = max(1, int(round(request.width * nominal_scale)))
            working_height = max(1, int(round(request.height * nominal_scale)))
            x_scale = working_width / request.width
            y_scale = working_height / request.height
            working_gray = cv2.resize(
                gray,
                (working_width, working_height),
                interpolation=cv2.INTER_AREA,
            )
            working_box = Box(
                min(
                    working_width - 1,
                    int(np.floor(request.query_box.x0 * x_scale)),
                ),
                min(
                    working_height - 1,
                    int(np.floor(request.query_box.y0 * y_scale)),
                ),
                min(
                    working_width,
                    max(1, int(np.ceil(request.query_box.x1 * x_scale))),
                ),
                min(
                    working_height,
                    max(1, int(np.ceil(request.query_box.y1 * y_scale))),
                ),
            )
            probability = baseline_probability(working_gray, working_box)
            probability = cv2.resize(
                probability,
                (request.width, request.height),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            probability = baseline_probability(gray, request.query_box)
        write_binary_png(
            probability >= threshold,
            predictions / f"{request.id}.png",
            (request.width, request.height),
        )


def _verify_challenge_revision(
    challenge_dir: Path, expected_revision: str
) -> None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(challenge_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"cannot verify challenge revision in {challenge_dir}"
        ) from exc
    actual_revision = completed.stdout.strip()
    if actual_revision != expected_revision:
        raise RuntimeError(
            f"challenge revision mismatch: expected {expected_revision}, "
            f"found {actual_revision}"
        )
    evaluator_path = challenge_dir / "evaluate.py"
    try:
        committed_evaluator = subprocess.run(
            [
                "git",
                "-C",
                str(challenge_dir),
                "show",
                f"{expected_revision}:evaluate.py",
            ],
            check=True,
            capture_output=True,
        ).stdout
        working_evaluator = evaluator_path.read_bytes()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"cannot verify committed evaluator bytes at {evaluator_path}"
        ) from exc
    if working_evaluator != committed_evaluator:
        raise RuntimeError(
            "challenge evaluate.py is modified relative to the pinned revision"
        )


def config_sha256(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a config file's raw bytes."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_run(run_dir: str | Path) -> list[str]:
    """Return missing or edited run files.

    A completed run has ``config.json``, ``metrics.json``, and
    ``provenance.json``, and the provenance hashes still match the first two
    files. An empty result is the only completed state.
    """

    root = Path(run_dir)
    missing = [
        name for name in REQUIRED_RUN_FILES if not (root / name).is_file()
    ]
    if missing:
        return missing
    try:
        provenance = json.loads(
            (root / "provenance.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return ["provenance.json"]
    artifacts = provenance.get("artifacts") if isinstance(provenance, dict) else None
    if not isinstance(artifacts, dict):
        return ["config.json", "metrics.json"]
    problems: list[str] = []
    for name in ("config.json", "metrics.json"):
        record = artifacts.get(name)
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if not isinstance(record, dict) or record.get("sha256") != digest:
            problems.append(name)
    return problems


def refuse_completed_overwrite(run_dir: str | Path) -> None:
    """Raise when a run already has unedited provenance for its metrics."""

    root = Path(run_dir)
    if validate_run(root) == []:
        raise FileExistsError(f"refusing to overwrite completed run: {root}")


def seal_run(
    run_dir: str | Path, config_path: str | Path, *, command: str
) -> dict[str, Any]:
    """Hash an existing metrics file into provenance without rewriting it."""

    root = Path(run_dir)
    source = Path(config_path).resolve()
    refuse_completed_overwrite(root)
    metrics_path = root / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"metrics.json is missing: {metrics_path}. "
            "Refusing to fabricate metrics."
        )
    digest = config_sha256(source)
    _write_json(
        root / "config.json",
        {
            "schema": "hatchmatch-run-config/v1",
            "command": command,
            "source": str(source),
            "sha256": digest,
        },
    )
    provenance = {
        "schema": "hatchmatch-run-provenance/v1",
        "command": command,
        "config_sha256": digest,
        "artifacts": {
            "config.json": _file_record(root / "config.json"),
            "metrics.json": _file_record(metrics_path),
        },
        "official_public_validation": command == "baseline",
    }
    _write_json(root / "provenance.json", provenance)
    return provenance


def ablation_factors() -> list[dict[str, str]]:
    """Name the declared ablations. Scoring them requires caller-supplied IoUs."""

    factors = [
        {
            "factor": name,
            "name": name,
            "scores": "caller-supplied",
        }
        for name in _ABLATION_NAMES
    ]
    factors.extend(
        {
            "factor": "tta",
            "name": f"tta:{mode}",
            "scores": "caller-supplied",
        }
        for mode in _TTA_MODES
    )
    return factors


def promote_ablation(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    sharp_drop: float = SHARP_DROP,
) -> dict[str, Any]:
    """Promote only a strict real-document macro gain without a collapse.

    ``generated_cad`` records are ignored. A document collapses when its mean
    IoU falls to zero from a positive baseline. A sharp drop is a loss greater
    than ``sharp_drop`` on any real document. The returned macros are computed
    only from the supplied scores and are not official public-validation metrics.
    """

    baseline_iou = _real_document_iou(baseline)
    candidate_iou = _real_document_iou(candidate)
    if set(baseline_iou) != set(candidate_iou):
        missing = sorted(set(baseline_iou) - set(candidate_iou))
        extra = sorted(set(candidate_iou) - set(baseline_iou))
        raise ValueError(
            "baseline and candidate real documents differ: "
            f"missing={missing}, extra={extra}"
        )
    collapsed = sorted(
        document_id
        for document_id, score in baseline_iou.items()
        if score > 0.0 and candidate_iou[document_id] == 0.0
    )
    sharp = sorted(
        document_id
        for document_id, score in baseline_iou.items()
        if score - candidate_iou[document_id] > sharp_drop
    )
    baseline_macro = _mean_document_iou(baseline_iou)
    candidate_macro = _mean_document_iou(candidate_iou)
    return {
        "promoted": candidate_macro > baseline_macro and not collapsed and not sharp,
        "baseline_document_macro_iou": baseline_macro,
        "candidate_document_macro_iou": candidate_macro,
        "baseline_document_iou": baseline_iou,
        "candidate_document_iou": candidate_iou,
        "collapsed_documents": collapsed,
        "sharp_drop_documents": sharp,
        "sharp_drop": sharp_drop,
        "official_public_validation": False,
    }


def select_ablations(scores: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the promotion rule to caller-supplied ablation comparisons."""

    if not isinstance(scores, Mapping) or scores.get("schema") != ABLATION_SCORE_SCHEMA:
        raise ValueError("unsupported ablation score schema")
    comparisons = scores.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("ablation comparisons must be a list")
    known = {item["name"] for item in ablation_factors()}
    promoted: list[str] = []
    rejected: list[str] = []
    decisions: dict[str, dict[str, Any]] = {}
    for comparison in comparisons:
        if not isinstance(comparison, Mapping):
            raise ValueError("ablation comparisons must be mappings")
        factor = comparison.get("factor")
        if not isinstance(factor, str) or factor not in known:
            raise ValueError(f"unknown ablation factor: {factor}")
        decision = promote_ablation(
            comparison["baseline"], comparison["candidate"]
        )
        decisions[factor] = decision
        if decision["promoted"]:
            promoted.append(factor)
        else:
            rejected.append(factor)
    return {
        "schema": "hatchmatch-ablation-decisions/v1",
        "promoted": promoted,
        "rejected": rejected,
        "decisions": decisions,
        "official_public_validation": False,
    }


def missing_score_gate_artifacts(config_path: str | Path) -> list[str]:
    """List checkpoint and out-of-fold inputs that are not on disk."""

    path = Path(config_path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("config must be a mapping")
    base = path.parent
    missing: list[str] = []
    checkpoints = loaded.get("checkpoints") or []
    if not isinstance(checkpoints, list) or not checkpoints:
        missing.append("checkpoints: config declares no checkpoints")
    else:
        for item in checkpoints:
            relative = item.get("path") if isinstance(item, dict) else item
            if not isinstance(relative, str) or not relative:
                missing.append("checkpoint: invalid entry")
                continue
            if not (base / relative).is_file():
                missing.append(f"checkpoint: {relative}")
    oof_relative = loaded.get("oof_manifest", DEFAULT_OOF_MANIFEST)
    oof_path = (base / str(oof_relative)).resolve()
    if not oof_path.is_file():
        missing.append(f"out-of-fold maps: {oof_path}")
    return missing


def run_ablate(config_path: str | Path, scores_path: str | Path | None = None) -> int:
    """Score declared ablations from caller-supplied maps, or fail closed."""

    refused = _refuse_missing_score_gate(config_path)
    if refused is not None:
        return refused
    if scores_path is None:
        print(
            "error: caller-supplied real-document out-of-fold scores are "
            "required; refusing to fabricate metrics or launch public validation",
            file=sys.stderr,
        )
        return 2
    payload = json.loads(Path(scores_path).read_text(encoding="utf-8"))
    decision = select_ablations(payload)
    print(
        json.dumps(
            {
                "promoted": decision["promoted"],
                "rejected": decision["rejected"],
                "official_public_validation": False,
            },
            sort_keys=True,
        )
    )
    print(
        "error: ablation decisions are not official metrics; "
        "refusing to write metrics.json",
        file=sys.stderr,
    )
    return 2


def run_final(config_path: str | Path, scores_path: str | Path | None = None) -> int:
    """Select a final ensemble and stop before public validation."""

    refused = _refuse_missing_score_gate(config_path)
    if refused is not None:
        return refused
    if scores_path is not None:
        payload = json.loads(Path(scores_path).read_text(encoding="utf-8"))
        select_ablations(payload)
    print(
        "error: official public validation was not launched; "
        "refusing to fabricate metrics",
        file=sys.stderr,
    )
    return 2


def run_train_folds(config_path: str | Path, *, execute: bool = False) -> int:
    """Validate a fold-training config and refuse to invent training metrics."""

    path = Path(config_path).resolve()
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        print("error: training config must be a mapping", file=sys.stderr)
        return 2
    data = loaded.get("data")
    if not isinstance(data, dict):
        data = {}
    missing: list[str] = []
    for key in ("manifest", "fold_manifest"):
        value = data.get(key)
        if not isinstance(value, str) or not (path.parent / value).is_file():
            missing.append(f"{key}: {value}")
    if missing:
        print(
            "error: refusing to launch fold training or fabricate metrics; "
            "required files are missing:",
            file=sys.stderr,
        )
        for item in missing:
            print(item, file=sys.stderr)
        return 2
    output = loaded.get("output_dir")
    if isinstance(output, str) and output:
        try:
            refuse_completed_overwrite((path.parent / output).resolve())
        except FileExistsError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    if not execute:
        print(
            "error: refusing to launch fold training without --execute; "
            "no metrics.json was written",
            file=sys.stderr,
        )
        return 2
    from hatchmatch.train import main as train_main

    folds = loaded.get("folds")
    seeds = loaded.get("seeds")
    if not isinstance(folds, list) or not isinstance(seeds, list):
        print("error: folds and seeds must be lists", file=sys.stderr)
        return 2
    for fold in folds:
        for seed in seeds:
            status = train_main(
                ["--config", str(path), "--fold", str(fold), "--seed", str(seed)]
            )
            if status != 0:
                return int(status)
    return 0


def run_predict_oof(config_path: str | Path, *, execute: bool = False) -> int:
    """Require checksummed checkpoints before any out-of-fold maps are written."""

    path = Path(config_path).resolve()
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        print("error: prediction config must be a mapping", file=sys.stderr)
        return 2
    missing = _missing_checkpoints(loaded, path.parent)
    if missing:
        print(
            "error: refusing to fabricate metrics; required checkpoints are missing:",
            file=sys.stderr,
        )
        for item in missing:
            print(item, file=sys.stderr)
        return 2
    if not execute:
        print(
            "error: refusing to write out-of-fold maps without --execute; "
            "no metrics.json was written",
            file=sys.stderr,
        )
        return 2
    print(
        "error: out-of-fold prediction was not run; refusing to fabricate maps",
        file=sys.stderr,
    )
    return 2


def run_calibrate(config_path: str | Path) -> int:
    """Delegate to the out-of-fold search, which fails when maps are absent."""

    from hatchmatch.calibrate import main as calibrate_main

    return int(calibrate_main(["--config", str(Path(config_path))]))


def run_baseline(config_path: str | Path) -> dict[str, Any]:
    """Generate baseline predictions and return official evaluator metrics."""

    path = Path(config_path).resolve()
    config = _load_config(path)
    run_dir: Path = config["run_dir"]
    refuse_completed_overwrite(run_dir)
    challenge_dir: Path = config["challenge_dir"]
    inputs: Path = config["inputs"]
    evaluation_manifest: Path = config["evaluation_manifest"]
    data_root: Path = config["data_root"]
    predictions = run_dir / "predictions"
    metrics_path = run_dir / "metrics.json"

    _verify_challenge_revision(challenge_dir, config["challenge_revision"])
    _predict(
        load_requests(inputs, data_root),
        predictions,
        config["threshold"],
        config["max_dimension"],
    )
    metrics_path.unlink(missing_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(challenge_dir / "evaluate.py"),
            "--manifest",
            str(evaluation_manifest),
            "--data-root",
            str(data_root),
            "--predictions",
            str(predictions),
            "--output",
            str(metrics_path),
        ],
        check=True,
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = metrics.get("summary")
    if not isinstance(summary, dict) or "document_macro_iou" not in summary:
        raise RuntimeError("official metrics lack summary.document_macro_iou")
    seal_run(run_dir, path, command="baseline")
    return metrics


def _real_document_iou(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("score records must be mappings")
        kind = record.get("kind", "real")
        if kind == "generated_cad":
            continue
        if kind != "real":
            raise ValueError(f"unsupported score kind: {kind!r}")
        document_id = record.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("document_id must be non-empty text")
        iou = record.get("iou")
        if (
            type(iou) not in (int, float)
            or isinstance(iou, bool)
            or not math.isfinite(float(iou))
            or not 0.0 <= float(iou) <= 1.0
        ):
            raise ValueError("iou must be a finite number in [0, 1]")
        grouped.setdefault(document_id, []).append(float(iou))
    if not grouped:
        raise ValueError("real-document out-of-fold scores are required")
    return {
        document_id: math.fsum(values) / len(values)
        for document_id, values in grouped.items()
    }


def _mean_document_iou(scores: Mapping[str, float]) -> float:
    ordered = [scores[document_id] for document_id in sorted(scores)]
    return math.fsum(ordered) / len(ordered)


def _missing_checkpoints(config: Mapping[str, Any], base: Path) -> list[str]:
    checkpoints = config.get("checkpoints") or []
    if not isinstance(checkpoints, list) or not checkpoints:
        return ["checkpoints: config declares no checkpoints"]
    missing: list[str] = []
    for item in checkpoints:
        relative = item.get("path") if isinstance(item, dict) else item
        if not isinstance(relative, str) or not relative:
            missing.append("checkpoint: invalid entry")
            continue
        if not (base / relative).is_file():
            missing.append(f"checkpoint: {relative}")
    return missing


def _refuse_missing_score_gate(config_path: str | Path) -> int | None:
    missing = missing_score_gate_artifacts(config_path)
    if not missing:
        return None
    print(
        "error: refusing to fabricate metrics; required checkpoints or "
        "out-of-fold maps are missing:",
        file=sys.stderr,
    )
    for item in missing:
        print(item, file=sys.stderr)
    return 2


def _file_record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "baseline",
        "train-folds",
        "predict-oof",
        "calibrate",
        "ablate",
        "final",
    ):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", required=True)
        if name in {"train-folds", "predict-oof"}:
            subparser.add_argument("--execute", action="store_true")
        if name in {"ablate", "final"}:
            subparser.add_argument("--scores", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "baseline":
        metrics = run_baseline(args.config)
        print(json.dumps(metrics["summary"], indent=2, sort_keys=True))
        return 0
    if args.command == "train-folds":
        return run_train_folds(args.config, execute=args.execute)
    if args.command == "predict-oof":
        return run_predict_oof(args.config, execute=args.execute)
    if args.command == "calibrate":
        return run_calibrate(args.config)
    if args.command == "ablate":
        return run_ablate(args.config, args.scores)
    if args.command == "final":
        return run_final(args.config, args.scores)
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
