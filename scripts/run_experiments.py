"""Minimal experiment runner for the classical texture baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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


def run_baseline(config_path: str | Path) -> dict[str, Any]:
    """Generate baseline predictions and return official evaluator metrics."""

    path = Path(config_path).resolve()
    config = _load_config(path)
    challenge_dir: Path = config["challenge_dir"]
    inputs: Path = config["inputs"]
    evaluation_manifest: Path = config["evaluation_manifest"]
    data_root: Path = config["data_root"]
    run_dir: Path = config["run_dir"]
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
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline_parser = subparsers.add_parser("baseline")
    baseline_parser.add_argument("--config", required=True)
    args = parser.parse_args()

    if args.command == "baseline":
        metrics = run_baseline(args.config)
        print(json.dumps(metrics["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
