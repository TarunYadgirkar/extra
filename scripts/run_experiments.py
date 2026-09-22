"""Minimal experiment runner for the classical texture baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import yaml

from hatchmatch.baseline import baseline_probability
from hatchmatch.contracts import Box
from hatchmatch.output import write_binary_png

CONFIG_KEYS = {
    "challenge_dir",
    "manifest",
    "data_root",
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
    for field in CONFIG_KEYS - {"threshold"}:
        config[field] = _config_path(config[field], path.parent, field)
    return config


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest.get("examples") if isinstance(manifest, dict) else manifest
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest must contain a non-empty examples list")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("every manifest example must be an object")
    return rows


def _safe_image_path(data_root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("manifest image must be a non-empty relative path")
    path = (data_root / relative).resolve()
    if not path.is_relative_to(data_root):
        raise ValueError(f"manifest image escapes data_root: {relative!r}")
    return path


def _predict(
    rows: list[dict[str, Any]],
    data_root: Path,
    predictions: Path,
    threshold: float,
) -> None:
    predictions.mkdir(parents=True, exist_ok=True)
    for stale in predictions.glob("*.png"):
        stale.unlink()

    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("manifest example id must be non-empty text")
        width, height = row.get("width"), row.get("height")
        if type(width) is not int or type(height) is not int:
            raise ValueError(f"{identifier}: width and height must be integers")
        raw_box = row.get("query_box")
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            raise ValueError(f"{identifier}: query_box must contain four coordinates")
        query_box = Box(*raw_box)
        if query_box.x1 > width or query_box.y1 > height:
            raise ValueError(f"{identifier}: query_box exceeds native image bounds")

        image_path = _safe_image_path(data_root, row.get("image"))
        with Image.open(image_path) as image:
            gray = np.asarray(image.convert("L"))
        if gray.shape != (height, width):
            raise ValueError(
                f"{identifier}: image shape {gray.shape} does not match "
                f"declared {(height, width)}"
            )

        probability = baseline_probability(gray, query_box)
        write_binary_png(
            probability >= threshold,
            predictions / f"{identifier}.png",
            (width, height),
        )


def run_baseline(config_path: str | Path) -> dict[str, Any]:
    """Generate baseline predictions and return official evaluator metrics."""

    path = Path(config_path).resolve()
    config = _load_config(path)
    challenge_dir: Path = config["challenge_dir"]
    manifest: Path = config["manifest"]
    data_root: Path = config["data_root"]
    run_dir: Path = config["run_dir"]
    predictions = run_dir / "predictions"
    metrics_path = run_dir / "metrics.json"

    _predict(
        _manifest_rows(manifest),
        data_root,
        predictions,
        config["threshold"],
    )
    metrics_path.unlink(missing_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(challenge_dir / "evaluate.py"),
            "--manifest",
            str(manifest),
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
