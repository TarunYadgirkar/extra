"""Label-free classical inference for hatch and tone regions.

The command reads only the challenge input manifest and the drawings. It does
not open annotation masks, expected counts, or private labels.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hatchmatch.contracts import Request, load_requests
from hatchmatch.output import write_binary_png
from hatchmatch.segment import DEFAULT_MAX_SIDE, segment_mask

SCHEMA = "hatchmatch-segment/v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    try:
        metadata = run_predictions(
            args.inputs,
            args.data_root,
            args.output_dir,
            config_path=args.config,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(metadata, sort_keys=True))
    return 0


def run_predictions(
    inputs: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write one native binary mask per request and return timing metadata."""

    started = time.perf_counter()
    settings = _load_settings(config_path)
    requests = load_requests(inputs, data_root)
    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[Path, list[Request]] = defaultdict(list)
    for request in requests:
        grouped[request.image].append(request)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    ) as staging_name:
        staging = Path(staging_name)
        for image_path, group in grouped.items():
            with Image.open(image_path) as image:
                gray = np.asarray(image.convert("L"))
            for request in group:
                if gray.shape != (request.height, request.width):
                    raise ValueError(
                        f"{request.id} image shape {gray.shape} does not match "
                        f"{request.height}x{request.width}"
                    )
                mask = segment_mask(
                    gray,
                    request.query_box,
                    max_side=int(settings["max_side"]),
                    tone_threshold=float(settings["tone_threshold"]),
                    alpha=float(settings["alpha"]),
                )
                write_binary_png(
                    mask,
                    staging / f"{request.id}.png",
                    (request.width, request.height),
                )
            del gray
        for path in sorted(staging.iterdir()):
            destination.mkdir(parents=True, exist_ok=True)
            os.replace(path, destination / path.name)
    elapsed = time.perf_counter() - started
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "elapsed_wall_seconds": elapsed,
        "runtime_seconds": elapsed,
        "time_spent_hours": elapsed / 3600.0,
        "peak_memory_mb": peak_kb / 1024.0,
        "peak_memory_method": "resource.getrusage.ru_maxrss",
        "count": len(requests),
        "settings": settings,
    }


def _load_settings(path: str | Path | None) -> dict[str, Any]:
    settings = {
        "schema": SCHEMA,
        "max_side": DEFAULT_MAX_SIDE,
        "tone_threshold": 0.7,
        "alpha": 0.75,
    }
    if path is None:
        return settings
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or loaded.get("schema") != SCHEMA:
        schema = loaded.get("schema") if isinstance(loaded, dict) else None
        raise ValueError(f"unsupported segment config schema: {schema!r}")
    for key in ("max_side", "tone_threshold", "alpha"):
        if key not in loaded:
            raise ValueError(f"segment config is missing {key}")
    max_side = loaded["max_side"]
    if type(max_side) is not int or max_side < 64:
        raise ValueError("max_side must be an integer of at least 64")
    for key in ("tone_threshold", "alpha"):
        value = loaded[key]
        if type(value) not in (int, float) or not np.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        if not 0.0 < float(value) <= 1.0:
            raise ValueError(f"{key} must be in (0, 1]")
    settings.update(
        {
            "max_side": max_side,
            "tone_threshold": float(loaded["tone_threshold"]),
            "alpha": float(loaded["alpha"]),
        }
    )
    return settings


if __name__ == "__main__":
    raise SystemExit(main())
