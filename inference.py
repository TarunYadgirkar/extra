"""Challenge-compatible, label-free inference entry point."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image

from hatchmatch.baseline import baseline_probability
from hatchmatch.calibrate import FusionCalibrator, FusionChannels, postprocess
from hatchmatch.checkpoints import checkpoint_sha256
from hatchmatch.contracts import Request, load_requests
from hatchmatch.features import foreground, local_variance, query_compatibility
from hatchmatch.output import write_binary_png
from hatchmatch.predict import predict_probability

SCHEMA = "hatchmatch-inference/v1"
CALIBRATION_SCHEMA = "hatchmatch-calibration/v1"
MIT_B2_REVISION = "3bb39e8739149c3777d0325349b2a6c32c6413db"
ROOT = Path(__file__).resolve().parent


def main(argv: Sequence[str] | None = None) -> int:
    """Write one native binary mask per label-free request."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        run_inference(
            args.inputs,
            args.data_root,
            args.output_dir,
            config_path=args.config,
            device=args.device,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def run_inference(
    inputs: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Predict every request and return the run metadata."""

    started = time.perf_counter()
    config, resolved_config, config_hash = _load_config(config_path)
    selected_device = _select_device(device, config)
    base = resolved_config.parent if resolved_config is not None else Path.cwd()
    models, checkpoint_hashes = _checkpoint_models(config, base)
    calibrator, parameters = _load_calibration(config, base)
    requests = load_requests(inputs, data_root)
    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    ) as staging_name:
        staging = Path(staging_name)
        for request in requests:
            mask = _request_mask(
                request,
                models,
                config,
                selected_device,
                calibrator,
                parameters,
            )
            write_binary_png(
                mask,
                staging / f"{request.id}.png",
                (request.width, request.height),
            )
        metadata = {
            "elapsed_wall_seconds": time.perf_counter() - started,
            "count": len(requests),
            "hardware": _hardware(selected_device),
            "package_versions": _package_versions(),
            "git_sha": _git_sha(),
            "config_sha256": config_hash,
            "checkpoint_hashes": checkpoint_hashes,
        }
        _write_json(staging / "run-metadata.json", metadata)
        _publish(staging, destination)
    return metadata


def _load_config(
    path: str | Path | None,
) -> tuple[dict[str, Any], Path | None, str]:
    if path is None:
        config = {
            "schema": SCHEMA,
            "mode": "classical",
            "threshold": 0.5,
            "checkpoints": [],
        }
        encoded = json.dumps(config, sort_keys=True).encode("utf-8")
        return config, None, hashlib.sha256(encoded).hexdigest()

    config_path = Path(path)
    raw = config_path.read_bytes()
    loaded = yaml.safe_load(raw)
    if not isinstance(loaded, dict) or loaded.get("schema") != SCHEMA:
        schema = loaded.get("schema") if isinstance(loaded, dict) else None
        raise ValueError(f"unsupported inference config schema: {schema!r}")
    mode = loaded.get("mode")
    if mode not in {"classical", "ensemble"}:
        raise ValueError("inference mode must be 'classical' or 'ensemble'")
    checkpoints = loaded.get("checkpoints", [])
    if checkpoints is None:
        checkpoints = []
    if not isinstance(checkpoints, list):
        raise ValueError("checkpoints must be a list")
    loaded["checkpoints"] = checkpoints
    if mode == "classical" and checkpoints:
        raise ValueError("classical mode cannot declare checkpoints")
    if mode == "ensemble" and not checkpoints:
        raise ValueError("ensemble mode requires checksummed checkpoints")
    if mode == "ensemble":
        model = loaded.get("model")
        if not isinstance(model, dict):
            raise ValueError("ensemble config requires a model mapping")
        if (
            model.get("backbone") == "nvidia/mit-b2"
            and model.get("revision") != MIT_B2_REVISION
        ):
            raise ValueError(
                "nvidia/mit-b2 requires pinned revision "
                f"{MIT_B2_REVISION}"
            )
    return loaded, config_path, hashlib.sha256(raw).hexdigest()


def _select_device(argument: str | None, config: Mapping[str, Any]) -> str:
    selected = argument if argument is not None else config.get("device")
    if selected is None:
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    device = str(selected)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device!r} was requested but CUDA is not available"
        )
    return device


def _resolve(value: str, base: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _checkpoint_models(
    config: Mapping[str, Any],
    base: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    raw_checkpoints = config.get("checkpoints", [])
    if not isinstance(raw_checkpoints, list) or not raw_checkpoints:
        return [], {}
    model = config["model"]
    architecture = {
        "backbone": str(model["backbone"]),
        "decoder_channels": int(model.get("decoder_channels", 128)),
    }
    missing: list[str] = []
    resolved_items: list[tuple[str, Path, str | None]] = []
    for item in raw_checkpoints:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("checkpoint entries must be mappings with a path")
        configured = item["path"]
        resolved = _resolve(configured, base)
        if not resolved.is_file():
            missing.append(configured)
        else:
            configured_digest = item.get("sha256")
            expected = None if configured_digest is None else str(configured_digest)
            resolved_items.append((configured, resolved, expected))
    if missing:
        rendered = "\n".join(missing)
        raise FileNotFoundError(
            "refusing to skip missing checkpoint file(s):\n" + rendered
        )

    models: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for _configured, resolved, expected in resolved_items:
        digest = _verified_digest(resolved, expected)
        hashes[str(resolved)] = digest
        models.append(
            {
                "path": resolved,
                "sha256": digest,
                "architecture": architecture,
            }
        )
    return models, hashes


def _verified_digest(path: Path, expected: str | None) -> str:
    actual = checkpoint_sha256(path)
    if expected is not None:
        digest = expected.lower()
        if digest != actual:
            raise ValueError(
                f"checkpoint SHA-256 mismatch for {path}: "
                f"expected {digest}, computed {actual}"
            )
        return actual
    sidecar = Path(f"{path}.sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"checkpoint SHA-256 sidecar is missing: {sidecar}")
    fields = sidecar.read_text(encoding="ascii").split()
    if not fields or fields[0].lower() != actual:
        found = fields[0].lower() if fields else ""
        raise ValueError(
            f"checkpoint SHA-256 mismatch for {path}: "
            f"sidecar {found}, computed {actual}"
        )
    return actual


def _load_calibration(
    config: Mapping[str, Any],
    base: Path,
) -> tuple[FusionCalibrator | None, Mapping[str, Any] | None]:
    configured = config.get("calibration")
    if not configured:
        return None, None
    path = _resolve(str(configured), base)
    if not path.is_file():
        raise FileNotFoundError(f"calibration file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("unsupported calibration schema")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("calibration parameters must be a mapping")
    return FusionCalibrator.from_dict(payload["fusion"]), parameters


def _request_mask(
    request: Request,
    models: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    device: str,
    calibrator: FusionCalibrator | None,
    parameters: Mapping[str, Any] | None,
) -> np.ndarray:
    gray = _load_gray(request)
    classical = baseline_probability(gray, request.query_box)
    neural = None
    if models:
        neural = predict_probability(
            request,
            models,
            _predict_config(config, device),
        )
        if neural.shape != gray.shape:
            raise ValueError(
                f"request {request.id} neural map shape {neural.shape} does not "
                f"match {(request.height, request.width)}"
            )
    if calibrator is not None:
        if parameters is None:
            raise ValueError("calibration parameters are required")
        foreground_map = foreground(gray)
        variance_map = local_variance(gray)
        compatibility = query_compatibility(gray, request.query_box)
        probability = calibrator.predict(
            [
                FusionChannels(
                    neural=(
                        neural
                        if neural is not None
                        else np.zeros_like(classical)
                    ),
                    classical=classical,
                    foreground=foreground_map,
                    local_variance=variance_map,
                    query_compatibility=compatibility,
                    document_id=request.id,
                    example_id=request.id,
                    source_fold=-1,
                    trained_example_ids=frozenset(),
                    kind="real",
                )
            ]
        )[0]
        return postprocess(probability, foreground_map, parameters)
    if neural is not None:
        return neural >= _threshold(config)
    return classical >= _threshold(config)


def _predict_config(config: Mapping[str, Any], device: str) -> dict[str, Any]:
    tile_size = int(config.get("tile_size", 256))
    overlap = (
        int(config["overlap"]) if "overlap" in config else tile_size // 4
    )
    transforms = config.get("tta", ["identity"])
    if not isinstance(transforms, list) or not transforms:
        raise ValueError("tta must be a non-empty list")
    return {
        "tile_size": tile_size,
        "overlap": overlap,
        "query_size": int(config.get("query_size", 96)),
        "batch_size": int(config.get("batch_size", 1)),
        "tta": tuple(str(name) for name in transforms),
        "device": device,
    }


def _load_gray(request: Request) -> np.ndarray:
    with Image.open(request.image) as image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if gray.shape != (request.height, request.width):
        raise ValueError(
            f"image size {gray.shape[::-1]} does not match declared size "
            f"{(request.width, request.height)}"
        )
    return gray


def _publish(staging: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(staging.iterdir()):
        os.replace(path, destination / path.name)


def _threshold(config: Mapping[str, Any]) -> float:
    value = config.get("threshold", 0.5)
    if type(value) not in (int, float) or not np.isfinite(value):
        raise ValueError("threshold must be a finite number")
    return float(value)


def _hardware(device: str) -> dict[str, Any]:
    cpu_count = os.cpu_count()
    return {
        "platform": platform.platform(),
        "cpu_count": 0 if cpu_count is None else cpu_count,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": device,
    }


def _package_versions() -> dict[str, str]:
    names = (
        "albumentations",
        "numpy",
        "opencv-python-headless",
        "optuna",
        "pillow",
        "pyyaml",
        "scikit-image",
        "scikit-learn",
        "torch",
        "transformers",
    )
    return {name: importlib.metadata.version(name) for name in names}


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
