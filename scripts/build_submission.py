"""Build a checksummed challenge submission without inventing scores.

A run is usable only when ``config.json`` and ``metrics.json`` still match
``provenance.json``. The measured document-macro IoU is copied from that
unedited metrics file. ``--run`` and ``--output`` are enough for a sealed run:
repository identity comes from git, dataset and evaluator identity come from
the pinned challenge checkout, and hardware comes from this machine. Timing
comes from the run when it was recorded. Unmeasured timing stays null.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from scripts.run_experiments import validate_run

HISTORICAL_REFERENCE = (
    "Historical reference: 73.73% document-macro IoU on a different "
    "37-query suite. That figure is not the score of this bundle."
)
ENSEMBLE_STATUS = "The neural ensemble has not been trained or scored here."
FIXTURE_EXAMPLES = (
    "The three visual examples are representative fixtures until real "
    "validation predictions exist."
)
NO_PRIVATE_TEST = "This bundle does not display private-test material."
TIMING_DEFINITION = (
    "Wall clock covering loading, preprocessing, inference, file writing, "
    "and initialization."
)
UNMEASURED_TIMING = (
    "Timing was not measured. The recorded nulls are not measurements."
)
DEFAULT_INFERENCE_COMMAND = (
    "python inference.py --inputs validation-inputs.json "
    "--data-root dataset --output-dir predictions --config configs/final.yaml"
)
ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_FIELDS = (
    "schema_version",
    "repository",
    "commit",
    "inference_command",
    "environment",
    "weights",
    "validation_metrics",
    "hardware",
    "runtime_seconds",
    "peak_memory_mb",
    "time_spent_hours",
    "external_resources",
    "ai_tools",
)


class SubmissionBuildError(RuntimeError):
    """The requested run cannot produce an honest submission bundle."""


def artifact_record(path: str | Path) -> dict[str, Any]:
    """Return the byte length and SHA-256 hex digest of a file."""

    data = Path(path).read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def normalize_repository_url(url: str) -> str:
    """Return an HTTPS repository URL without credentials or a ``.git`` suffix."""

    parts = urlsplit(url)
    host = parts.hostname
    if parts.scheme != "https" or not host:
        raise SubmissionBuildError("repository must be a valid HTTPS URL")
    path = parts.path[:-4] if parts.path.endswith(".git") else parts.path
    if path.endswith("/"):
        path = path[:-1]
    return urlunsplit(("https", host, path, "", ""))


def _repository_url() -> str:
    return normalize_repository_url(_git_output("remote", "get-url", "origin", cwd=ROOT))


def _git_head(cwd: Path) -> str:
    return _git_output("rev-parse", "HEAD", cwd=cwd)


def _git_output(*args: str, cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SubmissionBuildError(f"git {' '.join(args)} failed in {cwd}") from exc


def _pinned_dataset() -> dict[str, Any]:
    path = ROOT / "challenge" / "data" / "release.json"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SubmissionBuildError(f"pinned dataset release is unreadable: {path}") from exc
    if not isinstance(loaded, dict):
        raise SubmissionBuildError("pinned dataset release must be a JSON object")
    return loaded


def _machine_hardware() -> str:
    cpu_count = os.cpu_count() or 0
    cuda = False
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except ImportError:
        cuda = False
    return f"{platform.platform()}; cpus={cpu_count}; cuda={cuda}"


def _official_public_validation(run: Path) -> bool:
    try:
        provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SubmissionBuildError(
            "refusing to invent an official public-validation score; "
            "provenance.json is unreadable"
        ) from exc
    return isinstance(provenance, dict) and provenance.get("official_public_validation") is True


def _resolved_timing(
    run: Path,
    *,
    runtime_seconds: float | None,
    time_spent_hours: float | None,
    peak_memory_mb: float | None,
) -> tuple[float | None, float | None, float | None]:
    found = _timing_from_run(run)
    runtime = runtime_seconds if runtime_seconds is not None else found.get("runtime_seconds")
    spent = (
        time_spent_hours if time_spent_hours is not None else found.get("time_spent_hours")
    )
    if peak_memory_mb is not None:
        peak = peak_memory_mb
    elif "peak_memory_mb" in found:
        peak = found["peak_memory_mb"]
    else:
        peak = None
    return runtime, spent, peak


def _timing_from_run(run: Path) -> dict[str, Any]:
    found: dict[str, Any] = {}

    def absorb(payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        if "runtime_seconds" in payload and "runtime_seconds" not in found:
            found["runtime_seconds"] = payload["runtime_seconds"]
        elif "elapsed_wall_seconds" in payload and "runtime_seconds" not in found:
            found["runtime_seconds"] = payload["elapsed_wall_seconds"]
        if "time_spent_hours" in payload and "time_spent_hours" not in found:
            found["time_spent_hours"] = payload["time_spent_hours"]
        if "peak_memory_mb" in payload and "peak_memory_mb" not in found:
            found["peak_memory_mb"] = payload["peak_memory_mb"]

    timing_path = run / "timing.json"
    if timing_path.is_file():
        absorb(_read_json_object(timing_path))
    provenance = _read_json_object(run / "provenance.json")
    absorb(provenance.get("timing") if isinstance(provenance, dict) else None)
    metadata_path = run / "run-metadata.json"
    if metadata_path.is_file():
        absorb(_read_json_object(metadata_path))
    return found


def _read_json_object(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SubmissionBuildError(f"unreadable JSON: {path}") from exc


def _representative_fixtures() -> list[dict[str, Any]]:
    """Synthetic panels used when a run has no reviewed prediction overlay."""

    examples: list[dict[str, Any]] = []
    for index, fill in enumerate((40, 50, 60), start=1):
        drawing = np.full((4, 4), fill, dtype=np.uint8)
        positive = np.zeros((4, 4), dtype=bool)
        negative = np.zeros((4, 4), dtype=bool)
        prediction = np.zeros((4, 4), dtype=bool)
        positive[0, 0] = True
        negative[0, 1] = True
        prediction[0, 0] = True
        prediction[0, 2] = True
        examples.append(
            {
                "id": f"fixture-drawing-{index}",
                "split": "fixture",
                "drawing": drawing,
                "query_box": (3, 0, 4, 1),
                "prediction": prediction,
                "positive": positive,
                "negative": negative,
            }
        )
    return examples


def verify_manifest(output_dir: str | Path) -> list[str]:
    """Return artifact names whose bytes no longer match ``manifest.json``.

    Every file in the bundle except the manifest itself must be listed.
    """

    root = Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return ["manifest.json"]
    problems: list[str] = []
    listed: set[str] = set()
    for name, record in artifacts.items():
        if not isinstance(record, dict) or "path" not in record:
            problems.append(str(name))
            continue
        relative = str(record["path"])
        listed.add(relative)
        path = root / relative
        if not path.is_file():
            problems.append(str(name))
            continue
        actual = artifact_record(path)
        if actual["sha256"] != record.get("sha256") or actual["bytes"] != record.get(
            "bytes"
        ):
            problems.append(str(name))
    on_disk = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    problems.extend(sorted(on_disk - listed))
    return problems


def build_submission(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    repository: str | None = None,
    commit: str | None = None,
    inference_command: str | None = None,
    environment_lock: str | Path | None = None,
    hardware: str | None = None,
    runtime_seconds: float | None = None,
    time_spent_hours: float | None = None,
    peak_memory_mb: float | None = None,
    peak_memory_method: str | None = None,
    external_resources: Sequence[str] | None = None,
    ai_tools: Sequence[str] | None = None,
    evaluator_revision: str | None = None,
    dataset: Mapping[str, Any] | None = None,
    examples: Sequence[Mapping[str, Any]] | None = None,
    checkpoints: Sequence[Mapping[str, Any]] | None = None,
    timing_definition: str | None = None,
) -> Path:
    """Write ``submission.json``, checksums, fixtures, and a report.

    The destination is created only after the run's metrics are still the
    sealed bytes and the caller has supplied every disclosure the bundle needs.
    """

    run = Path(run_dir)
    output = Path(output_dir)
    problems = validate_run(run)
    if problems:
        raise SubmissionBuildError(
            "refusing to build a submission: run lacks unedited metrics: "
            + ", ".join(problems)
        )
    metrics_bytes = (run / "metrics.json").read_bytes()
    config_bytes = (run / "config.json").read_bytes()
    iou = _measured_iou(metrics_bytes)
    official = _official_public_validation(run)
    rendered = _prepare_examples(examples)
    predictions = _prediction_files(run / "predictions")
    runtime_seconds, time_spent_hours, peak_memory_mb = _resolved_timing(
        run,
        runtime_seconds=runtime_seconds,
        time_spent_hours=time_spent_hours,
        peak_memory_mb=peak_memory_mb,
    )
    identity = _identity(
        repository=repository if repository is not None else _repository_url(),
        commit=commit if commit is not None else _git_head(ROOT),
        inference_command=inference_command or DEFAULT_INFERENCE_COMMAND,
        environment_lock=(
            environment_lock if environment_lock is not None else ROOT / "pyproject.toml"
        ),
        hardware=hardware if hardware is not None else _machine_hardware(),
        runtime_seconds=runtime_seconds,
        time_spent_hours=time_spent_hours,
        peak_memory_mb=peak_memory_mb,
        evaluator_revision=(
            evaluator_revision
            if evaluator_revision is not None
            else _git_head(ROOT / "challenge")
        ),
        dataset=dataset if dataset is not None else _pinned_dataset(),
        external_resources=external_resources,
        ai_tools=ai_tools,
        official_public_validation=official,
    )
    if output.exists():
        raise SubmissionBuildError(f"refusing to overwrite submission output: {output}")

    definition = timing_definition or TIMING_DEFINITION
    output.mkdir(parents=True)
    try:
        records: dict[str, dict[str, Any]] = {}
        _store_bytes(output, records, "metrics.json", metrics_bytes)
        _store_bytes(output, records, "config.json", config_bytes)
        _store_bytes(
            output,
            records,
            "environment.lock",
            Path(identity["environment_lock"]).read_bytes(),
        )
        _archive_predictions(predictions, output / "predictions.tar")
        records["predictions.tar"] = {
            **artifact_record(output / "predictions.tar"),
            "path": "predictions.tar",
        }
        for checkpoint in checkpoints or ():
            _store_checkpoint(output, records, identity["weights"], checkpoint)
        for example_id, image in rendered:
            relative = f"examples/{example_id}.png"
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination, format="PNG")
            records[relative] = {**artifact_record(destination), "path": relative}
        report = _bundle_report(
            iou,
            official=official,
            timing_unmeasured=runtime_seconds is None and time_spent_hours is None,
        )
        _store_bytes(output, records, "technical-report.md", report.encode("utf-8"))
        submission = _submission(identity)
        _store_bytes(
            output,
            records,
            "submission.json",
            _json_bytes(submission),
        )
        manifest = {
            "schema": "hatchmatch-submission-manifest/v1",
            "artifacts": records,
            "dataset": identity["dataset"],
            "evaluator_revision": identity["evaluator_revision"],
            "timing": {
                "definition": definition,
                "peak_memory_mb": peak_memory_mb,
                "peak_memory_method": peak_memory_method,
                "runtime_seconds": runtime_seconds,
                "time_spent_hours": time_spent_hours,
            },
            "historical_reference": {
                "document_macro_iou_percent": "73.73%",
                "queries": 37,
                "suite": "different previously exposed development suite",
            },
            "metrics_document_macro_iou": iou,
            "official_public_validation_document_macro_iou": iou if official else None,
        }
        (output / "manifest.json").write_bytes(_json_bytes(manifest))
        mismatches = verify_manifest(output)
        if mismatches:
            raise SubmissionBuildError(
                "submission manifest does not match artifact bytes: "
                + ", ".join(mismatches)
            )
        if runtime_seconds is not None and time_spent_hours is not None:
            _validate_with_official_schema(output / "submission.json")
    except Exception:
        if output.exists():
            _remove_tree(output)
        raise
    return output


def main(argv: Sequence[str] | None = None) -> int:
    """Build a bundle, or exit 2 when the run lacks unedited metrics."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        build_submission(args.run, args.output)
    except SubmissionBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _measured_iou(metrics_bytes: bytes) -> float:
    try:
        metrics = json.loads(metrics_bytes)
    except json.JSONDecodeError as exc:
        raise SubmissionBuildError(
            "refusing to invent document_macro_iou; metrics.json is not JSON"
        ) from exc
    summary = metrics.get("summary") if isinstance(metrics, dict) else None
    if not isinstance(summary, dict) or "document_macro_iou" not in summary:
        raise SubmissionBuildError(
            "refusing to invent document_macro_iou; unedited metrics lack "
            "summary.document_macro_iou"
        )
    value = summary["document_macro_iou"]
    if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(
        value
    ):
        raise SubmissionBuildError(
            "refusing to invent document_macro_iou; summary.document_macro_iou "
            "is not a finite number"
        )
    return float(value)


def _prepare_examples(
    examples: Sequence[Mapping[str, Any]] | None,
) -> list[tuple[str, Image.Image]]:
    if examples is None:
        examples = _representative_fixtures()
    if len(examples) != 3:
        raise SubmissionBuildError(
            "refusing to invent visual examples; three representative fixtures "
            "are required"
        )
    rendered: list[tuple[str, Image.Image]] = []
    for example in examples:
        split = str(example.get("split", ""))
        example_id = str(example.get("id", ""))
        if "private" in split.lower() or "private" in example_id.lower():
            raise SubmissionBuildError(
                "refusing to display private-test material in visual examples"
            )
        if not re.fullmatch(r"[A-Za-z0-9._-]+", example_id):
            raise SubmissionBuildError(
                f"example id is not a safe file name: {example_id!r}"
            )
        rendered.append((example_id, _render_example(example)))
    return rendered


def _render_example(example: Mapping[str, Any]) -> Image.Image:
    drawing = np.asarray(example["drawing"], dtype=np.uint8)
    if drawing.ndim != 2:
        raise SubmissionBuildError("fixture drawing must be a grayscale array")
    height, width = drawing.shape
    positive = _mask(example["positive"], height, width, "positive")
    negative = _mask(example["negative"], height, width, "negative")
    prediction = _mask(example["prediction"], height, width, "prediction")
    try:
        x0, y0, x1, y1 = (int(value) for value in example["query_box"])
    except (TypeError, ValueError) as exc:
        raise SubmissionBuildError("query_box must contain four integers") from exc
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise SubmissionBuildError("query_box is outside the drawing")
    query = np.zeros((height, width), dtype=bool)
    query[y0:y1, x0:x1] = True

    gray = np.stack((drawing, drawing, drawing), axis=-1)
    query_panel = gray.copy()
    query_panel[query] = (255, 0, 255)
    prediction_panel = np.zeros((height, width, 3), dtype=np.uint8)
    prediction_panel[prediction] = (0, 255, 0)
    labels = np.empty((height, width, 3), dtype=np.uint8)
    labels[:] = (128, 128, 128)
    labels[query] = (255, 255, 0)
    labels[negative] = (0, 0, 255)
    labels[positive] = (255, 0, 0)
    composite = np.concatenate((gray, query_panel, prediction_panel, labels), axis=1)
    return Image.fromarray(composite, mode="RGB")


def _mask(value: Any, height: int, width: int, name: str) -> np.ndarray:
    mask = np.asarray(value, dtype=bool)
    if mask.shape != (height, width):
        raise SubmissionBuildError(f"{name} mask does not match the drawing")
    return mask


def _prediction_files(source: Path) -> list[Path]:
    if not source.is_dir():
        raise SubmissionBuildError(
            "refusing to build a submission: predictions are missing"
        )
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise SubmissionBuildError(
            "refusing to build a submission: predictions are missing"
        )
    return files


def _identity(
    *,
    repository: str | None,
    commit: str | None,
    inference_command: str | None,
    environment_lock: str | Path | None,
    hardware: str | None,
    runtime_seconds: float | None,
    time_spent_hours: float | None,
    peak_memory_mb: float | None,
    evaluator_revision: str | None,
    dataset: Mapping[str, Any] | None,
    external_resources: Sequence[str] | None,
    ai_tools: Sequence[str] | None,
    official_public_validation: bool,
) -> dict[str, Any]:
    if repository is None or commit is None or inference_command is None:
        raise SubmissionBuildError(
            "repository, commit, and inference_command are required"
        )
    if environment_lock is None or hardware is None or evaluator_revision is None:
        raise SubmissionBuildError(
            "environment lock, hardware, and evaluator revision are required"
        )
    if dataset is None:
        raise SubmissionBuildError("dataset revision is required")
    _https_url(repository, "repository")
    if urlsplit(repository).username or urlsplit(repository).password:
        raise SubmissionBuildError("repository URL must not contain credentials")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise SubmissionBuildError(
            "commit must contain exactly 40 hexadecimal characters"
        )
    command = _require_command(inference_command)
    if not isinstance(hardware, str) or not hardware.strip():
        raise SubmissionBuildError("hardware must be a nonempty string")
    if runtime_seconds is not None:
        _nonnegative(runtime_seconds, "runtime_seconds")
    if time_spent_hours is not None:
        _nonnegative(time_spent_hours, "time_spent_hours")
    if peak_memory_mb is not None:
        _nonnegative(peak_memory_mb, "peak_memory_mb")
    lock = Path(environment_lock)
    if not lock.is_file():
        raise SubmissionBuildError(f"environment lock is missing: {lock}")
    if not isinstance(evaluator_revision, str) or not evaluator_revision.strip():
        raise SubmissionBuildError("evaluator revision must be a nonempty string")
    dataset_record = _dataset_record(dataset)
    resources = _text_list(external_resources, "external_resources")
    tools = _text_list(ai_tools, "ai_tools")
    resources.extend(
        [
            HISTORICAL_REFERENCE,
            (
                f"Dataset release {dataset_record['version']} "
                f"{dataset_record['url']} sha256 {dataset_record['sha256']}"
            ),
            f"Evaluator revision {evaluator_revision}",
        ]
    )
    if not official_public_validation:
        resources.append(ENSEMBLE_STATUS)
    if runtime_seconds is None and time_spent_hours is None:
        resources.append(UNMEASURED_TIMING)
    return {
        "repository": repository,
        "commit": commit,
        "inference_command": command,
        "environment_lock": lock,
        "hardware": hardware,
        "runtime_seconds": runtime_seconds,
        "time_spent_hours": time_spent_hours,
        "peak_memory_mb": peak_memory_mb,
        "evaluator_revision": evaluator_revision,
        "dataset": dataset_record,
        "external_resources": resources,
        "ai_tools": tools,
        "weights": [],
    }


def _submission(identity: Mapping[str, Any]) -> dict[str, Any]:
    submission = {
        "schema_version": "1.0",
        "repository": identity["repository"],
        "commit": identity["commit"],
        "inference_command": identity["inference_command"],
        "environment": "environment.lock",
        "weights": identity["weights"],
        "validation_metrics": "metrics.json",
        "hardware": identity["hardware"],
        "runtime_seconds": identity["runtime_seconds"],
        "peak_memory_mb": identity["peak_memory_mb"],
        "time_spent_hours": identity["time_spent_hours"],
        "external_resources": identity["external_resources"],
        "ai_tools": identity["ai_tools"],
    }
    if set(submission) != set(OFFICIAL_FIELDS):
        raise SubmissionBuildError("submission fields drifted from the official schema")
    return submission


def _dataset_record(dataset: Mapping[str, Any]) -> dict[str, Any]:
    version = dataset.get("version")
    url = dataset.get("url")
    digest = dataset.get("sha256")
    size = dataset.get("bytes")
    if not isinstance(version, str) or not version.strip():
        raise SubmissionBuildError("dataset version must be a nonempty string")
    if not isinstance(url, str):
        raise SubmissionBuildError("dataset url must be a string")
    _https_url(url, "dataset url")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise SubmissionBuildError(
            "dataset sha256 must contain exactly 64 hexadecimal characters"
        )
    if type(size) is not int or size <= 0:
        raise SubmissionBuildError("dataset bytes must be a positive integer")
    return {"version": version, "url": url, "sha256": digest, "bytes": size}


def _store_checkpoint(
    output: Path,
    records: dict[str, dict[str, Any]],
    weights: list[dict[str, Any]],
    checkpoint: Mapping[str, Any],
) -> None:
    path = Path(checkpoint["path"])
    if not path.is_file():
        raise SubmissionBuildError(f"checkpoint is missing: {path}")
    relative = f"checkpoints/{path.name}"
    if relative in records:
        raise SubmissionBuildError(f"duplicate checkpoint name: {path.name}")
    _store_bytes(output, records, relative, path.read_bytes())
    url = checkpoint.get("url")
    if url is None:
        return
    if not isinstance(url, str):
        raise SubmissionBuildError("checkpoint url must be a string")
    _https_url(url, "weight url")
    record = records[relative]
    weights.append(
        {"url": url, "sha256": record["sha256"], "bytes": record["bytes"]}
    )


def _store_bytes(
    output: Path,
    records: dict[str, dict[str, Any]],
    relative: str,
    data: bytes,
) -> None:
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    records[relative] = {**artifact_record(destination), "path": relative}


def _archive_predictions(files: Sequence[Path], destination: Path) -> None:
    root = files[0].parent
    with tarfile.open(destination, "w", format=tarfile.PAX_FORMAT) as archive:
        for path in files:
            info = tarfile.TarInfo(path.relative_to(root).as_posix())
            info.size = path.stat().st_size
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.mode = 0o644
            info.type = tarfile.REGTYPE
            with path.open("rb") as handle:
                archive.addfile(info, handle)


def _bundle_report(iou: float, *, official: bool, timing_unmeasured: bool) -> str:
    measured = json.dumps(iou)
    lines = [
        "# Submission bundle report",
        "",
        HISTORICAL_REFERENCE,
        "",
        f"Measured document-macro IoU in the unedited metrics file: {measured}.",
        "",
    ]
    if official:
        lines.append(
            "Official public-validation document-macro IoU from the unedited "
            f"metrics file: {measured}."
        )
    else:
        lines.append(ENSEMBLE_STATUS)
    lines.extend(["", FIXTURE_EXAMPLES, "", NO_PRIVATE_TEST, ""])
    if timing_unmeasured:
        lines.extend([UNMEASURED_TIMING, ""])
    lines.extend(
        [
            "## Limitations",
            "",
            "Unknown pixels are unscored. The historical 37-query reference is a",
            "different suite. This bundle does not claim a public-validation score",
            "beyond the unedited metrics file copied beside this report. Fixture",
            "panels are not performance evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _require_command(command: str) -> str:
    if not isinstance(command, str) or not command.strip():
        raise SubmissionBuildError("inference_command must be a nonempty string")
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise SubmissionBuildError("inference_command has invalid quoting") from exc
    if not tokens or tokens[0].startswith("--"):
        raise SubmissionBuildError("inference_command must name a program")
    for flag in ("--inputs", "--data-root", "--output-dir"):
        matches = [
            index
            for index, token in enumerate(tokens)
            if token == flag or token.startswith(flag + "=")
        ]
        if len(matches) != 1:
            raise SubmissionBuildError(
                f"inference_command must contain exactly one {flag}"
            )
        index = matches[0]
        value = (
            tokens[index].split("=", 1)[1]
            if "=" in tokens[index]
            else (tokens[index + 1] if index + 1 < len(tokens) else "")
        )
        if not value or value.startswith("--"):
            raise SubmissionBuildError(f"{flag} requires a value")
    return command


def _text_list(values: Sequence[str] | None, name: str) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise SubmissionBuildError(f"{name} must be a list")
    texts: list[str] = []
    for entry in values:
        if not isinstance(entry, str) or not entry.strip():
            raise SubmissionBuildError(f"{name} entries must be nonempty strings")
        texts.append(entry)
    return texts


def _https_url(value: str, name: str) -> None:
    parts = urlsplit(value)
    host = parts.hostname
    if (
        parts.scheme != "https"
        or not host
        or parts.username
        or parts.password
        or any(character.isspace() for character in value)
        or host.endswith(".invalid")
    ):
        raise SubmissionBuildError(f"{name} must be a valid HTTPS URL")


def _nonnegative(value: Any, name: str) -> None:
    if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(
        value
    ) or value < 0:
        raise SubmissionBuildError(f"{name} must be a finite nonnegative number")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _validate_with_official_schema(path: Path) -> None:
    validator = Path(__file__).resolve().parents[1] / "challenge" / "validate_submission.py"
    if not validator.is_file():
        raise SubmissionBuildError(f"official validator is missing: {validator}")
    spec = importlib.util.spec_from_file_location(
        "hatchmatch_official_validate_submission",
        validator,
    )
    if spec is None or spec.loader is None:
        raise SubmissionBuildError("official validator could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.validate_submission(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise SubmissionBuildError(
            "official submission validator rejected the bundle: " + str(exc)
        ) from exc


def _remove_tree(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
