"""Checksummed submission bundles and the technical report contract."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import scripts.run_experiments as experiments
from scripts.build_submission import (
    SubmissionBuildError,
    artifact_record,
    build_submission,
    normalize_repository_url,
    verify_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_REVISION = "65b98e480f2a10b82f974f4feb519ee4e012d66c"
DATASET = {
    "version": "1.0",
    "url": (
        "https://github.com/TruTec-AI/hatch-matching-challenge/"
        "releases/download/dataset-v1/trutec-hatch-dataset-v1.zip"
    ),
    "sha256": "d5c1c845898669fe3941c363db3f328ed9d50c828acf7daeed7c9b2ae74b2229",
    "bytes": 422368602,
}
INFERENCE_COMMAND = (
    "python inference.py --inputs validation-inputs.json "
    "--data-root dataset --output-dir predictions "
    "--config configs/baseline-eval-1600.yaml"
)


def test_artifact_record(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"abc")
    record = artifact_record(path)
    assert record["bytes"] == 3
    assert record["sha256"] == hashlib.sha256(b"abc").hexdigest()


def test_normalize_repository_url_drops_credentials() -> None:
    url = normalize_repository_url(
        "https://x-access-token:secret@github.com/TarunYadgirkar/extra.git"
    )
    assert url == "https://github.com/TarunYadgirkar/extra"
    assert "secret" not in url


def test_builder_refuses_run_without_unedited_metrics(tmp_path: Path) -> None:
    output = tmp_path / "dist"
    with pytest.raises(SubmissionBuildError, match="unedited metrics") as raised:
        build_submission(tmp_path / "missing-run", output)
    message = str(raised.value)
    assert "config.json" in message
    assert "metrics.json" in message
    assert "provenance.json" in message
    assert not output.exists()


def test_builder_refuses_edited_metrics_without_writing_a_score(
    tmp_path: Path,
) -> None:
    run = _sealed_run(tmp_path, document_macro_iou=0.25)
    metrics = run / "metrics.json"
    original = metrics.read_bytes()
    metrics.write_bytes(original + b"\n")
    output = tmp_path / "dist"
    with pytest.raises(SubmissionBuildError, match="unedited metrics") as raised:
        _build(run, output)
    assert "metrics.json" in str(raised.value)
    assert not output.exists()
    assert metrics.read_bytes() == original + b"\n"


def test_builder_refuses_to_invent_document_macro_iou(tmp_path: Path) -> None:
    run = _sealed_run(tmp_path, metrics={"examples": [], "summary": {}})
    output = tmp_path / "dist"
    with pytest.raises(SubmissionBuildError, match="document_macro_iou") as raised:
        _build(run, output)
    assert "invent" in str(raised.value).lower()
    assert not output.exists()


def test_builder_refuses_private_examples(tmp_path: Path) -> None:
    run = _sealed_run(tmp_path, document_macro_iou=0.25)
    examples = _examples()
    examples[1]["split"] = "private_test"
    output = tmp_path / "dist"
    with pytest.raises(SubmissionBuildError, match="private") as raised:
        _build(run, output, examples=examples)
    assert "private-test" in str(raised.value)
    assert not output.exists()


def test_fixture_bundle_checksums_and_passes_official_validator(
    tmp_path: Path,
) -> None:
    run = _sealed_run(tmp_path, document_macro_iou=0.25)
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"not-a-real-weight")
    environment = tmp_path / "lock.txt"
    environment.write_bytes(b"python==3.12\n")
    output = _build(
        run,
        tmp_path / "dist",
        environment_lock=environment,
        checkpoints=(
            {
                "path": checkpoint,
                "url": "https://github.com/TarunYadgirkar/extra/releases/download/fixture/checkpoint.bin",
            },
        ),
    )

    metrics_bytes = (run / "metrics.json").read_bytes()
    config_bytes = (run / "config.json").read_bytes()
    assert (output / "metrics.json").read_bytes() == metrics_bytes
    assert (output / "config.json").read_bytes() == config_bytes
    assert (run / "metrics.json").read_bytes() == metrics_bytes

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert verify_manifest(output) == []
    artifacts = manifest["artifacts"]
    for name in (
        "predictions.tar",
        "metrics.json",
        "config.json",
        "environment.lock",
        "checkpoints/checkpoint.bin",
        "examples/fixture-drawing-1.png",
        "examples/fixture-drawing-2.png",
        "examples/fixture-drawing-3.png",
        "technical-report.md",
        "submission.json",
    ):
        assert name in artifacts
        record = artifact_record(output / artifacts[name]["path"])
        assert artifacts[name]["sha256"] == record["sha256"]
        assert artifacts[name]["bytes"] == record["bytes"]
    assert artifacts["metrics.json"]["sha256"] == hashlib.sha256(metrics_bytes).hexdigest()
    assert artifacts["environment.lock"]["bytes"] == len(b"python==3.12\n")
    assert artifacts["checkpoints/checkpoint.bin"]["sha256"] == hashlib.sha256(
        b"not-a-real-weight"
    ).hexdigest()

    listed = {record["path"] for record in artifacts.values()}
    on_disk = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert on_disk == listed

    submission = json.loads((output / "submission.json").read_text(encoding="utf-8"))
    assert set(submission) == {
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
    }
    assert submission["schema_version"] == "1.0"
    assert submission["repository"] == "https://github.com/TarunYadgirkar/extra"
    assert submission["commit"] == "ebfccb128a486ae0348665e64b269aa3047883fc"
    assert submission["inference_command"] == INFERENCE_COMMAND
    assert submission["environment"] == "environment.lock"
    assert submission["validation_metrics"] == "metrics.json"
    assert submission["runtime_seconds"] == 12.5
    assert submission["time_spent_hours"] == 3
    assert submission["peak_memory_mb"] is None
    assert submission["hardware"] == "fixture CPU; no GPU"
    assert "document_macro_iou" not in submission
    assert submission["weights"] == [
        {
            "url": "https://github.com/TarunYadgirkar/extra/releases/download/fixture/checkpoint.bin",
            "sha256": hashlib.sha256(b"not-a-real-weight").hexdigest(),
            "bytes": len(b"not-a-real-weight"),
        }
    ]
    assert manifest["evaluator_revision"] == EVALUATOR_REVISION
    assert manifest["dataset"]["sha256"] == DATASET["sha256"]
    assert manifest["timing"]["runtime_seconds"] == 12.5
    assert "loading" in manifest["timing"]["definition"]
    assert "file writing" in manifest["timing"]["definition"]
    assert manifest["official_public_validation_document_macro_iou"] is None

    report = (output / "technical-report.md").read_text(encoding="utf-8")
    assert (
        "Historical reference: 73.73% document-macro IoU on a different "
        "37-query suite. That figure is not the score of this bundle."
    ) in report
    assert "Measured document-macro IoU in the unedited metrics file: 0.25." in report
    assert "0.4779169321" not in report
    assert "The neural ensemble has not been trained or scored here." in report
    assert (
        "The three visual examples are representative fixtures until real "
        "validation predictions exist."
    ) in report
    assert "This bundle does not display private-test material." in report

    _assert_example_colors(output / "examples" / "fixture-drawing-1.png")
    _validate_official(output / "submission.json")


def test_second_build_reproduces_prediction_archive_checksum(tmp_path: Path) -> None:
    run = _sealed_run(tmp_path, document_macro_iou=0.25)
    first = _build(run, tmp_path / "one")
    second = _build(run, tmp_path / "two")
    assert artifact_record(first / "predictions.tar") == artifact_record(
        second / "predictions.tar"
    )


def test_cli_refuses_unsealed_final_run(tmp_path: Path) -> None:
    output = tmp_path / "dist"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_submission.py"),
            "--run",
            str(ROOT / "runs" / "final"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "unedited metrics" in completed.stderr
    assert not output.exists()


def test_submission_example_passes_official_validator() -> None:
    path = ROOT / "submission.example.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    encoded = json.dumps(payload)
    assert "REPLACE_" not in encoded
    assert "document_macro_iou" not in payload
    assert payload["weights"] == []
    assert payload["peak_memory_mb"] is None
    _validate_official(path)


def test_technical_report_separates_historical_reference_from_measured_score() -> None:
    report = (ROOT / "docs" / "technical-report.md").read_text(encoding="utf-8")
    assert (
        "The 73.73% figure is a historical reference on a different 37-query "
        "suite. It is not a score on this challenge's public-validation split."
    ) in report
    assert (
        "The only measured challenge score is the capped-1600 classical "
        "baseline document-macro IoU 0.4779169321."
    ) in report
    assert "0.4779169321376195" in report
    assert "The neural ensemble has not been trained or scored here." in report
    assert (
        "The report examples are representative fixtures until real "
        "validation predictions exist."
    ) in report
    assert "does not display private-test material" in report
    assert "04e1ac742653c468b52f8470bf8829273589d8b6412433ccad674de7d931667c" in report


def _build(
    run: Path,
    output: Path,
    *,
    examples: list[dict[str, object]] | None = None,
    environment_lock: Path | None = None,
    checkpoints: tuple[dict[str, object], ...] = (),
) -> Path:
    if environment_lock is None:
        environment_lock = run.parent / "environment.lock"
        environment_lock.write_bytes(b"locked\n")
    return build_submission(
        run,
        output,
        repository="https://github.com/TarunYadgirkar/extra",
        commit="ebfccb128a486ae0348665e64b269aa3047883fc",
        inference_command=INFERENCE_COMMAND,
        environment_lock=environment_lock,
        hardware="fixture CPU; no GPU",
        runtime_seconds=12.5,
        time_spent_hours=3,
        peak_memory_mb=None,
        evaluator_revision=EVALUATOR_REVISION,
        dataset=DATASET,
        examples=_examples() if examples is None else examples,
        checkpoints=checkpoints,
        external_resources=["Public challenge dataset only."],
        ai_tools=[
            "An automated coding agent assisted implementation. It did not train or score the neural ensemble."
        ],
    )


def _sealed_run(
    tmp_path: Path,
    *,
    document_macro_iou: float | None = None,
    metrics: dict[str, object] | None = None,
) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    if metrics is None:
        metrics = {
            "summary": {
                "document_macro_iou": document_macro_iou,
                "documents": 1,
                "examples": 1,
            }
        }
    (run / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    predictions = run / "predictions"
    predictions.mkdir()
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8), mode="L").save(
        predictions / "q-fixture.png"
    )
    config = tmp_path / "config.yaml"
    config.write_text("mode: classical\n", encoding="utf-8")
    experiments.seal_run(run, config, command="baseline")
    return run


def _examples() -> list[dict[str, object]]:
    examples = []
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


def _assert_example_colors(path: Path) -> None:
    image = np.asarray(Image.open(path))
    assert image.shape == (4, 16, 3)
    assert tuple(image[0, 0]) == (40, 40, 40)
    assert tuple(image[0, 4 + 3]) == (255, 0, 255)
    assert tuple(image[0, 8 + 0]) == (0, 255, 0)
    assert tuple(image[0, 8 + 1]) == (0, 0, 0)
    assert tuple(image[0, 8 + 2]) == (0, 255, 0)
    assert tuple(image[0, 12 + 0]) == (255, 0, 0)
    assert tuple(image[0, 12 + 1]) == (0, 0, 255)
    assert tuple(image[0, 12 + 2]) == (128, 128, 128)
    assert tuple(image[0, 12 + 3]) == (255, 255, 0)


def _validate_official(path: Path) -> None:
    validator = ROOT / "challenge" / "validate_submission.py"
    spec = importlib.util.spec_from_file_location("official_validate_submission", validator)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_submission(json.loads(path.read_text(encoding="utf-8")))
    completed = subprocess.run(
        [sys.executable, str(validator), "--submission", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
