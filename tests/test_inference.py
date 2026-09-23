import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

import inference
from hatchmatch.baseline import baseline_probability
from hatchmatch.calibrate import CHANNEL_NAMES
from hatchmatch.checkpoints import checkpoint_sha256, save_checkpoint
from hatchmatch.contracts import Box
from hatchmatch.model import QuerySegFormer

ROOT = Path(__file__).resolve().parents[1]
MIT_B2_REVISION = "3bb39e8739149c3777d0325349b2a6c32c6413db"
INPUT_SCHEMA = "hatch-matching-inputs/v1"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "inference.py"), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _official_manifest(example: dict[str, object]) -> dict[str, object]:
    return {"schema": INPUT_SCHEMA, "examples": [example]}


def test_inference_cli_writes_one_request_and_metadata(tmp_path: Path) -> None:
    config_path = ROOT / "configs" / "test.yaml"
    assert config_path.is_file()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["schema"] == "hatchmatch-inference/v1"
    assert config["mode"] == "classical"
    assert config.get("checkpoints") in (None, [])
    assert not config.get("calibration")

    Image.fromarray(np.full((16, 20), 255, np.uint8)).save(tmp_path / "drawing.png")
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps(
            _official_manifest(
                {
                    "id": "q1",
                    "image": "drawing.png",
                    "width": 20,
                    "height": 16,
                    "query_box": [1, 1, 8, 8],
                    "context_boxes": [],
                }
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "predictions"
    result = _run(
        [
            "--inputs",
            str(inputs),
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--config",
            str(config_path),
            "--device",
            "cpu",
        ]
    )

    assert result.returncode == 0, result.stderr
    with Image.open(output / "q1.png") as image:
        assert image.mode == "L"
        assert image.size == (20, 16)
        assert set(np.asarray(image).reshape(-1).tolist()) == {255}
    metadata = json.loads((output / "run-metadata.json").read_text(encoding="utf-8"))
    assert metadata["count"] == 1
    assert metadata["elapsed_wall_seconds"] >= 0
    assert metadata["elapsed_wall_seconds"] < 15
    assert metadata["hardware"]["device"] == "cpu"
    assert metadata["hardware"]["cuda_available"] is torch.cuda.is_available()
    assert isinstance(metadata["hardware"]["cpu_count"], int)
    assert isinstance(metadata["hardware"]["platform"], str)
    for name in ("numpy", "torch", "pillow", "pyyaml"):
        assert metadata["package_versions"][name] == importlib.metadata.version(name)
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    assert metadata["git_sha"] == git_sha
    assert metadata["config_sha256"] == hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    assert metadata["checkpoint_hashes"] == {}
    assert "labels" not in json.dumps(metadata)


def test_inference_cli_rejects_labeled_manifest(tmp_path: Path) -> None:
    Image.fromarray(np.full((16, 20), 255, np.uint8)).save(tmp_path / "drawing.png")
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps(
            {
                "schema": "hatch-matching-challenge/v1",
                "examples": [
                    {
                        "id": "q1",
                        "document_id": "d1",
                        "kind": "real",
                        "image": "drawing.png",
                        "width": 20,
                        "height": 16,
                        "query_box": [1, 1, 8, 8],
                        "context_boxes": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "predictions"
    result = _run(
        [
            "--inputs",
            str(inputs),
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--config",
            str(ROOT / "configs" / "test.yaml"),
        ]
    )

    assert result.returncode != 0
    assert "schema" in result.stderr
    assert not (output / "q1.png").exists()
    assert not (output / "run-metadata.json").exists()


def test_inference_cli_fails_the_run_when_any_request_is_invalid(
    tmp_path: Path,
) -> None:
    Image.fromarray(np.full((16, 20), 255, np.uint8)).save(tmp_path / "drawing.png")
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps(
            {
                "schema": INPUT_SCHEMA,
                "examples": [
                    {
                        "id": "q1",
                        "image": "drawing.png",
                        "width": 20,
                        "height": 16,
                        "query_box": [1, 1, 8, 8],
                        "context_boxes": [],
                    },
                    {
                        "id": "q2",
                        "image": "drawing.png",
                        "width": 4,
                        "height": 4,
                        "query_box": [0, 0, 1, 1],
                        "context_boxes": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "predictions"
    result = _run(
        [
            "--inputs",
            str(inputs),
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--config",
            str(ROOT / "configs" / "test.yaml"),
        ]
    )

    assert result.returncode != 0
    assert "size" in result.stderr
    assert list(output.glob("*.png")) == []
    assert not (output / "run-metadata.json").exists()


def test_final_config_declares_ensemble_and_refuses_missing_checkpoints(
    tmp_path: Path,
) -> None:
    config_path = ROOT / "configs" / "final.yaml"
    assert config_path.is_file()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["schema"] == "hatchmatch-inference/v1"
    assert config["mode"] == "ensemble"
    assert config["model"] == {
        "backbone": "nvidia/mit-b2",
        "pretrained": True,
        "revision": MIT_B2_REVISION,
        "decoder_channels": 128,
    }
    expected_paths = [
        f"../runs/train-b2/fold-{fold}/seed-{seed}/best.pt"
        for fold in (0, 1, 2, 3, 4)
        for seed in (20260922, 20260923, 20260924)
    ]
    assert [item["path"] for item in config["checkpoints"]] == expected_paths
    assert len(expected_paths) == 15
    assert config["calibration"] == "../runs/calibration/best.json"
    assert config["tta"] == ["identity", "horizontal", "vertical", "rot90"]
    assert config["tile_size"] == 256
    assert config["query_size"] == 96

    Image.fromarray(np.full((16, 20), 255, np.uint8)).save(tmp_path / "drawing.png")
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps(
            _official_manifest(
                {
                    "id": "q1",
                    "image": "drawing.png",
                    "width": 20,
                    "height": 16,
                    "query_box": [1, 1, 8, 8],
                    "context_boxes": [],
                }
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "predictions"
    result = _run(
        [
            "--inputs",
            str(inputs),
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--config",
            str(config_path),
        ]
    )

    assert result.returncode != 0
    assert "missing" in result.stderr.lower()
    assert result.stderr.count("best.pt") >= 15
    for path in expected_paths:
        assert path in result.stderr
    assert not (output / "q1.png").exists()
    assert not (output / "run-metadata.json").exists()


def test_calibration_file_replaces_raw_classical_threshold(tmp_path: Path) -> None:
    Image.fromarray(np.full((16, 20), 255, np.uint8)).save(tmp_path / "drawing.png")
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "schema": "hatchmatch-calibration/v1",
                "fusion": {
                    "features": [
                        "neural",
                        "classical",
                        "foreground",
                        "local_variance",
                        "query_compatibility",
                    ],
                    "coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "intercept": -20.0,
                },
                "parameters": {
                    "threshold": 0.5,
                    "min_component": 1,
                    "close_radius": 0,
                    "foreground_floor": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "classical-calibrated.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema": "hatchmatch-inference/v1",
                "mode": "classical",
                "threshold": 0.5,
                "calibration": str(calibration_path),
            }
        ),
        encoding="utf-8",
    )
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps(
            _official_manifest(
                {
                    "id": "q1",
                    "image": "drawing.png",
                    "width": 20,
                    "height": 16,
                    "query_box": [1, 1, 8, 8],
                    "context_boxes": [],
                }
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "predictions"
    result = _run(
        [
            "--inputs",
            str(inputs),
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--config",
            str(config_path),
        ]
    )

    assert result.returncode == 0, result.stderr
    with Image.open(output / "q1.png") as image:
        assert image.size == (20, 16)
        assert set(np.asarray(image).reshape(-1).tolist()) == {0}


def test_checksummed_checkpoint_drives_the_mask(tmp_path: Path) -> None:
    gray = np.zeros((16, 20), np.uint8)
    gray[2:10, 2:10] = 255
    Image.fromarray(gray).save(tmp_path / "drawing.png")
    classical = baseline_probability(gray, Box(2, 2, 8, 8))
    assert not np.all(classical >= 0.5)

    model = QuerySegFormer(
        backbone="nvidia/mit-b0",
        pretrained=False,
        decoder_channels=8,
    ).eval()
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.fill_(20.0)
    checkpoint_path = tmp_path / "fold.pt"
    save_checkpoint(
        checkpoint_path,
        {
            "state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "architecture": {
                "backbone": "nvidia/mit-b0",
                "pretrained": False,
                "revision": "b" * 40,
                "decoder_channels": 8,
            },
        },
    )
    digest = checkpoint_sha256(checkpoint_path)
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "schema": "hatchmatch-calibration/v1",
                "fusion": {
                    "features": [
                        "neural",
                        "classical",
                        "foreground",
                        "local_variance",
                        "query_compatibility",
                    ],
                    "coefficients": [12.0, 0.0, 0.0, 0.0, 0.0],
                    "intercept": -6.0,
                },
                "parameters": {
                    "threshold": 0.5,
                    "min_component": 1,
                    "close_radius": 0,
                    "foreground_floor": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "ensemble.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema": "hatchmatch-inference/v1",
                "mode": "ensemble",
                "threshold": 0.5,
                "tile_size": 64,
                "overlap": 0,
                "query_size": 32,
                "batch_size": 1,
                "tta": ["identity"],
                "model": {
                    "backbone": "nvidia/mit-b0",
                    "pretrained": False,
                    "revision": "b" * 40,
                    "decoder_channels": 8,
                },
                "calibration": str(calibration_path),
                "checkpoints": [
                    {"path": str(checkpoint_path), "sha256": digest},
                ],
            }
        ),
        encoding="utf-8",
    )
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps(
            _official_manifest(
                {
                    "id": "q1",
                    "image": "drawing.png",
                    "width": 20,
                    "height": 16,
                    "query_box": [2, 2, 8, 8],
                    "context_boxes": [],
                }
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "predictions"
    result = _run(
        [
            "--inputs",
            str(inputs),
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--config",
            str(config_path),
            "--device",
            "cpu",
        ]
    )

    assert result.returncode == 0, result.stderr
    with Image.open(output / "q1.png") as image:
        assert image.size == (20, 16)
        assert set(np.asarray(image).reshape(-1).tolist()) == {255}
    metadata = json.loads((output / "run-metadata.json").read_text(encoding="utf-8"))
    assert metadata["count"] == 1
    assert metadata["checkpoint_hashes"] == {str(checkpoint_path.resolve()): digest}


def test_inference_source_does_not_reference_label_loaders() -> None:
    path = ROOT / "inference.py"
    assert path.is_file()
    source = path.read_text(encoding="utf-8")
    for token in (
        "hatchmatch.data",
        "hatchmatch.train",
        "load_training_examples",
        "load_out_of_fold_examples",
        "val.json",
        "train.json",
    ):
        assert token not in source


def _two_orientation_drawing() -> np.ndarray:
    gray = np.full((32, 48), 255, np.uint8)
    gray[4:28:4, 4:22] = 0
    gray[4:28, 28:46:4] = 0
    return gray


def _classical_calibration(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "hatchmatch-calibration/v1",
                "fusion": {
                    "features": list(CHANNEL_NAMES),
                    "coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "intercept": -20.0,
                },
                "parameters": {
                    "threshold": 0.5,
                    "min_component": 1,
                    "close_radius": 0,
                    "foreground_floor": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )


def test_fusion_channels_use_distinct_public_feature_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gray = _two_orientation_drawing()
    Image.fromarray(gray).save(tmp_path / "drawing.png")
    query_box = Box(4, 4, 22, 28)
    calibration_path = tmp_path / "calibration.json"
    _classical_calibration(calibration_path)
    config_path = tmp_path / "calibrated.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema": "hatchmatch-inference/v1",
                "mode": "classical",
                "threshold": 0.5,
                "calibration": str(calibration_path),
            }
        ),
        encoding="utf-8",
    )
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps(
            _official_manifest(
                {
                    "id": "q1",
                    "image": "drawing.png",
                    "width": 48,
                    "height": 32,
                    "query_box": [4, 4, 22, 28],
                    "context_boxes": [],
                }
            )
        ),
        encoding="utf-8",
    )
    captured: list[object] = []
    original_predict = inference.FusionCalibrator.predict

    def spy(self: object, channels: list[object]) -> list[np.ndarray]:
        captured.extend(channels)
        return original_predict(self, channels)

    monkeypatch.setattr(inference.FusionCalibrator, "predict", spy)
    inference.run_inference(
        inputs,
        tmp_path,
        tmp_path / "predictions",
        config_path=config_path,
        device="cpu",
    )

    assert len(captured) == 1
    fusion = captured[0]
    planes = {name: np.asarray(getattr(fusion, name)) for name in CHANNEL_NAMES}
    assert [plane.shape for plane in planes.values()] == [gray.shape] * len(CHANNEL_NAMES)
    assert len({id(plane) for plane in planes.values()}) == len(CHANNEL_NAMES)
    assert not np.array_equal(planes["query_compatibility"], planes["classical"])
    np.testing.assert_array_equal(
        planes["classical"], baseline_probability(gray, query_box)
    )

    import hatchmatch.features as features

    foreground = getattr(features, "foreground", None)
    local_variance = getattr(features, "local_variance", None)
    query_compatibility = getattr(features, "query_compatibility", None)
    assert callable(foreground)
    assert callable(local_variance)
    assert callable(query_compatibility)
    np.testing.assert_array_equal(planes["foreground"], foreground(gray))
    np.testing.assert_array_equal(planes["local_variance"], local_variance(gray))
    np.testing.assert_array_equal(
        planes["query_compatibility"], query_compatibility(gray, query_box)
    )
    source = (ROOT / "inference.py").read_text(encoding="utf-8")
    assert "_iter_texture_channels" not in source
    for name in ("foreground", "local_variance", "query_compatibility"):
        assert f"{name}(" in source


def test_later_request_failure_publishes_no_earlier_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("q1.png", "q2.png"):
        Image.fromarray(np.full((16, 20), 255, np.uint8)).save(tmp_path / name)
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps(
            {
                "schema": INPUT_SCHEMA,
                "examples": [
                    {
                        "id": "q1",
                        "image": "q1.png",
                        "width": 20,
                        "height": 16,
                        "query_box": [1, 1, 8, 8],
                        "context_boxes": [],
                    },
                    {
                        "id": "q2",
                        "image": "q2.png",
                        "width": 20,
                        "height": 16,
                        "query_box": [1, 1, 8, 8],
                        "context_boxes": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    calls = {"count": 0}
    original = inference.baseline_probability

    def fail_second_mask(gray: np.ndarray, query_box: Box) -> np.ndarray:
        calls["count"] += 1
        if calls["count"] >= 2:
            raise RuntimeError("second mask failed")
        return original(gray, query_box)

    monkeypatch.setattr(inference, "baseline_probability", fail_second_mask)
    output = tmp_path / "predictions"

    with pytest.raises(RuntimeError, match="second mask failed"):
        inference.run_inference(
            inputs,
            tmp_path,
            output,
            config_path=ROOT / "configs" / "test.yaml",
            device="cpu",
        )

    assert calls["count"] == 2
    assert not (output / "q1.png").exists()
    assert not (output / "q2.png").exists()
    assert not (output / "run-metadata.json").exists()


def test_inference_cli_rejects_unknown_arguments(tmp_path: Path) -> None:
    result = _run(
        [
            "--inputs",
            str(tmp_path / "inputs.json"),
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "predictions"),
            "--labels",
            str(tmp_path / "labels.json"),
        ]
    )

    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr
    assert "--labels" in result.stderr
