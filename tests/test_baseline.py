import json
import numpy as np
from PIL import Image
from pathlib import Path
import subprocess
import sys
import pytest
import yaml

from hatchmatch.baseline import baseline_probability
from hatchmatch.contracts import Box, load_requests
import scripts.run_experiments as experiments

OFFICIAL_REVISION = "65b98e480f2a10b82f974f4feb519ee4e012d66c"


def _write_inputs(path: Path, examples: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {"schema": "hatch-matching-inputs/v1", "examples": examples}
        ),
        encoding="utf-8",
    )


def _example(**overrides: object) -> dict[str, object]:
    example: dict[str, object] = {
        "id": "example-1",
        "image": "page.png",
        "width": 28,
        "height": 20,
        "query_box": [2, 3, 14, 16],
        "context_boxes": [],
    }
    example.update(overrides)
    return example


def _fake_challenge(path: Path) -> str:
    path.mkdir()
    (path / "evaluate.py").write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--manifest")
parser.add_argument("--data-root")
parser.add_argument("--predictions")
parser.add_argument("--output")
args = parser.parse_args()
assert (Path(args.predictions) / "example-1.png").is_file()
Path(args.output).write_text(
    json.dumps({"summary": {"document_macro_iou": 0.25}}),
    encoding="utf-8",
)
        """,
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "evaluate.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_config(
    path: Path,
    *,
    challenge_dir: Path,
    revision: str,
    inputs: Path,
    evaluation_manifest: Path,
    data_root: Path,
    run_dir: Path,
    max_dimension: int | None,
) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "challenge_dir": str(challenge_dir),
                "challenge_revision": revision,
                "inputs": str(inputs),
                "evaluation_manifest": str(evaluation_manifest),
                "data_root": str(data_root),
                "run_dir": str(run_dir),
                "threshold": 0.5,
                "max_dimension": max_dimension,
            }
        ),
        encoding="utf-8",
    )


def test_baseline_prefers_matching_orientation():
    image = np.full((128, 192), 255, np.uint8)
    image[:, 8:64:6] = 0
    for offset in range(-64, 192, 6):
        yy = np.arange(128)
        xx = yy + offset
        valid = (xx >= 96) & (xx < 176)
        image[yy[valid], xx[valid]] = 0

    score = baseline_probability(image, Box(8, 8, 56, 120))

    assert score[32:96, 8:64].mean() > score[32:96, 112:168].mean() + 0.2


def test_baseline_returns_finite_native_probability_map():
    image = np.full((31, 47), 255, dtype=np.uint8)

    score = baseline_probability(image, Box(5, 7, 21, 26))

    assert score.shape == image.shape
    assert score.dtype == np.float32
    assert np.isfinite(score).all()
    assert score.min() >= 0.0
    assert score.max() <= 1.0


def test_baseline_is_deterministic():
    image = np.full((32, 48), 255, dtype=np.uint8)
    image[::5, :] = 0
    query_box = Box(2, 3, 30, 25)

    first = baseline_probability(image, query_box)
    second = baseline_probability(image, query_box)

    np.testing.assert_array_equal(first, second)


def test_baseline_runner_full_subcommand_uses_label_free_inputs(tmp_path):
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    Image.fromarray(np.full((20, 28), 255, np.uint8)).save(data_root / "page.png")
    inputs = tmp_path / "validation-inputs.json"
    _write_inputs(inputs, [_example()])
    evaluation_manifest = data_root / "val.json"
    evaluation_manifest.write_text(
        json.dumps({"examples": [{"labels": {"private": "not-for-inference"}}]}),
        encoding="utf-8",
    )
    challenge_dir = tmp_path / "challenge"
    revision = _fake_challenge(challenge_dir)
    run_dir = tmp_path / "run"
    config_path = tmp_path / "baseline.yaml"
    _write_config(
        config_path,
        challenge_dir=challenge_dir,
        revision=revision,
        inputs=inputs,
        evaluation_manifest=evaluation_manifest,
        data_root=data_root,
        run_dir=run_dir,
        max_dimension=None,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiments.py",
            "baseline",
            "--config",
            str(config_path),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["summary"]["document_macro_iou"] == 0.25
    with Image.open(run_dir / "predictions" / "example-1.png") as prediction:
        assert prediction.size == (28, 20)
        assert prediction.mode == "L"
        assert set(np.asarray(prediction).ravel()).issubset({0, 255})


def test_runner_rejects_mismatched_challenge_revision(tmp_path):
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    Image.fromarray(np.full((20, 28), 255, np.uint8)).save(data_root / "page.png")
    inputs = tmp_path / "inputs.json"
    _write_inputs(inputs, [_example()])
    evaluation_manifest = tmp_path / "val.json"
    evaluation_manifest.write_text("{}", encoding="utf-8")
    challenge_dir = tmp_path / "challenge"
    _fake_challenge(challenge_dir)
    config_path = tmp_path / "baseline.yaml"
    _write_config(
        config_path,
        challenge_dir=challenge_dir,
        revision="0" * 40,
        inputs=inputs,
        evaluation_manifest=evaluation_manifest,
        data_root=data_root,
        run_dir=tmp_path / "run",
        max_dimension=None,
    )

    with pytest.raises(RuntimeError, match="revision"):
        experiments.run_baseline(config_path)


def test_runner_rejects_modified_evaluator_at_matching_head(tmp_path):
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    Image.fromarray(np.full((20, 28), 255, np.uint8)).save(data_root / "page.png")
    inputs = tmp_path / "inputs.json"
    _write_inputs(inputs, [_example()])
    evaluation_manifest = tmp_path / "val.json"
    evaluation_manifest.write_text("{}", encoding="utf-8")
    challenge_dir = tmp_path / "challenge"
    revision = _fake_challenge(challenge_dir)
    (challenge_dir / "evaluate.py").write_text(
        "raise RuntimeError('modified evaluator executed')\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "baseline.yaml"
    run_dir = tmp_path / "run"
    _write_config(
        config_path,
        challenge_dir=challenge_dir,
        revision=revision,
        inputs=inputs,
        evaluation_manifest=evaluation_manifest,
        data_root=data_root,
        run_dir=run_dir,
        max_dimension=None,
    )

    with pytest.raises(RuntimeError, match="evaluate.py.*modified"):
        experiments.run_baseline(config_path)

    assert not (run_dir / "metrics.json").exists()


def test_resize_maps_query_with_actual_axis_scales_and_stays_registered(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    Image.fromarray(np.full((17, 31), 255, np.uint8)).save(data_root / "page.png")
    inputs = tmp_path / "inputs.json"
    _write_inputs(
        inputs,
        [
            _example(
                width=31,
                height=17,
                query_box=[3, 10, 20, 14],
            )
        ],
    )
    requests = load_requests(inputs, data_root)
    observed: dict[str, object] = {}

    def fake_baseline(gray, query_box):
        observed["shape"] = gray.shape
        observed["box"] = query_box
        probability = np.zeros(gray.shape, np.float32)
        probability[
            query_box.y0 : query_box.y1, query_box.x0 : query_box.x1
        ] = 1.0
        return probability

    monkeypatch.setattr(experiments, "baseline_probability", fake_baseline)
    predictions = tmp_path / "predictions"

    experiments._predict(requests, predictions, threshold=0.5, max_dimension=10)

    assert observed == {
        "shape": (5, 10),
        "box": Box(0, 2, 7, 5),
    }
    with Image.open(predictions / "example-1.png") as prediction:
        selected = np.asarray(prediction) != 0
    assert selected.any()
    assert selected[12, 11]


def test_native_resolution_is_the_default():
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/baseline.yaml").read_text()
    )

    assert config["max_dimension"] is None
    assert config["challenge_revision"] == OFFICIAL_REVISION
    assert "inputs" in config
    assert "evaluation_manifest" in config
    assert "manifest" not in config


def test_resource_bounded_evaluation_config_is_explicit():
    config = yaml.safe_load(
        (
            Path(__file__).parents[1] / "configs/baseline-eval-1600.yaml"
        ).read_text()
    )

    assert config["max_dimension"] == 1600
    assert config["challenge_revision"] == OFFICIAL_REVISION
    assert config["inputs"] == "../validation-inputs.json"
    assert config["evaluation_manifest"] == "../dataset/val.json"
