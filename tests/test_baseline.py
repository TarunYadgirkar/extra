import numpy as np
from PIL import Image
from pathlib import Path
import subprocess
import sys
import yaml

from hatchmatch.baseline import baseline_probability
from hatchmatch.contracts import Box
from scripts.run_experiments import run_baseline


def test_experiment_runner_supports_direct_script_invocation():
    project_root = Path(__file__).parents[1]

    completed = subprocess.run(
        [sys.executable, "scripts/run_experiments.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


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


def test_baseline_runner_writes_predictions_and_official_metrics(tmp_path):
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    Image.fromarray(np.full((20, 28), 255, np.uint8)).save(data_root / "page.png")
    manifest = data_root / "val.json"
    manifest.write_text(
        """
        {"examples": [{
          "id": "example-1", "document_id": "doc-1", "kind": "real",
          "image": "page.png", "width": 28, "height": 20,
          "query_box": [2, 3, 14, 16], "context_boxes": [],
          "labels": {"positive_boxes": [[0, 0, 1, 1]]}
        }]}
        """,
        encoding="utf-8",
    )
    challenge_dir = tmp_path / "challenge"
    challenge_dir.mkdir()
    (challenge_dir / "evaluate.py").write_text(
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
    run_dir = tmp_path / "run"
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "challenge_dir": str(challenge_dir),
                "manifest": str(manifest),
                "data_root": str(data_root),
                "run_dir": str(run_dir),
                "threshold": 0.5,
                "max_dimension": 16,
            }
        ),
        encoding="utf-8",
    )

    metrics = run_baseline(config_path)

    assert metrics["summary"]["document_macro_iou"] == 0.25
    assert (run_dir / "metrics.json").is_file()
    with Image.open(run_dir / "predictions" / "example-1.png") as prediction:
        assert prediction.size == (28, 20)
        assert set(np.asarray(prediction).ravel()).issubset({0, 255})
