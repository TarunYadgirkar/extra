"""Classical segmenter: tone fills stay local and outputs stay native."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from hatchmatch.contracts import Box
from hatchmatch.segment import pattern_score, segment_mask

ROOT = Path(__file__).resolve().parents[1]


def _gray_page() -> tuple[np.ndarray, Box]:
    page = np.full((200, 280), 255, dtype=np.uint8)
    page[30:170, 40:240] = 180
    return page, Box(50, 40, 58, 48)


def test_flat_gray_fill_covers_the_region_and_not_the_paper() -> None:
    page, query = _gray_page()
    mask = segment_mask(page, query, max_side=280)

    assert mask.shape == page.shape
    assert mask.dtype == np.bool_
    region = np.zeros(page.shape, dtype=bool)
    region[30:170, 40:240] = True
    region[query.y0 : query.y1, query.x0 : query.x1] = False
    paper = ~region
    paper[query.y0 : query.y1, query.x0 : query.x1] = False
    overlap = int((mask & region).sum())
    assert overlap / int(region.sum()) > 0.85
    assert int((mask & paper).sum()) / int(paper.sum()) < 0.05


def test_pattern_score_marks_flat_gray_as_tone() -> None:
    page, query = _gray_page()
    pattern = pattern_score(page, query, max_side=280)

    assert pattern.branch == "tone"
    assert pattern.native_size == (page.shape[1], page.shape[0])
    assert pattern.score.shape == page.shape


def test_segment_mask_is_deterministic() -> None:
    page, query = _gray_page()
    first = segment_mask(page, query, max_side=280)
    second = segment_mask(page, query, max_side=280)
    assert np.array_equal(first, second)


def test_predict_script_writes_a_native_mask_without_labels(tmp_path: Path) -> None:
    page, query = _gray_page()
    image_path = tmp_path / "page.png"
    Image.fromarray(page).save(image_path)
    inputs = {
        "schema": "hatch-matching-inputs/v1",
        "examples": [
            {
                "id": "gray-fill",
                "image": "page.png",
                "width": page.shape[1],
                "height": page.shape[0],
                "query_box": [query.x0, query.y0, query.x1, query.y1],
                "context_boxes": [],
            }
        ],
    }
    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps(inputs), encoding="utf-8")
    output = tmp_path / "predictions"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "segment_predict.py"),
            "--inputs",
            str(manifest),
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--config",
            str(ROOT / "configs" / "segment.yaml"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    metadata = json.loads(completed.stdout)
    assert metadata["count"] == 1
    assert metadata["runtime_seconds"] >= 0
    with Image.open(output / "gray-fill.png") as written:
        assert written.mode == "L"
        assert written.size == (page.shape[1], page.shape[0])
        values = set(np.asarray(written).reshape(-1).tolist())
    assert values <= {0, 255}
    assert 255 in values
