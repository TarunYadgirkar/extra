import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from hatchmatch.data import HatchTileDataset, load_training_examples, make_group_folds


SCHEMA = "hatch-matching-challenge/v1"


def _save_mask(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8) * 255, mode="L").save(path)


def _write_training_manifest(root: Path) -> Path:
    drawing = np.full((8, 8), 255, np.uint8)
    drawing[3:6, 3:6] = 0
    Image.fromarray(drawing, mode="L").save(root / "drawing.png")

    positive = np.zeros((8, 8), bool)
    positive[3:5, 3:5] = True
    known = np.ones((8, 8), bool)
    _save_mask(root / "positive.png", positive)
    _save_mask(root / "known.png", known)

    manifest = {
        "schema": SCHEMA,
        "examples": [
            {
                "id": "q1",
                "document_id": "doc1",
                "kind": "real",
                "image": "drawing.png",
                "width": 8,
                "height": 8,
                "query_box": [0, 0, 2, 2],
                "context_boxes": [[6, 6, 8, 8]],
                "labels": {
                    "positive_mask": "positive.png",
                    "known_mask": "known.png",
                },
            }
        ],
    }
    path = root / "train.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_group_folds_never_split_documents() -> None:
    examples = [
        {"id": f"q{i}", "document_id": f"d{i // 2}"} for i in range(20)
    ]

    folds = make_group_folds(examples, count=5, seed=20260922)

    validation_indices = []
    for fold in folds:
        train_docs = {examples[i]["document_id"] for i in fold.train}
        valid_docs = {examples[i]["document_id"] for i in fold.valid}
        assert train_docs.isdisjoint(valid_docs)
        validation_indices.extend(fold.valid)
    assert sorted(validation_indices) == list(range(len(examples)))


def test_generated_stratification_keeps_cad_document_groups_together() -> None:
    examples = [
        {
            "id": f"real-{document}-{query}",
            "document_id": f"doc-{document}",
            "kind": "real",
        }
        for document in range(5)
        for query in range(2)
    ] + [
        {
            "id": f"cad-{document}-{query}",
            "document_id": f"cad-{document}",
            "kind": "generated_cad",
        }
        for document in range(5)
        for query in range(4)
    ]

    folds = make_group_folds(examples, count=5, seed=7)

    for fold in folds:
        valid_kinds = {examples[index]["kind"] for index in fold.valid}
        assert valid_kinds == {"real", "generated_cad"}
        for example in examples:
            same_document = [
                index
                for index, candidate in enumerate(examples)
                if candidate["document_id"] == example["document_id"]
            ]
            assert set(same_document).issubset(fold.train) or set(
                same_document
            ).issubset(fold.valid)


def test_group_folds_require_at_least_one_group_per_fold() -> None:
    examples = [
        {"id": "a", "document_id": "one"},
        {"id": "b", "document_id": "two"},
    ]

    with pytest.raises(ValueError, match="groups"):
        make_group_folds(examples, count=3, seed=0)


def test_training_masks_exclude_unknown_query_and_context_pixels(
    tmp_path: Path,
) -> None:
    manifest = _write_training_manifest(tmp_path)
    examples = load_training_examples(manifest, tmp_path)
    dataset = HatchTileDataset(
        examples,
        tile_size=8,
        query_size=6,
        samples_per_epoch=10,
        seed=11,
        augment=False,
    )

    item = dataset[0]

    assert set(item) == {"image", "query", "texture", "target", "known"}
    assert item["image"].shape == (3, 8, 8)
    assert item["query"].shape == (3, 6, 6)
    assert item["texture"].shape == (4, 8, 8)
    assert item["target"].shape == item["known"].shape == (1, 8, 8)
    assert all(value.dtype == torch.float32 for value in item.values())
    assert not item["known"][0, :2, :2].any()
    assert not item["known"][0, 6:, 6:].any()
    assert not item["target"][0, :2, :2].any()
    assert torch.all(item["target"] <= item["known"])


def test_sampling_schedule_has_exact_four_three_two_one_ratio(
    tmp_path: Path,
) -> None:
    manifest = _write_training_manifest(tmp_path)
    dataset = HatchTileDataset(
        load_training_examples(manifest, tmp_path),
        tile_size=8,
        samples_per_epoch=20,
        seed=4,
        augment=False,
    )

    assert Counter(dataset.sample_kind(index) for index in range(20)) == {
        "positive": 8,
        "hard_negative": 6,
        "blank": 4,
        "random": 2,
    }


def test_dataset_is_deterministic_under_supplied_seed(tmp_path: Path) -> None:
    manifest = _write_training_manifest(tmp_path)
    examples = load_training_examples(manifest, tmp_path)
    first = HatchTileDataset(
        examples,
        tile_size=8,
        query_size=6,
        samples_per_epoch=10,
        seed=19,
        augment=True,
    )
    second = HatchTileDataset(
        examples,
        tile_size=8,
        query_size=6,
        samples_per_epoch=10,
        seed=19,
        augment=True,
    )

    for key in first[3]:
        torch.testing.assert_close(first[3][key], second[3][key], rtol=0, atol=0)
