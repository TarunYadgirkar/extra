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
    blank = np.zeros((8, 8), bool)
    blank[:2, 6:] = True
    _save_mask(root / "positive.png", positive)
    _save_mask(root / "known.png", known)
    _save_mask(root / "blank.png", blank)

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
                    "blank_mask": "blank.png",
                },
            }
        ],
    }
    path = root / "train.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_array_manifest(
    root: Path,
    definitions: list[dict[str, object]],
) -> Path:
    examples = []
    for index, definition in enumerate(definitions):
        identifier = f"sample-{index}"
        image = np.asarray(definition["image"], dtype=np.uint8)
        height, width = image.shape
        image_name = f"{identifier}-drawing.png"
        Image.fromarray(image, mode="L").save(root / image_name)
        labels: dict[str, object] = {}
        for key, value in dict(definition["labels"]).items():
            if isinstance(value, np.ndarray):
                mask_name = f"{identifier}-{key}.png"
                _save_mask(root / mask_name, value)
                labels[key] = mask_name
            else:
                labels[key] = value
        examples.append(
            {
                "id": identifier,
                "document_id": f"doc-{index}",
                "kind": "real",
                "image": image_name,
                "width": width,
                "height": height,
                "query_box": [0, 0, 1, 1],
                "context_boxes": [],
                "labels": labels,
            }
        )
    path = root / "arrays.json"
    path.write_text(
        json.dumps({"schema": SCHEMA, "examples": examples}),
        encoding="utf-8",
    )
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


def test_empty_explicit_known_is_not_inferred_from_positive(
    tmp_path: Path,
) -> None:
    positive = np.zeros((7, 7), bool)
    positive[3, 3] = True
    manifest = _write_array_manifest(
        tmp_path,
        [
            {
                "image": np.full((7, 7), 255, np.uint8),
                "labels": {
                    "positive_mask": positive,
                    "known_boxes": [],
                },
            }
        ],
    )
    dataset = HatchTileDataset(
        load_training_examples(manifest, tmp_path),
        tile_size=3,
        augment=False,
    )

    with pytest.raises(ValueError, match="outside the explicit known domain"):
        dataset[0]


@pytest.mark.parametrize("region", ["negative", "blank"])
def test_explicit_labels_must_be_subsets_of_known(
    tmp_path: Path,
    region: str,
) -> None:
    known = np.zeros((7, 7), bool)
    known[2, 2] = True
    outside = np.zeros((7, 7), bool)
    outside[4, 4] = True
    manifest = _write_array_manifest(
        tmp_path,
        [
            {
                "image": np.full((7, 7), 255, np.uint8),
                "labels": {
                    "known_mask": known,
                    f"{region}_mask": outside,
                },
            }
        ],
    )
    dataset = HatchTileDataset(
        load_training_examples(manifest, tmp_path),
        tile_size=3,
        augment=False,
    )

    with pytest.raises(ValueError, match="outside the explicit known domain"):
        dataset[0]


@pytest.mark.parametrize("region", ["negative", "blank"])
def test_positive_overlap_matches_official_contradiction_rule(
    tmp_path: Path,
    region: str,
) -> None:
    known = np.ones((7, 7), bool)
    positive = np.zeros((7, 7), bool)
    positive[3, 3] = True
    manifest = _write_array_manifest(
        tmp_path,
        [
            {
                "image": np.full((7, 7), 255, np.uint8),
                "labels": {
                    "known_mask": known,
                    "positive_mask": positive,
                    f"{region}_mask": positive,
                },
            }
        ],
    )
    dataset = HatchTileDataset(
        load_training_examples(manifest, tmp_path),
        tile_size=3,
        augment=False,
    )

    with pytest.raises(
        ValueError,
        match="positive and negative/blank labels overlap",
    ):
        dataset[0]


def test_sampling_realizes_each_requested_semantic_domain(
    tmp_path: Path,
) -> None:
    positive = np.zeros((9, 9), bool)
    positive[3:6, 3:6] = True
    hard_known = np.ones((9, 9), bool)
    blank = np.zeros((9, 9), bool)
    blank[3:6, 3:6] = True
    manifest = _write_array_manifest(
        tmp_path,
        [
            {
                "image": np.full((9, 9), 127, np.uint8),
                "labels": {
                    "positive_mask": positive,
                    "known_mask": positive,
                },
            },
            {
                "image": np.zeros((9, 9), np.uint8),
                "labels": {"known_mask": hard_known},
            },
            {
                "image": np.full((9, 9), 255, np.uint8),
                "labels": {
                    "known_mask": blank,
                    "blank_mask": blank,
                },
            },
        ],
    )
    dataset = HatchTileDataset(
        load_training_examples(manifest, tmp_path),
        tile_size=3,
        query_size=3,
        samples_per_epoch=10,
        seed=23,
        augment=False,
        cache_size=0,
    )

    for index in range(10):
        kind = dataset.sample_kind(index)
        item = dataset[index]
        center = (0, 1, 1)
        assert item["known"][center].item() == 1.0
        if kind == "positive":
            assert item["target"][center].item() == 1.0
        elif kind == "hard_negative":
            assert item["target"][center].item() == 0.0
            assert item["image"][center].item() < 0.1
        elif kind == "blank":
            assert item["target"][center].item() == 0.0
            assert item["image"][center].item() > 0.9


def test_missing_sampling_domain_raises_instead_of_relabeling_fallback(
    tmp_path: Path,
) -> None:
    positive = np.zeros((7, 7), bool)
    positive[2:5, 2:5] = True
    manifest = _write_array_manifest(
        tmp_path,
        [
            {
                "image": np.full((7, 7), 127, np.uint8),
                "labels": {
                    "positive_mask": positive,
                    "known_mask": positive,
                },
            }
        ],
    )
    dataset = HatchTileDataset(
        load_training_examples(manifest, tmp_path),
        tile_size=3,
        samples_per_epoch=10,
        seed=3,
        augment=False,
    )
    blank_index = next(
        index
        for index in range(10)
        if dataset.sample_kind(index) == "blank"
    )

    with pytest.raises(ValueError, match="no training example has blank"):
        dataset[blank_index]


def test_each_complete_schedule_block_is_four_three_two_one(
    tmp_path: Path,
) -> None:
    manifest = _write_training_manifest(tmp_path)
    examples = load_training_examples(manifest, tmp_path)
    first = HatchTileDataset(
        examples,
        samples_per_epoch=23,
        seed=88,
        augment=False,
    )
    second = HatchTileDataset(
        examples,
        samples_per_epoch=23,
        seed=88,
        augment=False,
    )

    assert [first.sample_kind(i) for i in range(23)] == [
        second.sample_kind(i) for i in range(23)
    ]
    expected = {
        "positive": 4,
        "hard_negative": 3,
        "blank": 2,
        "random": 1,
    }
    assert Counter(first.sample_kind(i) for i in range(10)) == expected
    assert Counter(first.sample_kind(i) for i in range(10, 20)) == expected


def test_white_gap_inside_hatch_is_not_a_blank_neighborhood(
    tmp_path: Path,
) -> None:
    image = np.zeros((9, 9), np.uint8)
    image[4, 4] = 255
    known = np.ones_like(image, dtype=bool)
    manifest = _write_array_manifest(
        tmp_path,
        [{"image": image, "labels": {"known_mask": known}}],
    )
    dataset = HatchTileDataset(
        load_training_examples(manifest, tmp_path),
        tile_size=3,
        samples_per_epoch=10,
        seed=5,
        augment=False,
    )
    blank_index = next(
        index
        for index in range(10)
        if dataset.sample_kind(index) == "blank"
    )

    with pytest.raises(ValueError, match="no training example has blank"):
        dataset[blank_index]


def test_explicit_blank_is_preferred_even_when_its_pixels_are_dark(
    tmp_path: Path,
) -> None:
    image = np.zeros((9, 9), np.uint8)
    known = np.ones((9, 9), bool)
    blank = np.zeros((9, 9), bool)
    blank[4, 4] = True
    manifest = _write_array_manifest(
        tmp_path,
        [
            {
                "image": image,
                "labels": {
                    "known_mask": known,
                    "blank_mask": blank,
                },
            }
        ],
    )
    dataset = HatchTileDataset(
        load_training_examples(manifest, tmp_path),
        tile_size=3,
        samples_per_epoch=10,
        seed=6,
        augment=False,
    )
    blank_index = next(
        index
        for index in range(10)
        if dataset.sample_kind(index) == "blank"
    )

    item = dataset[blank_index]

    assert item["known"][0, 1, 1].item() == 1.0
    assert item["image"][0, 1, 1].item() == 0.0


def test_default_cache_retains_at_most_one_full_example(tmp_path: Path) -> None:
    definitions = []
    for _ in range(3):
        known = np.ones((7, 7), bool)
        definitions.append(
            {
                "image": np.full((7, 7), 255, np.uint8),
                "labels": {"known_mask": known},
            }
        )
    manifest = _write_array_manifest(tmp_path, definitions)
    dataset = HatchTileDataset(
        load_training_examples(manifest, tmp_path),
        tile_size=3,
        augment=False,
    )

    dataset._load_arrays(0)
    dataset._load_arrays(1)
    dataset._load_arrays(2)

    assert dataset.cache_size <= 1
    assert len(dataset._cache) <= 1


def test_set_epoch_documents_persistent_worker_behavior() -> None:
    assert "persistent" in (HatchTileDataset.set_epoch.__doc__ or "").lower()
