"""Training-only labels, grouped folds, and deterministic tile sampling."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

import hatchmatch.normalize as image_norm
from hatchmatch.augment import augment_training_sample
from hatchmatch.contracts import Box
from hatchmatch.features import compact_texture_channels

TRAINING_SCHEMA = "hatch-matching-challenge/v1"
FOLD_SCHEMA = "hatchmatch-group-folds/v1"
KINDS = frozenset({"real", "generated_cad"})
LABEL_REGIONS = ("positive", "negative", "known", "blank")
LABEL_KEYS = frozenset(
    f"{region}_{suffix}"
    for region in LABEL_REGIONS
    for suffix in ("mask", "boxes")
)
SAMPLING_SCHEDULE = (
    "positive",
    "positive",
    "positive",
    "positive",
    "hard_negative",
    "hard_negative",
    "hard_negative",
    "blank",
    "blank",
    "random",
)
BLANK_NEIGHBORHOOD_SIZE = 9
BLANK_MAX_INK_AVERAGE = 5
HARD_NEGATIVE_MIN_INK_AVERAGE = 20


@dataclass(frozen=True)
class Fold:
    """Training and validation row indices for one grouped split."""

    train: tuple[int, ...]
    valid: tuple[int, ...]


@dataclass(frozen=True)
class TrainingLabels:
    """Resolved training-only annotation sources."""

    masks: Mapping[str, Path | None]
    boxes: Mapping[str, tuple[Box, ...]]
    explicit_known: bool = False
    explicit_blank: bool = False


@dataclass(frozen=True)
class TrainingExample:
    """One validated labeled example; unavailable to inference contracts."""

    id: str
    document_id: str
    kind: str
    image: Path
    width: int
    height: int
    query_box: Box
    context_boxes: tuple[Box, ...]
    labels: TrainingLabels


@dataclass(frozen=True)
class _ExampleArrays:
    image: np.ndarray
    query: np.ndarray
    target: np.ndarray
    known: np.ndarray
    blank: np.ndarray
    explicit_blank: bool


def _mapping_value(example: object, field: str, index: int) -> object:
    if isinstance(example, Mapping):
        if field not in example:
            raise ValueError(f"example {index} is missing {field!r}")
        return example[field]
    if not hasattr(example, field):
        raise ValueError(f"example {index} is missing {field!r}")
    return getattr(example, field)


def make_group_folds(
    examples: Sequence[object],
    count: int,
    seed: int,
) -> list[Fold]:
    """Split rows with kind stratification while keeping documents intact."""

    if type(count) is not int or count < 2:
        raise ValueError("count must be an integer of at least two")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if not examples:
        raise ValueError("examples must not be empty")

    groups: list[str] = []
    kinds: list[str] = []
    group_kinds: dict[str, str] = {}
    for index, example in enumerate(examples):
        document_id = _mapping_value(example, "document_id", index)
        if not isinstance(document_id, str) or not document_id:
            raise ValueError(f"example {index} has an invalid document_id")
        if isinstance(example, Mapping):
            kind = example.get("kind", "real")
        else:
            kind = getattr(example, "kind", "real")
        if kind not in KINDS:
            raise ValueError(f"example {index} has an unsupported kind: {kind!r}")
        previous = group_kinds.setdefault(document_id, kind)
        if previous != kind:
            raise ValueError(
                f"document group {document_id!r} contains multiple kinds"
            )
        groups.append(document_id)
        kinds.append(kind)

    unique_groups = set(groups)
    if len(unique_groups) < count:
        raise ValueError(
            f"count={count} requires at least {count} document groups; "
            f"found {len(unique_groups)}"
        )

    splitter = StratifiedGroupKFold(
        n_splits=count,
        shuffle=True,
        random_state=int(seed),
    )
    rows = np.zeros((len(examples), 1), dtype=np.uint8)
    folds = [
        Fold(
            train=tuple(sorted(int(index) for index in train)),
            valid=tuple(sorted(int(index) for index in valid)),
        )
        for train, valid in splitter.split(rows, np.asarray(kinds), groups)
    ]

    seen_valid: list[int] = []
    for fold in folds:
        train_groups = {groups[index] for index in fold.train}
        valid_groups = {groups[index] for index in fold.valid}
        if train_groups & valid_groups:
            raise RuntimeError("grouped fold construction split a document")
        seen_valid.extend(fold.valid)
    if sorted(seen_valid) != list(range(len(examples))):
        raise RuntimeError("grouped folds do not cover each row exactly once")
    return folds


def _safe_path(value: object, root: Path, location: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a non-empty relative path")
    candidate = (root / value).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"{location} escapes data_root")
    if not candidate.is_file():
        raise ValueError(f"{location} is missing: {value!r}")
    return candidate


def _positive_integer(value: object, location: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return value


def _box(
    value: object,
    *,
    width: int,
    height: int,
    location: str,
) -> Box:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{location} must contain four coordinates")
    try:
        box = Box(*value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} is invalid: {exc}") from exc
    if box.x1 > width or box.y1 > height:
        raise ValueError(f"{location} exceeds native image bounds")
    return box


def _boxes(
    value: object,
    *,
    width: int,
    height: int,
    location: str,
) -> tuple[Box, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a list")
    return tuple(
        _box(
            raw,
            width=width,
            height=height,
            location=f"{location}[{index}]",
        )
        for index, raw in enumerate(value)
    )


def load_training_examples(
    path: str | Path,
    data_root: str | Path,
) -> list[TrainingExample]:
    """Load labeled examples through the training-only module boundary."""

    manifest_path = Path(path)
    root = Path(data_root).resolve()
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or manifest.get("schema") != TRAINING_SCHEMA:
        raise ValueError("unsupported training manifest schema")
    if set(manifest) != {"schema", "examples"}:
        raise ValueError("training manifest contains unexpected fields")
    raw_examples = manifest["examples"]
    if not isinstance(raw_examples, list):
        raise ValueError("training manifest examples must be a list")

    examples: list[TrainingExample] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_examples):
        if not isinstance(raw, dict):
            raise ValueError(f"example {index} must be an object")
        required = {
            "id",
            "document_id",
            "kind",
            "image",
            "width",
            "height",
            "query_box",
            "context_boxes",
            "labels",
        }
        if set(raw) != required:
            missing = sorted(required - set(raw))
            unexpected = sorted(set(raw) - required)
            raise ValueError(
                f"example {index} fields differ: "
                f"missing={missing}, unexpected={unexpected}"
            )
        identifier = raw["id"]
        document_id = raw["document_id"]
        kind = raw["kind"]
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"example {index} id must be non-empty text")
        if identifier in seen_ids:
            raise ValueError(f"duplicate training example id: {identifier!r}")
        seen_ids.add(identifier)
        if not isinstance(document_id, str) or not document_id:
            raise ValueError(f"example {index} document_id must be non-empty text")
        if kind not in KINDS:
            raise ValueError(f"example {index} has unsupported kind: {kind!r}")
        width = _positive_integer(raw["width"], f"example {index} width")
        height = _positive_integer(raw["height"], f"example {index} height")
        image_path = _safe_path(raw["image"], root, f"example {index} image")
        try:
            with Image.open(image_path) as image:
                actual_size = image.size
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"example {index} image is not readable") from exc
        if actual_size != (width, height):
            raise ValueError(
                f"example {index} image size {actual_size} does not match "
                f"declared size {(width, height)}"
            )
        query_box = _box(
            raw["query_box"],
            width=width,
            height=height,
            location=f"example {index} query_box",
        )
        context_boxes = _boxes(
            raw["context_boxes"],
            width=width,
            height=height,
            location=f"example {index} context_boxes",
        )

        raw_labels = raw["labels"]
        if not isinstance(raw_labels, dict):
            raise ValueError(f"example {index} labels must be an object")
        unexpected_labels = sorted(set(raw_labels) - LABEL_KEYS)
        if unexpected_labels:
            raise ValueError(
                f"example {index} labels contain unsupported fields: "
                f"{unexpected_labels}"
            )
        masks: dict[str, Path | None] = {}
        label_boxes: dict[str, tuple[Box, ...]] = {}
        for region in LABEL_REGIONS:
            mask_key = f"{region}_mask"
            boxes_key = f"{region}_boxes"
            masks[region] = (
                _safe_path(
                    raw_labels[mask_key],
                    root,
                    f"example {index} labels.{mask_key}",
                )
                if mask_key in raw_labels
                else None
            )
            label_boxes[region] = (
                _boxes(
                    raw_labels[boxes_key],
                    width=width,
                    height=height,
                    location=f"example {index} labels.{boxes_key}",
                )
                if boxes_key in raw_labels
                else ()
            )
        if not raw_labels:
            raise ValueError(f"example {index} has no training labels")

        examples.append(
            TrainingExample(
                id=identifier,
                document_id=document_id,
                kind=kind,
                image=image_path,
                width=width,
                height=height,
                query_box=query_box,
                context_boxes=context_boxes,
                labels=TrainingLabels(
                    masks=masks,
                    boxes=label_boxes,
                    explicit_known=(
                        "known_mask" in raw_labels
                        or "known_boxes" in raw_labels
                    ),
                    explicit_blank=(
                        "blank_mask" in raw_labels
                        or "blank_boxes" in raw_labels
                    ),
                ),
            )
        )
    return examples


def _read_binary_mask(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    try:
        with Image.open(path) as image:
            if image.mode not in {"1", "L"}:
                raise ValueError(f"mask {path} must use mode 1 or L")
            if image.size != expected_size:
                raise ValueError(
                    f"mask {path} size {image.size} does not match {expected_size}"
                )
            values = np.asarray(image.convert("L"), dtype=np.uint8)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"mask {path} is not readable") from exc
    unique = set(int(value) for value in np.unique(values))
    if not (unique <= {0, 1} or unique <= {0, 255}):
        raise ValueError(f"mask {path} is not binary")
    return values != 0


def _paint_boxes(mask: np.ndarray, boxes: Sequence[Box]) -> None:
    for box in boxes:
        mask[box.y0 : box.y1, box.x0 : box.x1] = True


def _load_region(example: TrainingExample, region: str) -> np.ndarray:
    result = np.zeros((example.height, example.width), dtype=bool)
    path = example.labels.masks[region]
    if path is not None:
        result |= _read_binary_mask(path, (example.width, example.height))
    _paint_boxes(result, example.labels.boxes[region])
    return result


def _exclude_support(mask: np.ndarray, example: TrainingExample) -> None:
    for box in (example.query_box, *example.context_boxes):
        mask[box.y0 : box.y1, box.x0 : box.x1] = False


def _letterbox(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, min(size, int(round(width * scale))))
    resized_height = max(1, min(size, int(round(height * scale))))
    resized = np.asarray(
        Image.fromarray(image).resize(
            (resized_width, resized_height),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.uint8,
    )
    output = np.full((size, size), 255, dtype=np.uint8)
    x0 = (size - resized_width) // 2
    y0 = (size - resized_height) // 2
    output[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return output


def _crop_with_padding(
    array: np.ndarray,
    *,
    center: tuple[int, int],
    size: int,
    fill: int | bool,
) -> np.ndarray:
    height, width = array.shape[:2]
    center_y, center_x = center
    y0 = center_y - size // 2
    x0 = center_x - size // 2
    y1, x1 = y0 + size, x0 + size
    source_y0, source_y1 = max(y0, 0), min(y1, height)
    source_x0, source_x1 = max(x0, 0), min(x1, width)
    destination_y0 = source_y0 - y0
    destination_x0 = source_x0 - x0
    output_shape = (size, size, *array.shape[2:])
    output = np.full(output_shape, fill, dtype=array.dtype)
    output[
        destination_y0 : destination_y0 + source_y1 - source_y0,
        destination_x0 : destination_x0 + source_x1 - source_x0,
    ] = array[source_y0:source_y1, source_x0:source_x1]
    return output


def _sample_mask(
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    height, width = mask.shape
    for _ in range(2048):
        y = int(rng.integers(0, height))
        x = int(rng.integers(0, width))
        if mask[y, x]:
            return y, x
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    y = int(rows[int(rng.integers(0, rows.size))])
    columns = np.flatnonzero(mask[y])
    x = int(columns[int(rng.integers(0, columns.size))])
    return y, x


def _neighborhood_ink_average(gray: np.ndarray) -> np.ndarray:
    ink = np.where(gray < 245, 255, 0).astype(np.uint8)
    return cv2.boxFilter(
        ink,
        cv2.CV_8U,
        (BLANK_NEIGHBORHOOD_SIZE, BLANK_NEIGHBORHOOD_SIZE),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )


class HatchTileDataset(Dataset[dict[str, torch.Tensor]]):
    """Sample semantically realized tiles in deterministic 4:3:2:1 blocks.

    Every complete aligned block of ten indices contains four positive, three
    hard-negative, two blank, and one random-known request in a seeded order.
    A partial final block is the deterministic prefix of its seeded order. If
    the block's initial example lacks the requested domain, examples are
    searched deterministically; absence from the entire dataset is an error.

    ``cache_size`` bounds retained full-resolution examples per dataset
    instance and therefore per worker. It defaults to one to avoid retaining
    multiple large drawings, and callers may set it to zero.
    """

    def __init__(
        self,
        examples: Sequence[TrainingExample],
        *,
        tile_size: int = 256,
        query_size: int = 96,
        samples_per_epoch: int | None = None,
        seed: int = 0,
        augment: bool = True,
        cache_size: int = 1,
    ) -> None:
        if not examples or not all(
            isinstance(example, TrainingExample) for example in examples
        ):
            raise ValueError("examples must contain TrainingExample values")
        if type(tile_size) is not int or tile_size <= 0:
            raise ValueError("tile_size must be a positive integer")
        if type(query_size) is not int or query_size <= 0:
            raise ValueError("query_size must be a positive integer")
        if samples_per_epoch is None:
            samples_per_epoch = len(examples) * len(SAMPLING_SCHEDULE)
        if type(samples_per_epoch) is not int or samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be a positive integer")
        if not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        if type(augment) is not bool:
            raise ValueError("augment must be Boolean")
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")

        self.examples = tuple(examples)
        self.tile_size = tile_size
        self.query_size = query_size
        self.samples_per_epoch = samples_per_epoch
        self.seed = int(seed)
        self.augment = augment
        self.cache_size = cache_size
        self.epoch = 0
        self._cache: OrderedDict[int, _ExampleArrays] = OrderedDict()

    @classmethod
    def from_manifest(
        cls,
        path: str | Path,
        data_root: str | Path,
        **kwargs: Any,
    ) -> HatchTileDataset:
        return cls(load_training_examples(path, data_root), **kwargs)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        """Set deterministic epoch state before creating worker iterators.

        Persistent worker processes retain their copied dataset state. Callers
        using them must recreate the worker iterator after ``set_epoch``;
        non-persistent workers inherit the new epoch normally.
        """

        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def sample_kind(self, index: int) -> str:
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        block, offset = divmod(index, len(SAMPLING_SCHEDULE))
        schedule = list(SAMPLING_SCHEDULE)
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, block, 17])
        )
        rng.shuffle(schedule)
        return schedule[offset]

    def _load_arrays(self, index: int) -> _ExampleArrays:
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached

        example = self.examples[index]
        with Image.open(example.image) as image:
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
        positive = _load_region(example, "positive")
        negative = _load_region(example, "negative")
        blank = _load_region(example, "blank")
        if np.any(positive & (negative | blank)):
            raise ValueError(
                f"{example.id}: positive and negative/blank labels overlap"
            )
        if example.labels.explicit_known:
            known = _load_region(example, "known")
            labeled = positive.copy()
            labeled |= negative
            labeled |= blank
            if np.any(labeled & ~known):
                raise ValueError(
                    f"{example.id}: label lies outside the explicit known domain"
                )
        else:
            known = positive.copy()
            known |= negative
            known |= blank

        _exclude_support(known, example)
        _exclude_support(positive, example)
        _exclude_support(blank, example)
        positive &= known
        blank &= known
        if not known.any():
            raise ValueError(
                f"example {example.id!r} has no known pixels outside support"
            )
        query = gray[
            example.query_box.y0 : example.query_box.y1,
            example.query_box.x0 : example.query_box.x1,
        ].copy()
        arrays = _ExampleArrays(
            image=gray,
            query=query,
            target=positive,
            known=known,
            blank=blank,
            explicit_blank=example.labels.explicit_blank,
        )
        if self.cache_size:
            self._cache[index] = arrays
            self._cache.move_to_end(index)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return arrays

    def _rng(self, index: int, stream: int) -> np.random.Generator:
        sequence = np.random.SeedSequence(
            [self.seed, self.epoch, int(index), stream]
        )
        return np.random.default_rng(sequence)

    def _candidate_mask(
        self,
        arrays: _ExampleArrays,
        kind: str,
    ) -> np.ndarray:
        negative = arrays.known & ~arrays.target
        if kind == "positive":
            return arrays.target
        if kind == "random":
            return arrays.known

        ink_average = _neighborhood_ink_average(arrays.image)
        if kind == "hard_negative":
            return (
                negative
                & ~arrays.blank
                & (ink_average >= HARD_NEGATIVE_MIN_INK_AVERAGE)
            )
        if kind == "blank":
            if arrays.explicit_blank:
                return arrays.blank
            return negative & (ink_average <= BLANK_MAX_INK_AVERAGE)
        raise ValueError(f"unsupported sampling kind: {kind!r}")

    def _sample_location(
        self,
        index: int,
        kind: str,
    ) -> tuple[_ExampleArrays, tuple[int, int]]:
        block = index // len(SAMPLING_SCHEDULE)
        start_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, block, 29])
        )
        start = int(start_rng.integers(0, len(self.examples)))
        center_rng = self._rng(index, 0)
        for offset in range(len(self.examples)):
            example_index = (start + offset) % len(self.examples)
            arrays = self._load_arrays(example_index)
            center = _sample_mask(
                self._candidate_mask(arrays, kind),
                center_rng,
            )
            if center is not None:
                return arrays, center
        raise ValueError(
            f"no training example has {kind} sampling candidates"
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if type(index) is not int:
            raise TypeError("dataset index must be an integer")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)

        kind = self.sample_kind(index)
        arrays, center = self._sample_location(index, kind)
        image = _crop_with_padding(
            arrays.image,
            center=center,
            size=self.tile_size,
            fill=255,
        )
        target = _crop_with_padding(
            arrays.target,
            center=center,
            size=self.tile_size,
            fill=False,
        )
        known = _crop_with_padding(
            arrays.known,
            center=center,
            size=self.tile_size,
            fill=False,
        )
        query = _letterbox(arrays.query, self.query_size)
        if self.augment:
            augmented = augment_training_sample(
                image,
                query,
                target,
                known,
                seed=int(self._rng(index, 1).integers(0, 2**32, dtype=np.uint64)),
            )
            image = augmented["image"]
            query = augmented["query"]
            target = augmented["target"]
            known = augmented["known"]
        else:
            image = image.copy()
            query = query.copy()
            target = target.copy()
            known = known.copy()

        image_tensor = image_norm.segformer_image_tensor(image)
        query_tensor = image_norm.segformer_image_tensor(query)
        texture_tensor = torch.from_numpy(
            np.moveaxis(compact_texture_channels(image), -1, 0).copy()
        )
        target_tensor = torch.from_numpy(target[None].astype(np.float32))
        known_tensor = torch.from_numpy(known[None].astype(np.float32))
        return {
            "image": image_tensor,
            "query": query_tensor,
            "texture": texture_tensor,
            "target": target_tensor,
            "known": known_tensor,
        }


def _manifest_payload(
    examples: Sequence[TrainingExample],
    folds: Sequence[Fold],
    *,
    source: Path,
    seed: int,
) -> dict[str, object]:
    payload_folds: list[dict[str, object]] = []
    for index, fold in enumerate(folds):
        train_documents = sorted(
            {examples[row].document_id for row in fold.train}
        )
        valid_documents = sorted(
            {examples[row].document_id for row in fold.valid}
        )
        payload_folds.append(
            {
                "fold": index,
                "train_indices": list(fold.train),
                "valid_indices": list(fold.valid),
                "train_ids": [examples[row].id for row in fold.train],
                "valid_ids": [examples[row].id for row in fold.valid],
                "train_documents": train_documents,
                "valid_documents": valid_documents,
                "valid_kind_counts": {
                    kind: sum(examples[row].kind == kind for row in fold.valid)
                    for kind in sorted(KINDS)
                },
            }
        )
    return {
        "schema": FOLD_SCHEMA,
        "source_manifest": str(source.resolve()),
        "seed": seed,
        "fold_count": len(folds),
        "example_count": len(examples),
        "document_count": len({example.document_id for example in examples}),
        "folds": payload_folds,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create document-grouped training fold manifests."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--folds", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260922)
    arguments = parser.parse_args(argv)

    examples = load_training_examples(
        arguments.manifest,
        arguments.manifest.parent,
    )
    folds = make_group_folds(examples, arguments.folds, arguments.seed)
    payload = _manifest_payload(
        examples,
        folds,
        source=arguments.manifest,
        seed=arguments.seed,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(folds)} folds for {len(examples)} examples and "
        f"{payload['document_count']} document groups to {arguments.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
