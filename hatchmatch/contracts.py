"""Strict, label-free input contracts for challenge inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

SCHEMA = "hatch-matching-challenge/v1"


@dataclass(frozen=True)
class Box:
    """A half-open rectangle in native image coordinates."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        coordinates = (self.x0, self.y0, self.x1, self.y1)
        if any(type(value) is not int for value in coordinates):
            raise ValueError("box coordinates must be integers")
        if self.x0 < 0 or self.y0 < 0:
            raise ValueError("box coordinates must be non-negative")
        if self.x0 >= self.x1 or self.y0 >= self.y1:
            raise ValueError("box must have positive half-open extents")


@dataclass(frozen=True)
class Request:
    """One validated, label-free inference request."""

    id: str
    document_id: str
    kind: str
    image: Path
    width: int
    height: int
    query_box: Box
    context_boxes: tuple[Box, ...]


def _contains_labels(value: Any) -> bool:
    if isinstance(value, dict):
        return "labels" in value or any(
            _contains_labels(child) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_labels(child) for child in value)
    return False


def _required(example: dict[str, Any], field: str, index: int) -> Any:
    if field not in example:
        raise ValueError(f"example {index} is missing required field {field!r}")
    return example[field]


def _text_field(example: dict[str, Any], field: str, index: int) -> str:
    value = _required(example, field, index)
    if not isinstance(value, str) or not value:
        raise ValueError(f"example {index} field {field!r} must be non-empty text")
    return value


def _positive_integer(
    example: dict[str, Any], field: str, index: int
) -> int:
    value = _required(example, field, index)
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"example {index} field {field!r} must be a positive integer"
        )
    return value


def _box(
    value: Any,
    *,
    width: int,
    height: int,
    field: str,
) -> Box:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{field} must contain four integer coordinates")
    try:
        box = Box(*value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not a valid half-open box: {exc}") from exc
    if box.x1 > width or box.y1 > height:
        raise ValueError(
            f"{field} exceeds native image bounds {width}x{height}"
        )
    return box


def _safe_image_path(image: str, data_root: Path, index: int) -> Path:
    candidate = (data_root / image).resolve()
    if not candidate.is_relative_to(data_root):
        raise ValueError(
            f"example {index} image path escapes data_root: {image!r}"
        )
    if not candidate.is_file():
        raise ValueError(f"example {index} image is missing: {image!r}")
    return candidate


def load_requests(
    path: str | Path, data_root: str | Path
) -> list[Request]:
    """Load and validate a label-free challenge request manifest."""

    manifest_path = Path(path)
    root = Path(data_root).resolve()
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)

    if _contains_labels(manifest):
        raise ValueError("labels are forbidden in inference inputs")
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    if manifest.get("schema") != SCHEMA:
        raise ValueError(
            f"unsupported manifest schema: {manifest.get('schema')!r}"
        )

    examples = manifest.get("examples")
    if not isinstance(examples, list):
        raise ValueError("manifest field 'examples' must be a list")

    requests: list[Request] = []
    seen_ids: set[str] = set()
    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError(f"example {index} must be a JSON object")

        request_id = _text_field(example, "id", index)
        if request_id in seen_ids:
            raise ValueError(f"duplicate request id: {request_id!r}")
        seen_ids.add(request_id)

        width = _positive_integer(example, "width", index)
        height = _positive_integer(example, "height", index)
        image_name = _text_field(example, "image", index)
        image_path = _safe_image_path(image_name, root, index)

        try:
            with Image.open(image_path) as image:
                actual_size = image.size
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(
                f"example {index} image is not readable: {image_name!r}"
            ) from exc
        if actual_size != (width, height):
            raise ValueError(
                f"example {index} image size {actual_size} does not match "
                f"declared size {(width, height)}"
            )

        query_box = _box(
            _required(example, "query_box", index),
            width=width,
            height=height,
            field=f"example {index} query_box",
        )
        raw_context_boxes = _required(example, "context_boxes", index)
        if not isinstance(raw_context_boxes, list):
            raise ValueError(f"example {index} context_boxes must be a list")
        context_boxes = tuple(
            _box(
                value,
                width=width,
                height=height,
                field=f"example {index} context_boxes[{box_index}]",
            )
            for box_index, value in enumerate(raw_context_boxes)
        )

        requests.append(
            Request(
                id=request_id,
                document_id=_text_field(example, "document_id", index),
                kind=_text_field(example, "kind", index),
                image=image_path,
                width=width,
                height=height,
                query_box=query_box,
                context_boxes=context_boxes,
            )
        )

    return requests
