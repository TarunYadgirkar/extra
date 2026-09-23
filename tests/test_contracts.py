import json
from pathlib import Path

import pytest
from PIL import Image

from hatchmatch.contracts import Box, Request, load_requests


SCHEMA = "hatch-matching-inputs/v1"


def _write_image(path: Path, size: tuple[int, int] = (4, 3)) -> None:
    Image.new("L", size, 255).save(path)


def _example(**overrides: object) -> dict[str, object]:
    example: dict[str, object] = {
        "id": "x",
        "image": "x.png",
        "width": 4,
        "height": 3,
        "query_box": [0, 0, 1, 1],
        "context_boxes": [],
    }
    example.update(overrides)
    return example


def _write_manifest(path: Path, examples: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"schema": SCHEMA, "examples": examples}))


def test_loader_rejects_labels(tmp_path: Path) -> None:
    manifest = {
        "schema": SCHEMA,
        "examples": [
            _example(labels={"positive_mask": "leak.png"}),
        ],
    }
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="labels"):
        load_requests(path, tmp_path)


@pytest.mark.parametrize(
    "field",
    ["expected_count", "positive_mask", "private_data"],
)
def test_loader_rejects_unexpected_top_level_fields(
    tmp_path: Path, field: str
) -> None:
    path = tmp_path / "inputs.json"
    path.write_text(
        json.dumps({"schema": SCHEMA, "examples": [], field: "forbidden"})
    )

    with pytest.raises(ValueError, match=rf"unexpected.*{field}"):
        load_requests(path, tmp_path)


@pytest.mark.parametrize(
    "field",
    ["expected_count", "positive_mask", "private_data"],
)
def test_loader_rejects_unexpected_example_fields(
    tmp_path: Path, field: str
) -> None:
    path = tmp_path / "inputs.json"
    _write_manifest(path, [_example(**{field: "forbidden"})])

    with pytest.raises(ValueError, match=rf"unexpected.*{field}"):
        load_requests(path, tmp_path)


def test_loader_returns_frozen_native_coordinate_contracts(tmp_path: Path) -> None:
    _write_image(tmp_path / "x.png")
    path = tmp_path / "inputs.json"
    _write_manifest(
        path,
        [_example(query_box=[0, 0, 2, 1], context_boxes=[[2, 1, 4, 3]])],
    )

    requests = load_requests(path, tmp_path)

    assert requests == [
        Request(
            id="x",
            image=(tmp_path / "x.png").resolve(),
            width=4,
            height=3,
            query_box=Box(0, 0, 2, 1),
            context_boxes=(Box(2, 1, 4, 3),),
        )
    ]
    with pytest.raises(AttributeError):
        requests[0].width = 10  # type: ignore[misc]


@pytest.mark.parametrize("schema", ["unknown/v1", None])
def test_loader_rejects_unknown_schema(tmp_path: Path, schema: object) -> None:
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps({"schema": schema, "examples": []}))

    with pytest.raises(ValueError, match="schema"):
        load_requests(path, tmp_path)


def test_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    _write_image(tmp_path / "x.png")
    path = tmp_path / "inputs.json"
    _write_manifest(path, [_example(), _example(image="other.png")])

    with pytest.raises(ValueError, match="duplicate.*x"):
        load_requests(path, tmp_path)


@pytest.mark.parametrize("identifier", ["../escape", "/absolute", "a/b", "bad id"])
def test_loader_rejects_unsafe_prediction_ids(
    tmp_path: Path, identifier: str
) -> None:
    path = tmp_path / "inputs.json"
    _write_manifest(path, [_example(id=identifier)])

    with pytest.raises(ValueError, match="id"):
        load_requests(path, tmp_path)


@pytest.mark.parametrize("image", ["../outside.png", "/etc/passwd"])
def test_loader_rejects_paths_outside_data_root(
    tmp_path: Path, image: str
) -> None:
    path = tmp_path / "inputs.json"
    _write_manifest(path, [_example(image=image)])

    with pytest.raises(ValueError, match="data_root"):
        load_requests(path, tmp_path)


def test_loader_rejects_symlink_escaping_data_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.png"
    _write_image(outside)
    (tmp_path / "escape.png").symlink_to(outside)
    path = tmp_path / "inputs.json"
    _write_manifest(path, [_example(image="escape.png")])

    try:
        with pytest.raises(ValueError, match="data_root"):
            load_requests(path, tmp_path)
    finally:
        outside.unlink()


@pytest.mark.parametrize(
    "box",
    [
        [-1, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 1, 1, 1],
        [0, 0, 5, 1],
        [0, 0, 1, 4],
        [0, 0, 1],
        [0, 0, 1, 1.5],
    ],
)
def test_loader_rejects_invalid_half_open_boxes(
    tmp_path: Path, box: list[object]
) -> None:
    _write_image(tmp_path / "x.png")
    path = tmp_path / "inputs.json"
    _write_manifest(path, [_example(query_box=box)])

    with pytest.raises(ValueError, match="query_box"):
        load_requests(path, tmp_path)


def test_loader_rejects_invalid_context_box(tmp_path: Path) -> None:
    _write_image(tmp_path / "x.png")
    path = tmp_path / "inputs.json"
    _write_manifest(path, [_example(context_boxes=[[0, 0, 5, 1]])])

    with pytest.raises(ValueError, match="context_boxes"):
        load_requests(path, tmp_path)


def test_loader_rejects_image_size_mismatch(tmp_path: Path) -> None:
    _write_image(tmp_path / "x.png", (5, 3))
    path = tmp_path / "inputs.json"
    _write_manifest(path, [_example()])

    with pytest.raises(ValueError, match="size"):
        load_requests(path, tmp_path)


def test_loader_rejects_missing_image(tmp_path: Path) -> None:
    path = tmp_path / "inputs.json"
    _write_manifest(path, [_example()])

    with pytest.raises(ValueError, match="missing"):
        load_requests(path, tmp_path)
