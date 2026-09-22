from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hatchmatch.output import write_binary_png


def test_writer_emits_native_binary_l_png(tmp_path: Path) -> None:
    path = tmp_path / "mask.png"
    write_binary_png(
        np.array([[False, True], [True, False]]),
        path,
        (2, 2),
    )

    with Image.open(path) as image:
        assert image.mode == "L"
        assert image.size == (2, 2)
        assert set(np.asarray(image).reshape(-1).tolist()) == {0, 255}


def test_writer_preserves_numpy_height_width_order(tmp_path: Path) -> None:
    path = tmp_path / "mask.png"
    mask = np.array([[False, True, False], [True, False, True]])

    write_binary_png(mask, path, (3, 2))

    with Image.open(path) as image:
        assert image.size == (3, 2)
        np.testing.assert_array_equal(np.asarray(image), mask.astype(np.uint8) * 255)


@pytest.mark.parametrize(
    ("mask", "expected_size", "message"),
    [
        (np.zeros((2, 3), dtype=bool), (2, 3), "shape"),
        (np.zeros((2, 3, 1), dtype=bool), (3, 2), "two-dimensional"),
        (np.zeros((2, 3), dtype=np.uint8), (3, 2), "boolean"),
    ],
)
def test_writer_rejects_invalid_masks_without_creating_output(
    tmp_path: Path,
    mask: np.ndarray,
    expected_size: tuple[int, int],
    message: str,
) -> None:
    path = tmp_path / "mask.png"

    with pytest.raises(ValueError, match=message):
        write_binary_png(mask, path, expected_size)

    assert not path.exists()


def test_writer_creates_destination_parent_and_leaves_no_temporary_file(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "nested"
    path = output_dir / "mask.png"

    write_binary_png(np.zeros((2, 3), dtype=bool), path, (3, 2))

    assert sorted(item.name for item in output_dir.iterdir()) == ["mask.png"]
