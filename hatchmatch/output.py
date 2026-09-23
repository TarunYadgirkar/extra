"""Atomic native-resolution binary PNG output."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def _validated_size(expected_size: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(expected_size, tuple)
        or len(expected_size) != 2
        or any(type(value) is not int or value <= 0 for value in expected_size)
    ):
        raise ValueError("expected_size must be a positive (width, height) tuple")
    return expected_size


def write_binary_png(
    mask: np.ndarray,
    path: str | Path,
    expected_size: tuple[int, int],
) -> None:
    """Atomically write a Boolean ``(height, width)`` mask as a binary PNG."""

    width, height = _validated_size(expected_size)
    if not isinstance(mask, np.ndarray):
        raise ValueError("mask must be a NumPy array")
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if mask.dtype != np.bool_:
        raise ValueError("mask must have boolean dtype")
    if mask.shape != (height, width):
        raise ValueError(
            f"mask shape {mask.shape} does not match expected "
            f"(height, width) {(height, width)}"
        )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".png",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)

        pixels = mask.astype(np.uint8) * np.uint8(255)
        Image.fromarray(pixels).save(temporary_path, format="PNG")

        with Image.open(temporary_path) as written:
            written.load()
            values = set(np.asarray(written).reshape(-1).tolist())
            if written.mode != "L":
                raise RuntimeError(
                    f"temporary output has mode {written.mode!r}, expected 'L'"
                )
            if written.size != (width, height):
                raise RuntimeError(
                    f"temporary output has size {written.size}, "
                    f"expected {(width, height)}"
                )
            if not values.issubset({0, 255}):
                raise RuntimeError(
                    f"temporary output contains non-binary values: {values}"
                )

        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
