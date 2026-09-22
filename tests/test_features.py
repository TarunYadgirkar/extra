import numpy as np
import pytest

from hatchmatch.contracts import Box
from hatchmatch.features import texture_channels


def test_texture_channels_preserve_native_shape_and_have_finite_unit_range():
    gray = np.tile(np.arange(48, dtype=np.uint8), (32, 1))

    channels = texture_channels(gray, Box(4, 6, 20, 24))

    assert channels.shape[:2] == gray.shape
    assert channels.ndim == 3
    assert channels.shape[2] == 153
    assert channels.dtype == np.float32
    assert np.isfinite(channels).all()
    assert channels.min() >= 0.0
    assert channels.max() <= 1.0


def test_texture_channels_are_deterministic_for_constant_images():
    gray = np.full((24, 40), 127, dtype=np.uint8)
    query_box = Box(3, 4, 18, 20)

    first = texture_channels(gray, query_box)
    second = texture_channels(gray, query_box)

    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()


@pytest.mark.parametrize(
    ("gray", "box", "message"),
    [
        (np.zeros((8, 8, 1), np.uint8), Box(0, 0, 4, 4), "two-dimensional"),
        (np.zeros((8, 8), np.float32), Box(0, 0, 4, 4), "uint8"),
        (np.zeros((8, 8), np.uint8), Box(0, 0, 9, 4), "bounds"),
    ],
)
def test_texture_channels_reject_invalid_inputs(gray, box, message):
    with pytest.raises(ValueError, match=message):
        texture_channels(gray, box)
