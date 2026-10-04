"""Query-conditioned classical texture baseline."""

from __future__ import annotations

import numpy as np

from hatchmatch.contracts import Box
from hatchmatch.features import _iter_texture_channels


def baseline_probability(gray: np.ndarray, query_box: Box) -> np.ndarray:
    """Score native-resolution pixels by robust texture similarity to a query."""

    distance_sum = np.zeros(gray.shape, dtype=np.float32)
    channel_count = 0
    query_slice = np.s_[query_box.y0 : query_box.y1, query_box.x0 : query_box.x1]

    for channel in _iter_texture_channels(gray, query_box):
        query_values = channel[query_slice]
        median = np.median(query_values)
        mad = np.median(np.abs(query_values - median))
        robust_scale = max(float(mad) * 1.4826, 0.025)
        standardized = np.minimum(
            np.abs(channel - median) / np.float32(robust_scale),
            np.float32(6.0),
        )
        distance_sum += standardized * standardized
        channel_count += 1

    distance = np.sqrt(distance_sum / np.float32(channel_count))
    probability = np.exp(np.float32(-1.5) * distance)
    return np.clip(probability, 0.0, 1.0).astype(np.float32, copy=False)
