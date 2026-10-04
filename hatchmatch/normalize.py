"""RGB tensors normalized with the pinned SegFormer processor statistics."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from transformers.models.segformer.image_processing_pil_segformer import (
    SegformerImageProcessorPil,
)


def segformer_image_tensor(gray: np.ndarray) -> Tensor:
    """Map one uint8 grayscale image to a normalized RGB CHW float tensor.

    Mean and standard deviation are the ImageNet statistics on
    ``SegformerImageProcessorPil``, the processor paired with the pinned
    SegFormer weights.
    """

    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("gray image must be a uint8 array of shape [H, W]")
    scaled = gray.astype(np.float32) / np.float32(255.0)
    mean = np.asarray(SegformerImageProcessorPil.image_mean, dtype=np.float32)
    std = np.asarray(SegformerImageProcessorPil.image_std, dtype=np.float32)
    channels = np.repeat(scaled[None], 3, axis=0)
    normalized = (channels - mean.reshape(3, 1, 1)) / std.reshape(3, 1, 1)
    return torch.from_numpy(np.ascontiguousarray(normalized))
