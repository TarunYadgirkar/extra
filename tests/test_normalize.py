"""Shared SegFormer ImageNet normalization for train, validation, and predict."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from hatchmatch.contracts import Box, Request
from hatchmatch.data import HatchTileDataset, TrainingExample, TrainingLabels
from hatchmatch.predict import predict_probability
from hatchmatch.train import _ValidationDataset, _validate, _validation_collate
from transformers.models.segformer.image_processing_pil_segformer import (
    SegformerImageProcessorPil,
)


class _RecordingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[torch.Tensor] = []

    def forward(
        self, image: torch.Tensor, query: torch.Tensor, texture: torch.Tensor
    ) -> torch.Tensor:
        self.images.append(image.detach().cpu())
        return torch.zeros(
            image.shape[0],
            1,
            image.shape[2],
            image.shape[3],
            dtype=image.dtype,
        )


def _mid_gray_example(tmp_path: Path) -> TrainingExample:
    drawing = np.full((8, 8), 128, dtype=np.uint8)
    positive = np.zeros((8, 8), dtype=np.uint8)
    positive[4, 4] = 255
    known = np.full((8, 8), 255, dtype=np.uint8)
    Image.fromarray(drawing).save(tmp_path / "drawing.png")
    Image.fromarray(positive).save(tmp_path / "positive.png")
    Image.fromarray(known).save(tmp_path / "known.png")
    labels = TrainingLabels(
        masks={
            "positive": tmp_path / "positive.png",
            "negative": None,
            "known": tmp_path / "known.png",
            "blank": None,
        },
        boxes={name: () for name in ("positive", "negative", "known", "blank")},
        explicit_known=True,
    )
    return TrainingExample(
        id="mid",
        document_id="doc-mid",
        kind="real",
        image=tmp_path / "drawing.png",
        width=8,
        height=8,
        query_box=Box(0, 0, 2, 2),
        context_boxes=(),
        labels=labels,
    )


def _expected_mid_gray() -> torch.Tensor:
    raw = np.float32(128.0 / 255.0)
    mean = np.asarray(SegformerImageProcessorPil.image_mean, dtype=np.float32)
    std = np.asarray(SegformerImageProcessorPil.image_std, dtype=np.float32)
    return torch.from_numpy((raw - mean) / std)


def _assert_mid_gray(pixel: torch.Tensor) -> None:
    expected = _expected_mid_gray()
    torch.testing.assert_close(pixel, expected, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(pixel, torch.full_like(pixel, 0.5))
    assert float(torch.max(torch.abs(pixel - 0.5))) > 0.05


def _prediction_config() -> dict[str, object]:
    return {
        "tile_size": 8,
        "overlap": 0,
        "query_size": 4,
        "batch_size": 1,
        "tta": ["identity"],
        "device": "cpu",
    }


def test_train_validation_and_predict_share_segformer_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    example = _mid_gray_example(tmp_path)
    dataset = HatchTileDataset(
        [example],
        tile_size=8,
        query_size=4,
        samples_per_epoch=10,
        seed=0,
        augment=False,
    )
    _assert_mid_gray(dataset[0]["image"][:, 0, 0])

    validation = _ValidationDataset([example], size=8, query_size=4)
    loader = DataLoader(validation, batch_size=1, collate_fn=_validation_collate)
    recorder = _RecordingModel()
    _validate(recorder, loader, torch.device("cpu"), amp=False, threshold=0.5)
    assert recorder.images
    _assert_mid_gray(recorder.images[0][0, :, 0, 0])

    request = Request(
        id="mid",
        image=tmp_path / "drawing.png",
        width=8,
        height=8,
        query_box=Box(0, 0, 2, 2),
        context_boxes=(),
    )
    predictor = _RecordingModel()
    predict_probability(request, [predictor], _prediction_config())
    assert predictor.images
    _assert_mid_gray(predictor.images[0][0, :, 0, 0])

    import importlib

    try:
        normalize = importlib.import_module("hatchmatch.normalize")
    except ImportError:
        normalize = None
    assert normalize is not None
    shared = getattr(normalize, "segformer_image_tensor", None)
    assert callable(shared)

    calls: list[str] = []

    def spy(gray: np.ndarray) -> torch.Tensor:
        calls.append("shared")
        return shared(gray)

    monkeypatch.setattr(normalize, "segformer_image_tensor", spy)
    dataset[0]
    _validate(
        _RecordingModel(),
        loader,
        torch.device("cpu"),
        amp=False,
        threshold=0.5,
    )
    predict_probability(request, [_RecordingModel()], _prediction_config())
    assert calls.count("shared") >= 3
