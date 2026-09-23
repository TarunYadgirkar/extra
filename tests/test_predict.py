from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from hatchmatch.checkpoints import save_checkpoint
from hatchmatch.contracts import Box, Request
from hatchmatch.features import compact_texture_channels
from hatchmatch.model import QuerySegFormer
from hatchmatch.predict import blend_tiles, iter_tiles, predict_probability


def test_blended_constant_tiles_have_no_seams():
    tiles = list(iter_tiles(513, 769, size=256, overlap=64))
    values = [np.ones((tile.height, tile.width), np.float32) for tile in tiles]
    merged = blend_tiles((513, 769), tiles, values)
    np.testing.assert_allclose(merged, 1.0, atol=1e-6)


def test_iter_tiles_covers_odd_dimensions_and_images_smaller_than_a_tile():
    odd = list(iter_tiles(513, 769, size=256, overlap=64))
    covered = np.zeros((513, 769), dtype=np.int16)
    for tile in odd:
        assert tile.y >= 0 and tile.x >= 0
        assert tile.y + tile.height <= 513
        assert tile.x + tile.width <= 769
        assert 0 < tile.height <= 256
        assert 0 < tile.width <= 256
        covered[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width] += 1

    assert covered.min() >= 1
    assert len(odd) == 12

    small = list(iter_tiles(9, 7, size=32, overlap=8))
    assert [(tile.y, tile.x, tile.height, tile.width) for tile in small] == [
        (0, 0, 9, 7)
    ]
    merged = blend_tiles(
        (9, 7),
        small,
        [np.ones((small[0].height, small[0].width), np.float32)],
    )
    np.testing.assert_allclose(merged, 1.0, atol=1e-6)
    assert merged.dtype == np.float32
    assert merged.shape == (9, 7)


def test_overlapping_tiles_use_a_clipped_hann_blend():
    tiles = list(iter_tiles(80, 48, size=48, overlap=16))
    assert [(tile.y, tile.x, tile.height, tile.width) for tile in tiles] == [
        (0, 0, 48, 48),
        (32, 0, 48, 48),
    ]
    merged = blend_tiles(
        (80, 48),
        tiles,
        [
            np.zeros((48, 48), np.float32),
            np.ones((48, 48), np.float32),
        ],
    )

    np.testing.assert_allclose(merged[:32], 0.0, atol=1e-6)
    np.testing.assert_allclose(merged[48:], 1.0, atol=1e-6)
    overlap = merged[32:48, 24]
    assert np.all(np.diff(overlap) > 0)
    assert overlap[0] < 0.05
    assert overlap[-1] > 0.95
    assert float(overlap.min()) > 0.0


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)


def _write_request(
    tmp_path: Path,
    gray: np.ndarray,
    *,
    query_box: tuple[int, int, int, int] = (1, 1, 5, 5),
) -> Request:
    path = tmp_path / "drawing.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray, mode="L").save(path)
    height, width = gray.shape
    return Request(
        id="q1",
        image=path,
        width=width,
        height=height,
        query_box=Box(*query_box),
        context_boxes=(),
    )


def _pattern(height: int, width: int) -> np.ndarray:
    values = np.arange(height * width, dtype=np.uint16).reshape(height, width)
    return np.mod(values, 251).astype(np.uint8)


class _ContentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[np.ndarray] = []
        self.queries: list[np.ndarray] = []
        self.textures: list[np.ndarray] = []

    def forward(
        self, image: torch.Tensor, query: torch.Tensor, texture: torch.Tensor
    ) -> torch.Tensor:
        self.images.append(image.detach().cpu().numpy().copy())
        self.queries.append(query.detach().cpu().numpy().copy())
        self.textures.append(texture.detach().cpu().numpy().copy())
        return image[:, :1] * 50 - 25


class _ConstantLogitModel(nn.Module):
    def __init__(self, logit: float) -> None:
        super().__init__()
        self.logit = float(logit)

    def forward(
        self, image: torch.Tensor, query: torch.Tensor, texture: torch.Tensor
    ) -> torch.Tensor:
        return torch.full(
            (image.shape[0], 1, image.shape[2], image.shape[3]),
            self.logit,
            dtype=image.dtype,
        )


def _prediction_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "tile_size": 32,
        "overlap": 8,
        "query_size": 16,
        "batch_size": 2,
        "tta": ["identity"],
        "device": "cpu",
    }
    config.update(overrides)
    return config


def test_predict_probability_returns_native_float32_for_odd_and_small_images(
    tmp_path: Path,
):
    model = _ContentModel()
    for height, width in ((15, 17), (35, 41)):
        gray = _pattern(height, width)
        request = _write_request(tmp_path / f"{height}x{width}", gray)
        probability = predict_probability(
            request,
            [model],
            _prediction_config(tile_size=16, overlap=4, batch_size=3),
        )
        expected = _sigmoid(gray.astype(np.float32) / 255.0 * 50 - 25)
        assert probability.shape == (height, width)
        assert probability.dtype == np.float32
        np.testing.assert_allclose(probability, expected, atol=1e-5)

    repeated = predict_probability(
        request,
        [model],
        _prediction_config(tile_size=16, overlap=4, batch_size=3),
    )
    np.testing.assert_array_equal(probability, repeated)


def test_small_image_is_reflect_padded_and_uses_compact_texture(tmp_path: Path):
    gray = _pattern(8, 10)
    gray[0, 0] = 3
    gray[-1, 0] = 240
    request = _write_request(tmp_path, gray)
    model = _ContentModel()

    probability = predict_probability(
        request,
        [model],
        _prediction_config(tile_size=32, overlap=8),
    )

    expected = _sigmoid(gray.astype(np.float32) / 255.0 * 50 - 25)
    np.testing.assert_allclose(probability, expected, atol=1e-5)
    seen = model.images[0]
    assert seen.shape[-2:] == (32, 32)
    np.testing.assert_allclose(seen[0, 0, :8, :10], gray / 255.0, atol=1e-6)
    assert seen[0, 0, 8, 0] == pytest.approx(gray[-2, 0] / 255.0)
    padded = np.round(seen[0, 0] * 255.0).astype(np.uint8)
    np.testing.assert_allclose(
        model.textures[0][0],
        np.moveaxis(compact_texture_channels(padded), -1, 0),
        atol=1e-5,
    )
    assert not model.training


def test_tta_is_reversed_into_native_coordinates(tmp_path: Path):
    gray = _pattern(8, 10)
    gray[1:5, 1:5] = np.array(
        [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120], [130, 140, 150, 160]],
        dtype=np.uint8,
    )
    request = _write_request(tmp_path, gray)
    expected = _sigmoid(gray.astype(np.float32) / 255.0 * 50 - 25)
    transforms = {
        "horizontal": lambda array: np.flip(array, axis=-1),
        "vertical": lambda array: np.flip(array, axis=-2),
        "rot90": lambda array: np.rot90(array, k=-1, axes=(-2, -1)),
    }
    for name, invert in transforms.items():
        model = _ContentModel()
        probability = predict_probability(
            request,
            [model],
            _prediction_config(tta=[name], batch_size=1),
        )
        np.testing.assert_allclose(probability, expected, atol=1e-5)
        native = model.images[0][0, 0, : gray.shape[0], : gray.shape[1]]
        transformed = {
            "horizontal": np.fliplr(gray),
            "vertical": np.flipud(gray),
            "rot90": np.rot90(gray, k=1),
        }[name]
        observed = native[:, : transformed.shape[1]]
        if name == "rot90":
            observed = model.images[0][0, 0, : transformed.shape[0], : transformed.shape[1]]
        np.testing.assert_allclose(observed, transformed / 255.0, atol=1e-6)
        restored_query = invert(model.queries[0][0])
        identity = _ContentModel()
        predict_probability(request, [identity], _prediction_config())
        np.testing.assert_allclose(restored_query, identity.queries[0][0], atol=1e-6)
        assert not np.allclose(model.queries[0], identity.queries[0])


def test_fold_checkpoints_are_averaged_in_native_coordinates(tmp_path: Path):
    request = _write_request(tmp_path, _pattern(9, 13))
    probability = predict_probability(
        request,
        [_ConstantLogitModel(0.0), _ConstantLogitModel(4.0)],
        _prediction_config(tile_size=16, overlap=4),
    )
    expected = (_sigmoid(np.float32(0.0)) + _sigmoid(np.float32(4.0))) / np.float32(2.0)
    np.testing.assert_allclose(probability, expected, atol=1e-6)


def test_default_overlap_is_one_quarter_of_tile_size(tmp_path: Path):
    request = _write_request(tmp_path, _pattern(90, 70), query_box=(0, 0, 4, 4))
    counts: dict[str, int] = {}

    def runner_for(name: str):
        def runner(
            model: nn.Module,
            image: torch.Tensor,
            query: torch.Tensor,
            texture: torch.Tensor,
        ) -> torch.Tensor:
            counts[name] = counts.get(name, 0) + int(image.shape[0])
            return torch.zeros(
                image.shape[0], 1, image.shape[2], image.shape[3], dtype=image.dtype
            )

        return runner

    base = _prediction_config(tile_size=32, batch_size=5, tta=["identity"])
    base.pop("overlap")
    predict_probability(
        request, [_ConstantLogitModel(0.0)], base, batch_runner=runner_for("default")
    )
    predict_probability(
        request,
        [_ConstantLogitModel(0.0)],
        {**base, "overlap": 8},
        batch_runner=runner_for("quarter"),
    )
    predict_probability(
        request,
        [_ConstantLogitModel(0.0)],
        {**base, "overlap": 0},
        batch_runner=runner_for("zero"),
    )

    assert counts["default"] == counts["quarter"] == 12
    assert counts["zero"] == 9


def test_cuda_oom_retries_the_same_tiles_by_halving_until_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    request = _write_request(tmp_path, _pattern(90, 70), query_box=(0, 0, 4, 4))
    fingerprints: list[tuple[float, ...]] = []
    cache_calls = 0

    def empty_cache() -> None:
        nonlocal cache_calls
        cache_calls += 1

    def runner(
        model: nn.Module,
        image: torch.Tensor,
        query: torch.Tensor,
        texture: torch.Tensor,
    ) -> torch.Tensor:
        fingerprints.append(tuple(float(value) for value in image[:, 0, 0, 0]))
        if image.shape[0] > 1:
            raise torch.cuda.OutOfMemoryError("injected oom")
        return model(image, query, texture)

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    oom = predict_probability(
        request,
        [_ContentModel()],
        _prediction_config(tile_size=32, overlap=8, batch_size=8),
        batch_runner=runner,
    )
    direct = predict_probability(
        request,
        [_ContentModel()],
        _prediction_config(tile_size=32, overlap=8, batch_size=1),
    )

    assert [len(batch) for batch in fingerprints] == [8, 4, 2, *([1] * 12)]
    ordered = [batch[0] for batch in fingerprints if len(batch) == 1]
    assert list(fingerprints[0]) == ordered[:8]
    assert list(fingerprints[1]) == ordered[:4]
    assert list(fingerprints[2]) == ordered[:2]
    assert cache_calls == 3
    np.testing.assert_allclose(oom, direct, atol=1e-5)
    expected = _sigmoid(_pattern(90, 70).astype(np.float32) / 255.0 * 50 - 25)
    np.testing.assert_allclose(oom, expected, atol=1e-5)


def test_cuda_oom_at_batch_size_one_clears_unused_cache_then_reraises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    request = _write_request(tmp_path, _pattern(30, 30), query_box=(0, 0, 4, 4))
    attempts: list[int] = []
    cache_calls = 0

    def empty_cache() -> None:
        nonlocal cache_calls
        cache_calls += 1

    def runner(
        model: nn.Module,
        image: torch.Tensor,
        query: torch.Tensor,
        texture: torch.Tensor,
    ) -> torch.Tensor:
        attempts.append(int(image.shape[0]))
        raise torch.cuda.OutOfMemoryError("still oom")

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)

    with pytest.raises(torch.cuda.OutOfMemoryError, match="still oom"):
        predict_probability(
            request,
            [_ConstantLogitModel(0.0)],
            _prediction_config(tile_size=16, overlap=4, batch_size=6),
            batch_runner=runner,
        )

    assert attempts == [6, 3, 1]
    assert cache_calls == 2


def test_non_cuda_inference_errors_are_fatal(tmp_path: Path):
    request = _write_request(tmp_path, _pattern(40, 40))
    attempts: list[int] = []

    def runner(
        model: nn.Module,
        image: torch.Tensor,
        query: torch.Tensor,
        texture: torch.Tensor,
    ) -> torch.Tensor:
        attempts.append(int(image.shape[0]))
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        predict_probability(
            request,
            [_ConstantLogitModel(0.0)],
            _prediction_config(tile_size=16, overlap=4, batch_size=4),
            batch_runner=runner,
        )

    assert attempts == [4]


def test_verified_checkpoints_supply_averaged_fold_weights(tmp_path: Path):
    torch.manual_seed(7)
    first = QuerySegFormer(
        backbone="nvidia/mit-b0",
        pretrained=False,
        decoder_channels=8,
    ).eval()
    second = QuerySegFormer(
        backbone="nvidia/mit-b0",
        pretrained=False,
        decoder_channels=8,
    ).eval()
    architecture = {
        "backbone": "nvidia/mit-b0",
        "pretrained": True,
        "revision": "a" * 40,
        "decoder_channels": 8,
    }
    paths = []
    for index, model in enumerate((first, second)):
        path = tmp_path / f"fold-{index}.pt"
        save_checkpoint(
            path,
            {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "architecture": architecture,
            },
        )
        paths.append(path)

    request = _write_request(tmp_path, np.full((8, 9), 180, np.uint8))
    config = _prediction_config(tile_size=64, overlap=16, batch_size=1, query_size=96)
    from_modules = predict_probability(request, [first, second], config)
    from_checkpoints = predict_probability(request, paths, config)
    np.testing.assert_allclose(from_checkpoints, from_modules, atol=1e-5)
    assert from_checkpoints.shape == (8, 9)
    assert from_checkpoints.dtype == np.float32

    with pytest.raises(ValueError, match="SHA-256"):
        predict_probability(
            request,
            [{"path": paths[0], "sha256": "0" * 64}],
            config,
        )


def test_tile_batches_keep_one_spatial_shape(
    tmp_path: Path,
):
    request = _write_request(tmp_path, _pattern(35, 41))
    shapes: list[tuple[int, ...]] = []

    def runner(
        model: nn.Module,
        image: torch.Tensor,
        query: torch.Tensor,
        texture: torch.Tensor,
    ) -> torch.Tensor:
        shapes.append(tuple(image.shape))
        assert len({(image.shape[-2], image.shape[-1])}) == 1
        assert texture.shape[-2:] == image.shape[-2:]
        assert query.shape[0] == image.shape[0]
        return torch.zeros(
            image.shape[0], 1, image.shape[2], image.shape[3], dtype=image.dtype
        )

    predict_probability(
        request,
        [_ConstantLogitModel(0.0)],
        _prediction_config(tile_size=16, overlap=4, batch_size=3, tta=["rot90"]),
        batch_runner=runner,
    )

    assert shapes
    assert len({shape[-2:] for shape in shapes}) == 1
    assert all(shape[0] <= 3 for shape in shapes)
    assert sum(shape[0] for shape in shapes) > 3
