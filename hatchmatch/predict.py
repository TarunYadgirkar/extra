"""Native-resolution tiled inference with reversible test-time augmentation."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

import hatchmatch.normalize as image_norm
from hatchmatch.checkpoints import load_verified_checkpoint
from hatchmatch.contracts import Request
from hatchmatch.features import compact_texture_channels
from hatchmatch.model import QuerySegFormer

BatchRunner = Callable[[nn.Module, Tensor, Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class Tile:
    """One native-coordinate window of a drawing."""

    y: int
    x: int
    height: int
    width: int
    size: int
    overlap: int


def iter_tiles(
    height: int, width: int, size: int, overlap: int
) -> Iterator[Tile]:
    """Yield overlapping native windows, including images smaller than a tile."""

    if type(height) is not int or type(width) is not int or height <= 0 or width <= 0:
        raise ValueError("height and width must be positive integers")
    if type(size) is not int or size <= 0:
        raise ValueError("size must be a positive integer")
    if type(overlap) is not int or not 0 <= overlap < size:
        raise ValueError("overlap must be an integer in [0, size)")
    stride = size - overlap
    for y in _axis_origins(height, size, stride):
        tile_height = min(size, height - y)
        for x in _axis_origins(width, size, stride):
            yield Tile(
                y=y,
                x=x,
                height=tile_height,
                width=min(size, width - x),
                size=size,
                overlap=overlap,
            )


def blend_tiles(
    shape: tuple[int, int],
    tiles: Sequence[Tile],
    values: Sequence[np.ndarray],
) -> np.ndarray:
    """Blend native tile values with a normalized clipped Hann window."""

    height, width = shape
    if len(tiles) != len(values):
        raise ValueError("tiles and values must have the same length")
    accumulator = np.zeros((height, width), dtype=np.float64)
    weight = np.zeros((height, width), dtype=np.float64)
    for tile, value in zip(tiles, values, strict=True):
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (tile.height, tile.width):
            raise ValueError("tile value shape must match the native tile")
        window = _tile_window(tile)
        y1 = tile.y + tile.height
        x1 = tile.x + tile.width
        accumulator[tile.y:y1, tile.x:x1] += array * window
        weight[tile.y:y1, tile.x:x1] += window
    if np.any(weight <= 0):
        raise ValueError("tiles do not cover the output")
    return (accumulator / weight).astype(np.float32)


def predict_probability(
    request: Request,
    models: Sequence[nn.Module | str | Path | Mapping[str, object]],
    config: Mapping[str, object],
    *,
    batch_runner: BatchRunner | None = None,
) -> np.ndarray:
    """Return averaged float32 probabilities with shape ``(height, width)``."""

    if not isinstance(request, Request):
        raise TypeError("request must be a label-free Request")
    if not models:
        raise ValueError("at least one model is required")
    gray = _load_gray(request)
    query = gray[
        request.query_box.y0 : request.query_box.y1,
        request.query_box.x0 : request.query_box.x1,
    ]
    tile_size = int(config.get("tile_size", 256))
    overlap = int(config["overlap"]) if "overlap" in config else tile_size // 4
    query_size = int(config.get("query_size", 96))
    batch_size = int(config.get("batch_size", 4))
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    transforms = tuple(config.get("tta", ("identity", "horizontal", "vertical", "rot90")))
    if not transforms:
        raise ValueError("at least one TTA view is required")
    device = torch.device(str(config.get("device", "cpu")))
    prepared = [
        _prepare_model(_resolve_model(model, device), device) for model in models
    ]
    letterboxed_query = _letterbox(query, query_size)

    accumulator = np.zeros(gray.shape, dtype=np.float64)
    for name in transforms:
        transformed = _apply_transform(gray, str(name))
        query_tensor = _image_tensor(_apply_transform(letterboxed_query, str(name)))
        view_maps = [
            _predict_view(
                transformed,
                query_tensor,
                model,
                tile_size=tile_size,
                overlap=overlap,
                batch_size=batch_size,
                device=device,
                batch_runner=batch_runner,
            )
            for model in prepared
        ]
        averaged = np.mean(np.stack(view_maps, axis=0), axis=0)
        accumulator += _invert_transform(averaged, str(name))
    return (accumulator / len(transforms)).astype(np.float32)


def _axis_origins(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    count = math.ceil((length - size) / stride) + 1
    return [index * stride for index in range(count)]


def _clipped_hann(length: int, overlap: int) -> np.ndarray:
    window = np.ones(length, dtype=np.float64)
    if overlap <= 0 or length <= 1:
        return window
    ramp_width = min(overlap, length)
    positions = (np.arange(ramp_width, dtype=np.float64) + 0.5) / ramp_width
    ramp = np.clip(0.5 * (1.0 - np.cos(np.pi * positions)), 1e-3, None)
    window[:ramp_width] = np.minimum(window[:ramp_width], ramp)
    window[-ramp_width:] = np.minimum(window[-ramp_width:], ramp[::-1])
    return window


def _tile_window(tile: Tile) -> np.ndarray:
    vertical = _clipped_hann(tile.size, tile.overlap)[: tile.height]
    horizontal = _clipped_hann(tile.size, tile.overlap)[: tile.width]
    return vertical[:, None] * horizontal[None, :]


def _load_gray(request: Request) -> np.ndarray:
    with Image.open(request.image) as image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if gray.shape != (request.height, request.width):
        raise ValueError(
            f"image size {gray.shape[::-1]} does not match "
            f"declared size {(request.width, request.height)}"
        )
    return gray


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


def _apply_transform(image: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return np.ascontiguousarray(image)
    if name == "horizontal":
        return np.ascontiguousarray(np.flip(image, axis=-1))
    if name == "vertical":
        return np.ascontiguousarray(np.flip(image, axis=-2))
    if name == "rot90":
        return np.ascontiguousarray(np.rot90(image, k=1, axes=(-2, -1)))
    raise ValueError(f"unsupported TTA transform: {name!r}")


def _invert_transform(probability: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return np.ascontiguousarray(probability)
    if name == "horizontal":
        return np.ascontiguousarray(np.flip(probability, axis=-1))
    if name == "vertical":
        return np.ascontiguousarray(np.flip(probability, axis=-2))
    if name == "rot90":
        return np.ascontiguousarray(np.rot90(probability, k=-1, axes=(-2, -1)))
    raise ValueError(f"unsupported TTA transform: {name!r}")


def _reflect_pad(image: np.ndarray, bottom: int, right: int) -> np.ndarray:
    if bottom == 0 and right == 0:
        return np.ascontiguousarray(image)
    return cv2.copyMakeBorder(
        image,
        0,
        bottom,
        0,
        right,
        borderType=cv2.BORDER_REFLECT_101,
    )


def _extract_tile(gray: np.ndarray, tile: Tile, size: int) -> np.ndarray:
    height, width = gray.shape
    y1 = min(tile.y + size, height)
    x1 = min(tile.x + size, width)
    return _reflect_pad(
        gray[tile.y:y1, tile.x:x1],
        bottom=tile.y + size - y1,
        right=tile.x + size - x1,
    )


def _image_tensor(gray: np.ndarray) -> Tensor:
    return image_norm.segformer_image_tensor(gray)


def _texture_tensor(gray: np.ndarray) -> Tensor:
    texture = np.moveaxis(compact_texture_channels(gray), -1, 0)
    return torch.from_numpy(np.ascontiguousarray(texture))


def _resolve_model(
    model: nn.Module | str | Path | Mapping[str, object],
    device: torch.device,
) -> nn.Module:
    if isinstance(model, nn.Module):
        return model
    if isinstance(model, (str, Path)):
        path: str | Path = model
        expected = None
        architecture = None
    elif isinstance(model, Mapping):
        path = model["path"]  # type: ignore[assignment]
        expected = model.get("sha256")
        architecture = model.get("architecture")
    else:
        raise TypeError("models must be modules, checkpoint paths, or mappings")
    checkpoint = load_verified_checkpoint(
        path,
        None if expected is None else str(expected),
        map_location=device,
    )
    if not isinstance(architecture, Mapping):
        architecture = checkpoint.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("checkpoint is missing model architecture")
    resolved = QuerySegFormer(
        backbone=str(architecture["backbone"]),
        pretrained=False,
        decoder_channels=int(architecture.get("decoder_channels", 128)),
    )
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint is missing state_dict")
    resolved.load_state_dict(state, strict=True)
    return resolved


def _prepare_model(model: nn.Module, device: torch.device) -> nn.Module:
    model.to(device)
    model.eval()
    return model


def _default_runner(
    model: nn.Module, image: Tensor, query: Tensor, texture: Tensor
) -> Tensor:
    return model(image, query, texture)


def _predict_view(
    gray: np.ndarray,
    query: Tensor,
    model: nn.Module,
    *,
    tile_size: int,
    overlap: int,
    batch_size: int,
    device: torch.device,
    batch_runner: BatchRunner | None,
) -> np.ndarray:
    tiles = list(iter_tiles(gray.shape[0], gray.shape[1], tile_size, overlap))
    crops = [_extract_tile(gray, tile, tile_size) for tile in tiles]
    grouped: dict[tuple[int, int], list[int]] = {}
    for index, crop in enumerate(crops):
        grouped.setdefault(crop.shape, []).append(index)
    native: list[np.ndarray | None] = [None] * len(tiles)
    runner = batch_runner or _default_runner
    for indices in grouped.values():
        _run_shape_group(
            indices,
            crops,
            tiles,
            native,
            query=query,
            model=model,
            batch_size=batch_size,
            device=device,
            runner=runner,
        )
    values = [value for value in native if value is not None]
    if len(values) != len(tiles):
        raise RuntimeError("tile batch did not produce every native window")
    return blend_tiles(gray.shape, tiles, values)


def _run_shape_group(
    indices: Sequence[int],
    crops: Sequence[np.ndarray],
    tiles: Sequence[Tile],
    native: list[np.ndarray | None],
    *,
    query: Tensor,
    model: nn.Module,
    batch_size: int,
    device: torch.device,
    runner: BatchRunner,
) -> None:
    start = 0
    current = min(batch_size, len(indices))
    while start < len(indices):
        size = min(current, len(indices) - start)
        batch_ids = indices[start : start + size]
        image = texture = batched_query = logits = None
        retry_smaller = False
        try:
            image = torch.stack([_image_tensor(crops[index]) for index in batch_ids]).to(
                device
            )
            texture = torch.stack(
                [_texture_tensor(crops[index]) for index in batch_ids]
            ).to(device)
            batched_query = query.unsqueeze(0).expand(size, -1, -1, -1).contiguous().to(
                device
            )
            with torch.no_grad():
                logits = runner(model, image, batched_query, texture)
        except torch.cuda.OutOfMemoryError:
            if size == 1:
                raise
            retry_smaller = True
        if retry_smaller:
            del image, texture, batched_query, logits
            torch.cuda.empty_cache()
            current = size // 2
            continue
        if logits is None:
            raise RuntimeError("tile forward did not return logits")
        probabilities = (
            torch.sigmoid(logits[:, 0].detach().float()).cpu().numpy().astype(np.float32)
        )
        for offset, index in enumerate(batch_ids):
            tile = tiles[index]
            native[index] = np.ascontiguousarray(
                probabilities[offset, : tile.height, : tile.width]
            )
        start += size
        current = min(current, batch_size)
