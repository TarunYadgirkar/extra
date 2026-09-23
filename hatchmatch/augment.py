"""Deterministic augmentations for hatch-training samples."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class _Geometry:
    scale: float
    angle: float
    perspective: tuple[tuple[float, float], ...]


def _validate_image(name: str, value: np.ndarray) -> None:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    if value.dtype != np.uint8:
        raise ValueError(f"{name} must have uint8 dtype")
    if value.ndim not in (2, 3) or (value.ndim == 3 and value.shape[2] not in (1, 3)):
        raise ValueError(f"{name} must be grayscale or three-channel")
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")


def _validate_mask(
    name: str,
    value: np.ndarray,
    shape: tuple[int, int],
) -> None:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    if value.dtype != np.bool_:
        raise ValueError(f"{name} must have bool dtype")
    if value.shape != shape:
        raise ValueError(f"{name} must match the drawing dimensions")


def _geometry(rng: np.random.Generator) -> _Geometry:
    return _Geometry(
        scale=float(rng.uniform(0.9, 1.1)),
        angle=float(rng.uniform(-12.0, 12.0)),
        perspective=tuple(
            (float(x), float(y))
            for x, y in rng.uniform(-0.025, 0.025, size=(4, 2))
        ),
    )


def _homography(shape: tuple[int, int], geometry: _Geometry) -> np.ndarray:
    height, width = shape
    corners = np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )
    center = ((width - 1.0) / 2.0, (height - 1.0) / 2.0)
    affine = cv2.getRotationMatrix2D(center, geometry.angle, geometry.scale)
    destination = cv2.transform(corners[None, :, :], affine)[0]
    jitter_scale = np.array([max(width - 1, 1), max(height - 1, 1)], np.float32)
    destination += np.asarray(geometry.perspective, np.float32) * jitter_scale
    return cv2.getPerspectiveTransform(corners, destination)


def _warp_image(image: np.ndarray, geometry: _Geometry) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.warpPerspective(
        image,
        _homography((height, width), geometry),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255 if image.ndim == 2 else (255, 255, 255),
    )


def _warp_mask(mask: np.ndarray, geometry: _Geometry) -> np.ndarray:
    height, width = mask.shape
    warped = cv2.warpPerspective(
        mask.astype(np.uint8),
        _homography(mask.shape, geometry),
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped != 0


def _line_dropout(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    color: int | tuple[int, int, int]
    color = 255 if output.ndim == 2 else (255, 255, 255)
    diagonal = float(np.hypot(height, width))
    for _ in range(int(rng.integers(1, 4))):
        center_x = int(rng.integers(0, width))
        center_y = int(rng.integers(0, height))
        angle = float(rng.uniform(0.0, np.pi))
        half_length = float(rng.uniform(0.1, 0.35)) * diagonal
        delta_x = int(round(np.cos(angle) * half_length))
        delta_y = int(round(np.sin(angle) * half_length))
        cv2.line(
            output,
            (center_x - delta_x, center_y - delta_y),
            (center_x + delta_x, center_y + delta_y),
            color,
            thickness=int(rng.integers(1, 4)),
            lineType=cv2.LINE_AA,
        )
    return output


def _degrade_appearance(
    image: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    sigma = float(rng.uniform(0.15, 1.25))
    output = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT,
    )
    contrast = float(rng.uniform(0.82, 1.18))
    brightness = float(rng.uniform(-12.0, 12.0))
    output = np.clip(output.astype(np.float32) * contrast + brightness, 0, 255)
    noise = rng.normal(0.0, float(rng.uniform(1.0, 7.0)), size=output.shape)
    output = np.clip(output + noise, 0, 255).astype(np.uint8)

    quality = int(rng.integers(55, 96))
    encoded, payload = cv2.imencode(
        ".jpg",
        output,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not encoded:
        raise RuntimeError("JPEG augmentation failed to encode")
    flag = cv2.IMREAD_GRAYSCALE if output.ndim == 2 else cv2.IMREAD_COLOR
    decoded = cv2.imdecode(payload, flag)
    if decoded is None:
        raise RuntimeError("JPEG augmentation failed to decode")
    return _line_dropout(decoded, rng)


def augment_training_sample(
    image: np.ndarray,
    query: np.ndarray,
    target: np.ndarray,
    known: np.ndarray,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Apply deterministic shared geometry and independent appearance noise.

    Drawing, target, and known arrays receive one geometric transform. The
    query receives equivalent normalized geometry, then drawing and query
    appearance are degraded with independent random streams. Masks use nearest
    interpolation and remain Boolean.
    """

    _validate_image("image", image)
    _validate_image("query", query)
    _validate_mask("target", target, image.shape[:2])
    _validate_mask("known", known, image.shape[:2])
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    geometry_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0]))
    image_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1]))
    query_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 2]))
    geometry = _geometry(geometry_rng)

    transformed_known = _warp_mask(known, geometry)
    transformed_target = _warp_mask(target, geometry) & transformed_known
    transformed_image = _degrade_appearance(
        _warp_image(image, geometry),
        image_rng,
    )
    transformed_query = _degrade_appearance(
        _warp_image(query, geometry),
        query_rng,
    )
    return {
        "image": transformed_image,
        "query": transformed_query,
        "target": transformed_target,
        "known": transformed_known,
    }
