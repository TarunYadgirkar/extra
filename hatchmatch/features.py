"""Native-coordinate, multi-scale texture features for hatch matching."""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np

from hatchmatch.contracts import Box

ORIENTATIONS = tuple(np.linspace(0.0, np.pi, 12, endpoint=False))
WAVELENGTHS = (4.0, 6.0, 9.0, 13.0)
SCALE_FACTORS = (0.75, 1.0, 1.5)
LOCAL_RADII = (2, 4, 8)


def _validate_inputs(gray: np.ndarray, query_box: Box) -> None:
    if not isinstance(gray, np.ndarray):
        raise ValueError("gray must be a NumPy array")
    if gray.ndim != 2:
        raise ValueError("gray must be two-dimensional")
    if gray.dtype != np.uint8:
        raise ValueError("gray must have uint8 dtype")
    if not isinstance(query_box, Box):
        raise ValueError("query_box must be a Box")
    height, width = gray.shape
    if query_box.x1 > width or query_box.y1 > height:
        raise ValueError("query_box exceeds native image bounds")


def _enhanced_ink(gray: np.ndarray) -> np.ndarray:
    ink_u8 = np.uint8(255) - gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(ink_u8).astype(np.float32) / np.float32(255.0)


def _foreground_mask(ink: np.ndarray) -> np.ndarray:
    ink_u8 = np.rint(ink * np.float32(255.0)).astype(np.uint8)
    threshold, _ = cv2.threshold(
        ink_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return (ink_u8 > max(float(threshold), 12.0)).astype(np.float32)


def _unit_channel(channel: np.ndarray) -> np.ndarray:
    bounded = np.clip(channel, 0.0, 1.0).astype(np.float32, copy=False)
    return np.round(bounded, decimals=6).astype(np.float32, copy=False)


def _gabor_energy(
    ink: np.ndarray,
    theta: float,
    wavelength: float,
    scale: float,
) -> np.ndarray:
    sigma = max(1.0, 0.45 * wavelength * scale)
    radius = min(15, max(2, int(np.ceil(3.0 * sigma))))
    size = 2 * radius + 1
    real_kernel = cv2.getGaborKernel(
        (size, size),
        sigma,
        theta,
        wavelength,
        0.5,
        0.0,
        ktype=cv2.CV_32F,
    )
    imag_kernel = cv2.getGaborKernel(
        (size, size),
        sigma,
        theta,
        wavelength,
        0.5,
        np.pi / 2.0,
        ktype=cv2.CV_32F,
    )
    real_kernel -= real_kernel.mean()
    imag_kernel -= imag_kernel.mean()
    normalizer = max(
        float(np.abs(real_kernel).sum()),
        float(np.abs(imag_kernel).sum()),
        np.finfo(np.float32).eps,
    )
    real = cv2.filter2D(
        ink, cv2.CV_32F, real_kernel / normalizer, borderType=cv2.BORDER_REFLECT
    )
    imag = cv2.filter2D(
        ink, cv2.CV_32F, imag_kernel / normalizer, borderType=cv2.BORDER_REFLECT
    )
    energy = cv2.magnitude(real, imag)
    smoothing_sigma = max(0.5, wavelength * scale / 4.0)
    energy = cv2.GaussianBlur(
        energy,
        (0, 0),
        smoothing_sigma,
        borderType=cv2.BORDER_REFLECT,
    )
    return _unit_channel(energy)


def _iter_texture_channels(
    gray: np.ndarray, query_box: Box
) -> Iterator[np.ndarray]:
    """Yield texture channels without materializing the complete feature cube."""

    _validate_inputs(gray, query_box)
    ink = _enhanced_ink(gray)
    foreground = _foreground_mask(ink)
    distance = cv2.distanceTransform(
        np.uint8(1) - foreground.astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )

    for radius in LOCAL_RADII:
        size = 2 * radius + 1
        density = cv2.boxFilter(
            foreground,
            cv2.CV_32F,
            (size, size),
            normalize=True,
            borderType=cv2.BORDER_REFLECT,
        )
        yield _unit_channel(density)
        yield _unit_channel(np.exp(-distance / np.float32(radius)))

        mean = cv2.boxFilter(
            ink,
            cv2.CV_32F,
            (size, size),
            normalize=True,
            borderType=cv2.BORDER_REFLECT,
        )
        mean_square = cv2.boxFilter(
            ink * ink,
            cv2.CV_32F,
            (size, size),
            normalize=True,
            borderType=cv2.BORDER_REFLECT,
        )
        variance = np.maximum(mean_square - mean * mean, 0.0)
        yield _unit_channel(variance * np.float32(4.0))

    for scale in SCALE_FACTORS:
        for wavelength in WAVELENGTHS:
            for theta in ORIENTATIONS:
                yield _gabor_energy(ink, theta, wavelength, scale)


def texture_channels(gray: np.ndarray, query_box: Box) -> np.ndarray:
    """Return ``(height, width, channels)`` texture features in ``[0, 1]``."""

    return np.stack(tuple(_iter_texture_channels(gray, query_box)), axis=-1)
