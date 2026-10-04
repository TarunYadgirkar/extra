"""Query-conditioned classical segmenter for hatch and tone regions.

Flat gray fills are matched by local median level. Line, dash, stipple, and
lattice patterns are matched by multi-scale Gabor energy, using only the query
pixels that carry the pattern so a symbol inside the query box does not set
the prototype.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from hatchmatch.contracts import Box

WAVELENGTHS = (4.0, 8.0, 12.0, 18.0, 28.0)
ORIENTATION_COUNT = 6
TONE_MODE_MAX = 242
TONE_FRACTION_MIN = 0.34
DEFAULT_MAX_SIDE = 2000


@dataclass(frozen=True)
class PatternScore:
    """Working-resolution similarity in ``[0, 1]`` plus the decisions behind it."""

    score: np.ndarray
    branch: str
    selection: np.ndarray
    query: tuple[int, int, int, int]
    native_size: tuple[int, int]
    working_shape: tuple[int, int]


def segment_mask(
    gray: np.ndarray,
    query_box: Box,
    *,
    max_side: int = DEFAULT_MAX_SIDE,
    tone_threshold: float = 0.7,
    alpha: float = 0.75,
) -> np.ndarray:
    """Return a native Boolean mask for the pattern inside ``query_box``."""

    pattern = pattern_score(gray, query_box, max_side=max_side)
    threshold = decision_threshold(
        pattern,
        tone_threshold=tone_threshold,
        alpha=alpha,
    )
    return mask_from_score(pattern, threshold)


def pattern_score(
    gray: np.ndarray,
    query_box: Box,
    *,
    max_side: int = DEFAULT_MAX_SIDE,
) -> PatternScore:
    """Score a resized page. The result is not yet a native mask."""

    _validate_gray(gray)
    if not isinstance(query_box, Box):
        raise ValueError("query_box must be a Box")
    height, width = gray.shape
    if query_box.x1 > width or query_box.y1 > height:
        raise ValueError("query_box exceeds native image bounds")
    if type(max_side) is not int or max_side < 64:
        raise ValueError("max_side must be an integer of at least 64")

    native_query = (query_box.x0, query_box.y0, query_box.x1, query_box.y1)
    crop = gray[query_box.y0 : query_box.y1, query_box.x0 : query_box.x1]
    mode, fraction = _gray_mode(crop)
    if mode <= TONE_MODE_MAX and fraction >= TONE_FRACTION_MIN:
        # Keep the native gray level. Downsampling blends black lines into a
        # flat fill and the fill no longer matches the query mode.
        score = _tone_score(gray, mode, min(crop.shape))
        selection = np.zeros(gray.shape, dtype=bool)
        selection[query_box.y0 : query_box.y1, query_box.x0 : query_box.x1] = True
        return PatternScore(
            score=score,
            branch="tone",
            selection=selection,
            query=native_query,
            native_size=(width, height),
            working_shape=gray.shape,
        )

    working, scale_x, scale_y = _resize_max(gray, max_side)
    query = _scale_box(query_box, scale_x, scale_y, working.shape[1], working.shape[0])
    selection = _pattern_pixels(working, query)
    score = _gabor_score(working, selection)
    return PatternScore(
        score=score,
        branch="texture",
        selection=selection,
        query=query,
        native_size=(width, height),
        working_shape=working.shape,
    )


def decision_threshold(
    pattern: PatternScore,
    *,
    tone_threshold: float = 0.7,
    alpha: float = 0.75,
) -> float:
    """Pick a label-free cutoff from the query's own score."""

    if pattern.branch == "tone":
        return float(tone_threshold)
    selected = pattern.score[pattern.selection]
    if selected.size == 0:
        reference = 0.7
    else:
        reference = float(np.percentile(selected, 40))
    return float(np.clip(alpha * reference, 0.28, 0.9))


def mask_from_score(pattern: PatternScore, threshold: float) -> np.ndarray:
    """Threshold a working score and expand it to the native page size."""

    working = pattern.score >= np.float32(threshold)
    native_width, native_height = pattern.native_size
    if working.shape == (native_height, native_width):
        return working
    restored = cv2.resize(
        working.astype(np.uint8),
        (native_width, native_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return restored.astype(bool, copy=False)


def _validate_gray(gray: np.ndarray) -> None:
    if not isinstance(gray, np.ndarray):
        raise ValueError("gray must be a NumPy array")
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("gray must be a two-dimensional uint8 array")
    if gray.shape[0] == 0 or gray.shape[1] == 0:
        raise ValueError("gray must not be empty")


def _resize_max(
    gray: np.ndarray, max_side: int
) -> tuple[np.ndarray, float, float]:
    height, width = gray.shape
    longest = max(height, width)
    if longest <= max_side:
        return gray, 1.0, 1.0
    scale = max_side / longest
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        gray,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    return resized, resized_width / width, resized_height / height


def _scale_box(
    box: Box,
    scale_x: float,
    scale_y: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x0 = min(width - 1, max(0, int(np.floor(box.x0 * scale_x))))
    y0 = min(height - 1, max(0, int(np.floor(box.y0 * scale_y))))
    x1 = min(width, max(x0 + 1, int(np.ceil(box.x1 * scale_x))))
    y1 = min(height, max(y0 + 1, int(np.ceil(box.y1 * scale_y))))
    return x0, y0, x1, y1


def _gray_mode(crop: np.ndarray) -> tuple[int, float]:
    histogram = np.bincount(crop.ravel(), minlength=256).astype(np.float32)
    smoothed = cv2.GaussianBlur(histogram.reshape(1, -1), (1, 0), 2).ravel()
    peak = int(np.argmax(smoothed))
    weights = (np.abs(np.arange(256) - peak) <= 2).astype(np.float32)
    fraction = float(weights.dot(smoothed) / max(float(smoothed.sum()), 1.0))
    return peak, fraction


def _tone_score(gray: np.ndarray, mode: int, query_side: int) -> np.ndarray:
    band = (
        np.abs(gray.astype(np.int16) - np.int16(mode)) <= np.int16(8)
    ).astype(np.uint8)
    radius = int(np.clip(round(query_side * 0.15), 2, 12))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    closed = cv2.morphologyEx(band, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(
        closed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return opened.astype(np.float32, copy=False)


def _pattern_pixels(gray: np.ndarray, query: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = query
    crop = gray[y0:y1, x0:x1]
    paper = float(np.percentile(crop, 95))
    faint = (gray < paper - 5) & (gray > paper - 60)
    selected = np.zeros(gray.shape, dtype=bool)
    selected[y0:y1, x0:x1] = faint[y0:y1, x0:x1]
    minimum = max(30, int(0.03 * crop.size))
    if int(selected.sum()) < minimum:
        dark = gray < paper - 25
        selected[y0:y1, x0:x1] = dark[y0:y1, x0:x1]
    if int(selected.sum()) < 20:
        selected[y0:y1, x0:x1] = True
    return selected


def _gabor_kernel(theta: float, wavelength: float) -> tuple[np.ndarray, np.ndarray]:
    sigma = max(1.0, 0.5 * wavelength)
    radius = min(14, max(2, int(np.ceil(2.4 * sigma))))
    size = 2 * radius + 1
    real = cv2.getGaborKernel(
        (size, size), sigma, theta, wavelength, 0.5, 0.0, ktype=cv2.CV_32F
    )
    imag = cv2.getGaborKernel(
        (size, size),
        sigma,
        theta,
        wavelength,
        0.5,
        np.pi / 2.0,
        ktype=cv2.CV_32F,
    )
    real = real - real.mean()
    imag = imag - imag.mean()
    normalizer = max(float(np.abs(real).sum()), float(np.abs(imag).sum()), 1e-6)
    return (real / normalizer).astype(np.float32), (imag / normalizer).astype(np.float32)


def _kernels() -> list[tuple[float, np.ndarray, np.ndarray]]:
    cached = getattr(_kernels, "cache", None)
    if cached is not None:
        return cached
    built: list[tuple[float, np.ndarray, np.ndarray]] = []
    orientations = np.linspace(0.0, np.pi, ORIENTATION_COUNT, endpoint=False)
    for wavelength in WAVELENGTHS:
        smooth = max(1.2, wavelength * 0.8)
        for theta in orientations:
            real, imag = _gabor_kernel(float(theta), float(wavelength))
            built.append((smooth, real, imag))
    _kernels.cache = built  # type: ignore[attr-defined]
    return built


def _gabor_score(gray: np.ndarray, selected: np.ndarray) -> np.ndarray:
    ink = (np.float32(255.0) - gray.astype(np.float32)) / np.float32(255.0)
    if selected.any():
        cap = float(np.percentile(ink[selected], 90))
    else:
        cap = 0.5
    ink = np.minimum(ink, np.float32(max(cap, 0.05)))
    dot = np.zeros(gray.shape, dtype=np.float32)
    energy = np.zeros(gray.shape, dtype=np.float32)
    prototype_energy = 0.0
    for smooth, real, imag in _kernels():
        response = cv2.magnitude(
            cv2.filter2D(ink, cv2.CV_32F, real, borderType=cv2.BORDER_REFLECT),
            cv2.filter2D(ink, cv2.CV_32F, imag, borderType=cv2.BORDER_REFLECT),
        )
        response = cv2.GaussianBlur(
            response, (0, 0), smooth, borderType=cv2.BORDER_REFLECT
        )
        prototype = float(np.median(response[selected])) if selected.any() else 0.0
        dot += np.float32(prototype) * response
        energy += response * response
        prototype_energy += prototype * prototype
        del response
    local_norm = np.sqrt(np.maximum(energy, np.float32(1e-12)))
    prototype_norm = np.float32(np.sqrt(max(prototype_energy, 1e-12)))
    cosine = dot / (local_norm * prototype_norm)
    ratio = local_norm / prototype_norm
    magnitude = np.exp(
        -np.abs(np.log(np.clip(ratio, np.float32(1e-3), np.float32(30.0))))
        / np.float32(0.55)
    )
    return (np.clip(cosine, 0.0, 1.0) * magnitude).astype(np.float32, copy=False)
