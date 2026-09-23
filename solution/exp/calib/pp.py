"""Label-free post-processing ops on a probability grid (stride st native px). Pure functions of (grid, st, guide)."""
import cv2
import numpy as np
from scipy import ndimage as ndi


def median(k_px):
    def f(g, st, gd=None):
        k = max(3, int(round(k_px / st)) | 1)
        q = np.clip(g * 255 + 0.5, 0, 255).astype(np.uint8)
        return cv2.medianBlur(q, k).astype(np.float32) / 255 if k <= 255 else ndi.median_filter(g, k)
    return f


def blur(sig_px):
    return lambda g, st, gd=None: cv2.GaussianBlur(g, (0, 0), max(sig_px / st, 0.3))


def fill_holes(a_px, t=0.5):
    """Set enclosed below-threshold components smaller than a_px native px^2 to just above threshold."""
    def f(g, st, gd=None):
        lab, n = ndi.label(g <= t)
        if n == 0:
            return g
        area = np.bincount(lab.ravel(), minlength=n + 1) * st * st
        border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        fill = area < a_px
        fill[0] = False
        fill[border] = False
        return np.where(fill[lab], np.maximum(g, t + 0.01), g)
    return f


def fill_holes_b(a_px, t=0.5):
    """As fill_holes but border-touching components also count (matches the s14 sweep)."""
    def f(g, st, gd=None):
        lab, n = ndi.label(g <= t)
        if n == 0:
            return g
        area = np.bincount(lab.ravel(), minlength=n + 1) * st * st
        fill = area < a_px
        fill[0] = False
        return np.where(fill[lab], np.maximum(g, t + 0.01), g)
    return f


def guided(r_px, eps):
    """He et al. guided filter with the ink-density guide gd (same grid as g)."""
    def f(g, st, gd):
        r = max(1, int(round(r_px / st)))
        box = lambda a: cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1))
        mi, mp = box(gd), box(g)
        a = (box(gd * g) - mi * mp) / (box(gd * gd) - mi * mi + eps)
        b = mp - a * mi
        return np.clip(box(a) * gd + box(b), 0, 1)
    return f


def chain(*fs):
    def f(g, st, gd=None):
        for h in fs:
            g = h(g, st, gd)
        return g
    return f


def ink_guide(gray, st, smooth_px=8):
    """Ink density on the stride-st grid, lightly smoothed (label-free guide)."""
    H, W = gray.shape
    gh, gw = -(-H // st), -(-W // st)
    ink = np.zeros((gh * st, gw * st), np.float32)
    ink[:H, :W] = gray < 160
    d = ink.reshape(gh, st, gw, st).mean((1, 3))
    return cv2.GaussianBlur(d, (0, 0), max(smooth_px / st, 0.3))


def box_scaled(op, c, lo_px, hi_px):
    """Apply op(size_px) with size = clip(c * sqrt(query box area), lo, hi). Returns f(g, st, gd, box)."""
    def f(g, st, gd=None, box=None):
        x0, y0, x1, y1 = box
        k = float(np.clip(c * np.sqrt((x1 - x0) * (y1 - y0)), lo_px, hi_px))
        return op(k)(g, st, gd)
    return f


def min_area_rel(frac, t=0.5):
    """Drop above-threshold components smaller than frac * query box area."""
    def f(g, st, gd=None, box=None):
        x0, y0, x1, y1 = box
        lab, n = ndi.label(g > t)
        if n == 0:
            return g
        area = np.bincount(lab.ravel(), minlength=n + 1) * st * st
        drop = area < frac * (x1 - x0) * (y1 - y0)
        drop[0] = False
        return np.where(drop[lab], np.minimum(g, t - 0.01), g)
    return f


def chain4(*fs):
    def f(g, st, gd, box):
        for h in fs:
            g = h(g, st, gd, box)
        return g
    return f


def hyst(lo, hi):
    """Keep components of g > lo that contain a pixel > hi (as 0.51), zero out the rest below 0.5."""
    def f(g, st, gd=None):
        lab, n = ndi.label(g > lo)
        if n == 0:
            return g
        mx = ndi.maximum(g, lab, np.arange(1, n + 1))
        keep = np.r_[False, mx > hi]
        return np.where(keep[lab], np.maximum(g, 0.51), np.minimum(g, 0.49))
    return f
