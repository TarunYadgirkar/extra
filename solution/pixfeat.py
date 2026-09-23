"""Per-location features for the query-conditioned classifier. Shared by dev and inference."""
import cv2
import numpy as np

DOWN = 4  # classifier grid stride in native pixels
INK = 160


def _norm(f):
    f = f.astype(np.float32)
    return f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-6)


def _cells(box, stride, gh, gw):
    x0, y0, x1, y1 = box
    iy = np.arange(int(y0 // stride), int(np.ceil(y1 / stride)))
    ix = np.arange(int(x0 // stride), int(np.ceil(x1 / stride)))
    cy, cx = (iy + 0.5) * stride, (ix + 0.5) * stride
    iy2, ix2 = iy[(cy >= y0) & (cy < y1)], ix[(cx >= x0) & (cx < x1)]
    iy = iy2 if len(iy2) else iy
    ix = ix2 if len(ix2) else ix
    return np.clip(iy, 0, gh - 1), np.clip(ix, 0, gw - 1)


def sample_grid(grid, stride, ys, xs):
    gh, gw = grid.shape[:2]
    fy = np.clip((ys + 0.5) / stride - 0.5, 0, gh - 1)
    fx = np.clip((xs + 0.5) / stride - 0.5, 0, gw - 1)
    y0 = np.floor(fy).astype(np.int64); x0 = np.floor(fx).astype(np.int64)
    y1 = np.minimum(y0 + 1, gh - 1); x1 = np.minimum(x0 + 1, gw - 1)
    wy, wx = fy - y0, fx - x0
    if grid.ndim == 3:
        wy, wx = wy[:, None], wx[:, None]
    return (grid[y0, x0] * (1 - wy) * (1 - wx) + grid[y0, x1] * (1 - wy) * wx
            + grid[y1, x0] * wy * (1 - wx) + grid[y1, x1] * wy * wx)


def tone_maps(gray):
    """Low-res tone/ink statistics maps on the DOWN grid."""
    g = gray.astype(np.float32) / 255.0
    ink = (gray < INK).astype(np.float32)
    h, w = gray.shape
    sz = (w // DOWN, h // DOWN)
    small = cv2.resize(g, sz, interpolation=cv2.INTER_AREA)
    small_ink = cv2.resize(ink, sz, interpolation=cv2.INTER_AREA)
    maps = {}
    for r in (4, 12):  # radius in DOWN cells -> 16px, 48px
        k = 2 * r + 1
        m = cv2.blur(small, (k, k))
        m2 = cv2.blur(small * small, (k, k))
        maps[f"mean{r}"] = m
        maps[f"std{r}"] = np.sqrt(np.maximum(m2 - m * m, 0))
        maps[f"ink{r}"] = cv2.blur(small_ink, (k, k))
        maps[f"bg{r}"] = cv2.dilate(cv2.medianBlur((small * 255).astype(np.uint8), 5), np.ones((k, k), np.uint8)).astype(np.float32) / 255
    # tone of non-ink paper: mean of small pixels that contain little ink
    paper = (small_ink < 0.05).astype(np.float32)
    for r in (4, 12):
        k = 2 * r + 1
        maps[f"paper{r}"] = cv2.blur(small * paper, (k, k)) / (cv2.blur(paper, (k, k)) + 1e-3)
        maps[f"paperfrac{r}"] = cv2.blur(paper, (k, k))
    return maps


NBINS = 8


def hist_maps(gray):
    """Local orientation (doubled-angle) and intensity histograms on the DOWN grid."""
    g = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi * NBINS
    b = np.minimum(ang.astype(np.int32), NBINS - 1)
    h, w = gray.shape
    sz = (w // DOWN, h // DOWN)
    ori = np.stack([cv2.resize(mag * (b == i), sz, interpolation=cv2.INTER_AREA) for i in range(NBINS)], -1)
    ib = np.minimum((g * NBINS).astype(np.int32), NBINS - 1)
    inten = np.stack([cv2.resize((ib == i).astype(np.float32), sz, interpolation=cv2.INTER_AREA) for i in range(NBINS)], -1)
    out = {}
    for r in (4, 12):
        k = 2 * r + 1
        o = cv2.blur(ori, (k, k))
        out[f"ori{r}"] = o
        out[f"int{r}"] = cv2.blur(inten, (k, k))
    return out


def _hist_compare(local, q):
    """cosine and L1 distance between local histograms (N,B) and query histogram (B,)."""
    ln = local / (local.sum(1, keepdims=True) + 1e-6)
    qn = q / (q.sum() + 1e-6)
    cos = (local @ q) / (np.linalg.norm(local, axis=1) * np.linalg.norm(q) + 1e-6)
    return cos, np.abs(ln - qn).sum(1), local.sum(1)


class ImageContext:
    def __init__(self, gray, dino):
        """dino: list of (feature_grid, stride)."""
        self.shape = gray.shape
        self.tone = tone_maps(gray)
        self.hist = hist_maps(gray)
        self.dino = [(_norm(f), s) for f, s in dino]
        self.stats = []
        for f, _ in self.dino:
            flat = f.reshape(-1, f.shape[-1])
            sub = flat[:: max(1, len(flat) // 60000)]
            mu = sub.mean(0)
            cov = np.cov(sub, rowvar=False)
            cov += np.eye(len(mu)) * (0.1 * np.trace(cov) / len(mu))
            self.stats.append((mu, np.linalg.inv(cov).astype(np.float32)))


def query_features(ctx, box):
    x0, y0, x1, y1 = box
    q = {}
    sx0, sy0 = x0 // DOWN, y0 // DOWN
    sx1, sy1 = max(x1 // DOWN, sx0 + 1), max(y1 // DOWN, sy0 + 1)
    for k, m in ctx.tone.items():
        q[k] = float(np.median(m[sy0:sy1, sx0:sx1]))
    for k, m in ctx.hist.items():
        q["h_" + k] = m[sy0:sy1, sx0:sx1].reshape(-1, m.shape[-1]).mean(0)
    protos = []
    for f, s in ctx.dino:
        iy, ix = _cells(box, s, *f.shape[:2])
        cells = f[np.ix_(iy, ix)].reshape(-1, f.shape[-1])
        mean = cells.mean(0); mean /= np.linalg.norm(mean) + 1e-6
        self_sim = cells @ mean
        # robust prototype: drop the least typical third of cells (text, stray lines)
        keep = self_sim >= np.quantile(self_sim, 0.34) if len(cells) >= 3 else np.ones(len(cells), bool)
        rob = cells[keep].mean(0); rob /= np.linalg.norm(rob) + 1e-6
        mu, icov = ctx.stats[len(protos)]
        wv = icov @ (cells.mean(0) - mu)
        lo, hi = float(mu @ wv), float(cells.mean(0) @ wv)
        wr = icov @ (cells[keep].mean(0) - mu)
        lor, hir = float(mu @ wr), float(cells[keep].mean(0) @ wr)
        protos.append({"lda": (wv, lo, hi), "ldar": (wr, lor, hir), "mean": mean, "rob": rob, "cells": cells, "self_mean": float(self_sim.mean()),
                       "self_min": float(self_sim.min()), "n": len(cells)})
    return q, protos


def sim_grids(ctx, protos):
    out = []
    for (f, s), p in zip(ctx.dino, protos):
        flat = f.reshape(-1, f.shape[-1])
        sm = (flat @ p["mean"]).reshape(f.shape[:2])
        sr = (flat @ p["rob"]).reshape(f.shape[:2])
        cs = flat @ p["cells"].T
        k = min(3, cs.shape[1])
        topk = np.sort(cs, axis=1)[:, -k:].mean(1).reshape(f.shape[:2])
        wv, lo, hi = p["lda"]; wr, lor, hir = p["ldar"]
        lda = ((flat @ wv - lo) / (hi - lo + 1e-6)).reshape(f.shape[:2])
        ldar = ((flat @ wr - lor) / (hir - lor + 1e-6)).reshape(f.shape[:2])
        out.append((s, {"sm": sm, "sr": sr, "tk": topk, "lda": lda, "ldar": ldar}))
    return out


def features_at(ctx, box, ys, xs, cache=None):
    """Feature matrix for native pixel coords (ys, xs)."""
    if cache is None:
        cache = {}
    if "q" not in cache:
        cache["q"], cache["protos"] = query_features(ctx, box)
        cache["sims"] = sim_grids(ctx, cache["protos"])
        cache["quant"] = [{k: np.quantile(v, np.linspace(0, 1, 101)) for k, v in g.items()} for _, g in cache["sims"]]
    q, protos = cache["q"], cache["protos"]
    cols, names = [], []
    for k, m in ctx.tone.items():
        v = sample_grid(m, DOWN, ys, xs)
        cols += [v, v - q[k], np.abs(v - q[k])]; names += [k, k + "_d", k + "_ad"]
        cols.append(np.full_like(v, q[k])); names.append("q_" + k)
    for k, m in ctx.hist.items():
        qh = q["h_" + k]
        cos, l1, tot = _hist_compare(sample_grid(m, DOWN, ys, xs), qh)
        cols += [cos, l1, tot - qh.sum(), np.full(len(ys), qh.sum())]
        names += [k + "_cos", k + "_l1", k + "_totd", "q_" + k + "_tot"]
    for i, ((s, g), p, qt) in enumerate(zip(cache["sims"], protos, cache["quant"])):
        for k, grid in g.items():
            v = sample_grid(grid, s, ys, xs)
            cols += [v, v - p["self_mean"], v - p["self_min"], np.interp(v, qt[k], np.linspace(0, 1, 101))]
            names += [f"{k}{i}", f"{k}{i}_rm", f"{k}{i}_rmin", f"{k}{i}_rank"]
        cols += [np.full(len(ys), p["self_mean"]), np.full(len(ys), p["self_min"]), np.full(len(ys), p["n"])]
        names += [f"selfmean{i}", f"selfmin{i}", f"n{i}"]
    x0, y0, x1, y1 = box
    cols += [np.full(len(ys), x1 - x0), np.full(len(ys), y1 - y0)]; names += ["qw", "qh"]
    return np.stack(cols, 1).astype(np.float32), names
