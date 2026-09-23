"""Shared inputs for the neural head: per-image stride-14 input stack and per-query fixed similarity maps."""
import sys
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SOL = HERE.parents[1]
ROOT = SOL.parent
sys.path.insert(0, str(SOL))
from pixfeat import INK, _norm, query_features, sim_grids  # noqa: E402

S = 14
NPCA = 64
SIM_KEYS = ("sm", "sr", "tk", "lda", "ldar")
IMG_DIR = ROOT / "cache" / "nn_img"
Q_DIR = ROOT / "cache" / "nn_q"
PCA_PATH = HERE / "weights" / "pca.npz"


def tone14(gray, gh, gw):
    """Mean, std, ink fraction and edge density per 14px cell (image padded with paper)."""
    H, W = gray.shape
    g = np.ones((gh * S, gw * S), np.float32)
    g[:H, :W] = gray.astype(np.float32) / 255.0
    ink = (g < INK / 255.0).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    area = lambda a: a.reshape(gh, S, gw, S).mean((1, 3))
    m = area(g)
    sd = np.sqrt(np.maximum(area(g * g) - m * m, 0))
    return np.stack([m, sd, area(ink), area(edge)], -1)


def up28(f28, gh, gw):
    """Bilinear (half-pixel) upsample of a stride-28 grid onto the stride-14 grid."""
    h, w, c = f28.shape
    up = cv2.resize(f28.astype(np.float32), (w * 2, h * 2), interpolation=cv2.INTER_LINEAR).reshape(h * 2, w * 2, c)
    out = np.empty((gh, gw, c), np.float32)
    hh, ww = min(gh, h * 2), min(gw, w * 2)
    out[:hh, :ww] = up[:hh, :ww]
    out[hh:] = out[hh - 1:hh]; out[:, ww:] = out[:, ww - 1:ww]
    return out


def image_input(gray, f14, f28, pca):
    """(gh, gw, 2*NPCA+4) float16: PCA of normalized DINO at s14 and s28 (upsampled) + tone."""
    gh, gw = f14.shape[:2]
    a = _norm(f14).reshape(-1, f14.shape[-1])
    p14 = ((a - pca["mu14"]) @ pca["W14"]).reshape(gh, gw, -1)
    b = _norm(f28).reshape(-1, f28.shape[-1])
    p28 = ((b - pca["mu28"]) @ pca["W28"]).reshape(*f28.shape[:2], -1)
    return np.concatenate([p14, up28(p28, gh, gw), tone14(gray, gh, gw)], -1).astype(np.float16)


class LiteCtx:
    """Just the DINO part of pixfeat.ImageContext (no tone/hist maps)."""

    def __init__(self, dino):
        self.tone, self.hist = {}, {}
        self.dino = [(_norm(f), s) for f, s in dino]
        self.stats = []
        for f, _ in self.dino:
            flat = f.reshape(-1, f.shape[-1])
            sub = flat[:: max(1, len(flat) // 60000)]
            mu = sub.mean(0)
            cov = np.cov(sub, rowvar=False)
            cov += np.eye(len(mu)) * (0.1 * np.trace(cov) / len(mu))
            self.stats.append((mu, np.linalg.inv(cov).astype(np.float32)))


def query_sims(ctx, box, gh, gw):
    """(gh, gw, 10) fixed similarity maps on the s14 grid and their (10, 101) image quantiles."""
    _, protos = query_features(ctx, box)
    maps = []
    for s, g in sim_grids(ctx, protos):
        for k in SIM_KEYS:
            m = g[k][..., None]
            maps.append(m[..., 0] if s == S else up28(m, gh, gw)[..., 0])
    maps = np.stack(maps, -1).astype(np.float32)
    qs = np.quantile(maps.reshape(-1, maps.shape[-1]), np.linspace(0, 1, 101), axis=0).T
    return maps.astype(np.float16), qs.astype(np.float32)


def query_cells(box, gh, gw):
    """s14 cell index ranges whose centres fall inside the box (same rule as pixfeat._cells)."""
    x0, y0, x1, y1 = box
    iy = np.arange(int(y0 // S), int(np.ceil(y1 / S))); ix = np.arange(int(x0 // S), int(np.ceil(x1 / S)))
    cy, cx = (iy + 0.5) * S, (ix + 0.5) * S
    iy2, ix2 = iy[(cy >= y0) & (cy < y1)], ix[(cx >= x0) & (cx < x1)]
    iy = iy2 if len(iy2) else iy
    ix = ix2 if len(ix2) else ix
    iy, ix = np.clip(iy, 0, gh - 1), np.clip(ix, 0, gw - 1)
    return int(iy.min()), int(iy.max()) + 1, int(ix.min()), int(ix.max()) + 1
