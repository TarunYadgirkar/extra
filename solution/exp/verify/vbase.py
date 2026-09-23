"""Shared dev helpers for the component verifier (exp/verify). Never used at inference."""
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "calib"))
from common import CACHE, ROOT, SOL, boot, fmt, doc_macro, examples, domain, cv_split  # noqa: E402,F401
from pixfeat import sample_grid  # noqa: E402

DOWN, THR = 4, 0.5


def pp(g, t=THR):
    """Exactly the infer.py post-processing (blur sigma 14 native px, fill every enclosed hole)."""
    g = cv2.GaussianBlur(g, (0, 0), 14 / DOWN)
    lab, n = ndi.label(g <= t)
    fill = np.ones(n + 1, bool); fill[0] = False
    fill[np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])] = False
    return np.where(fill[lab], t + 0.01, g)


def queries(split):
    if split == "val":
        exs = examples("val")
        return exs, [(i, e, -1) for i, e in enumerate(exs)]
    exs, qf = cv_split()
    return exs, [(i, e, int(qf[i])) for i, e in enumerate(exs) if e["kind"] == "real"]


def load_raw(e):
    return np.load(CACHE / "cal_p4" / f"{e['id']}.npy").astype(np.float32)


def components(h, t=THR):
    lab, n = ndi.label(h > t)
    return lab, n


def dom_cells(ys, xs, shape):
    fy = np.clip(np.round((ys + 0.5) / DOWN - 0.5), 0, shape[0] - 1).astype(int)
    fx = np.clip(np.round((xs + 0.5) / DOWN - 0.5), 0, shape[1] - 1).astype(int)
    return fy, fx


class Dom:
    """Known-domain pixels of one query, optionally subsampled; predictions come from the upsampled grid like infer.py."""

    def __init__(self, e, max_px=None, seed=0):
        d = domain(e)
        idx = np.arange(len(d["ys"]))
        if max_px and len(idx) > max_px:
            idx = np.sort(np.random.default_rng(seed).choice(idx, max_px, replace=False))
        self.ys, self.xs, self.y = d["ys"][idx].astype(np.float32), d["xs"][idx].astype(np.float32), d["pos"][idx]

    def mask(self, h, t=THR):
        return sample_grid(h, DOWN, self.ys, self.xs) > t

    def iou(self, m):
        u = (m | self.y).sum()
        return (m & self.y).sum() / u if self.y.any() else np.nan
