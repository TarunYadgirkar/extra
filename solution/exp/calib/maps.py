"""Dev: load full-image prob grids and score them on the known domain (all known pixels)."""
import numpy as np
from common import CACHE, examples, domain, cv_split, doc_macro
from pixfeat import sample_grid

SRC = {"s14": (CACHE / "nn_gbm" / "tx_v1", 14, True), "p4": (CACHE / "cal_p4", 4, False)}


def queries(split):
    if split == "val":
        exs = examples("val")
        return exs, [(i, e, -1) for i, e in enumerate(exs)]
    exs, qf = cv_split()
    return exs, [(i, e, int(qf[i])) for i, e in enumerate(exs) if e["kind"] == "real"]


def load_grid(src, e):
    d, s, is_logit = SRC[src]
    g = np.load(d / f"{e['id']}.npy").astype(np.float32)
    return (1 / (1 + np.exp(-g)) if is_logit else g), s


class Scorer:
    """Caches domains (subsampled for speed) so many post-processing variants can be scored."""

    def __init__(self, split, max_px=400_000, seed=0):
        self.exs, self.qs = queries(split)
        rng = np.random.default_rng(seed)
        self.dom = {}
        for i, e, _ in self.qs:
            d = domain(e)
            idx = np.arange(len(d["ys"]))
            if len(idx) > max_px:
                idx = np.sort(rng.choice(idx, max_px, replace=False))
            self.dom[i] = (d["ys"][idx].astype(np.float32), d["xs"][idx].astype(np.float32), d["pos"][idx])

    def probs(self, i, grid, stride):
        ys, xs, _ = self.dom[i]
        return sample_grid(grid, stride, ys, xs)

    def score(self, per_q_prob, t=0.5):
        """per_q_prob: {i: prob at dom pixels} or {i: bool}. Returns (doc_macro, per-doc dict, per-query iou)."""
        per = {}
        for i, p in per_q_prob.items():
            y = self.dom[i][2]
            m = p if p.dtype == bool else p > (t[i] if isinstance(t, dict) else t)
            u = (m | y).sum()
            per[i] = (m & y).sum() / u if y.any() else np.nan
        dm, docs = doc_macro(per, self.exs)
        return dm, docs, per
