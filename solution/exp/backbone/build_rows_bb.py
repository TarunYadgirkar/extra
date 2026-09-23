"""build_rows.py with configurable DINO grids. Usage: GRIDS=feat_b_0.5:28,feat_b_1.0:14 build_rows_bb.py SPLIT TAG [uniform]"""
import os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import build_rows
import pixfeat

build_rows.SCALES = [(d, float(s)) for d, s in (g.split(":") for g in os.environ["GRIDS"].split(","))]
ROW_CHUNK = 65536


def _topk_mean(flat, cells, k):
    out = np.empty(len(flat), np.float32)
    for i in range(0, len(flat), ROW_CHUNK):
        cs = flat[i:i + ROW_CHUNK] @ cells.T
        out[i:i + ROW_CHUNK] = np.sort(np.partition(cs, cs.shape[1] - k, axis=1)[:, -k:], axis=1).mean(1)
    return out


def sim_grids(ctx, protos):
    """pixfeat.sim_grids with top-k similarity computed in row chunks: same values, bounded memory for fine grids."""
    out = []
    for (f, s), p in zip(ctx.dino, protos):
        flat = f.reshape(-1, f.shape[-1])
        sm = (flat @ p["mean"]).reshape(f.shape[:2])
        sr = (flat @ p["rob"]).reshape(f.shape[:2])
        topk = _topk_mean(flat, p["cells"], min(3, len(p["cells"]))).reshape(f.shape[:2])
        wv, lo, hi = p["lda"]; wr, lor, hir = p["ldar"]
        lda = ((flat @ wv - lo) / (hi - lo + 1e-6)).reshape(f.shape[:2])
        ldar = ((flat @ wr - lor) / (hir - lor + 1e-6)).reshape(f.shape[:2])
        out.append((s, {"sm": sm, "sr": sr, "tk": topk, "lda": lda, "ldar": ldar}))
    return out


pixfeat.sim_grids = sim_grids

if __name__ == "__main__":
    build_rows.run(sys.argv[1], sys.argv[2], *sys.argv[3:])
