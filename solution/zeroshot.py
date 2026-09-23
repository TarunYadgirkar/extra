import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from devdata import examples, domain, score, ROOT


def query_cells(box, stride, gh, gw):
    x0, y0, x1, y1 = box
    gx0 = int(np.floor(x0 / stride)); gx1 = int(np.ceil(x1 / stride))
    gy0 = int(np.floor(y0 / stride)); gy1 = int(np.ceil(y1 / stride))
    # keep cells whose center falls in the box; fall back to the overlapping ones
    cy = (np.arange(gy0, gy1) + 0.5) * stride; cx = (np.arange(gx0, gx1) + 0.5) * stride
    iy = np.arange(gy0, gy1)[(cy >= y0) & (cy < y1)]; ix = np.arange(gx0, gx1)[(cx >= x0) & (cx < x1)]
    if len(iy) == 0: iy = np.arange(gy0, gy1)
    if len(ix) == 0: ix = np.arange(gx0, gx1)
    return np.clip(iy, 0, gh - 1), np.clip(ix, 0, gw - 1)


def bilinear(grid, stride, ys, xs):
    gh, gw = grid.shape[:2]
    fy = np.clip((ys + 0.5) / stride - 0.5, 0, gh - 1); fx = np.clip((xs + 0.5) / stride - 0.5, 0, gw - 1)
    y0 = np.floor(fy).astype(int); x0 = np.floor(fx).astype(int)
    y1 = np.minimum(y0 + 1, gh - 1); x1 = np.minimum(x0 + 1, gw - 1)
    wy = (fy - y0)[..., None] if grid.ndim == 3 else fy - y0
    wx = (fx - x0)[..., None] if grid.ndim == 3 else fx - x0
    return (grid[y0, x0] * (1 - wy) * (1 - wx) + grid[y0, x1] * (1 - wy) * wx
            + grid[y1, x0] * wy * (1 - wx) + grid[y1, x1] * wy * wx)


def sim_map(feat, stride, box, mode="mean"):
    f = feat.astype(np.float32)
    f /= np.linalg.norm(f, axis=-1, keepdims=True) + 1e-6
    iy, ix = query_cells(box, stride, *f.shape[:2])
    q = f[np.ix_(iy, ix)].reshape(-1, f.shape[-1])
    if mode == "mean":
        p = q.mean(0); p /= np.linalg.norm(p)
        return f @ p
    return (f.reshape(-1, f.shape[-1]) @ q.T).max(1).reshape(f.shape[:2])


if __name__ == "__main__":
    tag, stride, mode = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    exs = examples("val")
    sims = []
    for e in exs:
        feat = np.load(ROOT / "cache" / tag / f"{Path(e['image']).stem[:16]}.npy")
        d = domain(e)
        s = sim_map(feat, stride, e["query_box"], mode)
        sims.append((e, d, bilinear(s, stride, d["ys"], d["xs"])))
    for t in np.arange(0.3, 0.95, 0.05):
        r = score([(e, d, v > t) for e, d, v in sims])
        print(f"t={t:.2f} doc={r['doc_macro']:.4f} q={r['query_mean']:.4f} P={r['precision']:.3f} R={r['recall']:.3f}")
