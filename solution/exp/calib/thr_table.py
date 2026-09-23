"""Per-query oracle threshold vs label-free map stats (after fill+median pp)."""
import numpy as np
from explore1 import SPL, G, fill_holes, median
pp = lambda g, st: median(42)(fill_holes(100000)(g, st), st)
TS = np.round(np.arange(0.05, 0.96, 0.05), 2)
np.set_printoptions(linewidth=200)
for s, S in SPL.items():
    print("==", s)
    for i, (g, st) in G[s].items():
        h = pp(g, st); ys, xs, y = S.dom[i]
        if not y.any():
            continue
        pv = S.probs(i, h, st)
        ious = np.array([((pv > t) & y).sum() / max(((pv > t) | y).sum(), 1) for t in TS])
        k5 = list(TS).index(0.5)
        if ious.max() - ious[k5] < 0.03:
            continue
        e = S.exs[i]; x0, y0, x1, y1 = [v // st for v in e["query_box"]]
        qb = np.median(h[y0:y1 + 1, x0:x1 + 1])
        hi = h[h > 0.5]
        print(e["document_id"][4:10], e["id"][-6:], f"best t={TS[ious.argmax()]:.2f} gain {ious.max()-ious[k5]:+.3f} iou@.5 {ious[k5]:.3f}",
              f"| qbox {qb:.2f} frac>.5 {np.mean(h>.5):.3f} frac>.9 {np.mean(h>.9):.3f} q10/50 of >.5: {np.percentile(hi,10) if len(hi) else 0:.2f}/{np.percentile(hi,50) if len(hi) else 0:.2f}",
              "iou(t):", (ious[::2] * 100).astype(int))
