"""Oracle headroom of component-level keep/drop after fill+median, and per-query threshold oracle after it."""
import numpy as np
from scipy import ndimage as ndi
from explore1 import SPL, G, fill_holes, median, boot, fmt
from pixfeat import sample_grid

pp = lambda g, st: median(42)(fill_holes(100000)(g, st), st)
for s, S in SPL.items():
    P0, Porc, Pthr = {}, {}, {}
    for i, (g, st) in G[s].items():
        h = pp(g, st)
        ys, xs, y = S.dom[i]
        P0[i] = S.probs(i, h, st) > 0.5
        lab, n = ndi.label(h > 0.5)
        fy = np.clip(np.round((ys + 0.5) / st - 0.5), 0, h.shape[0] - 1).astype(int)
        fx = np.clip(np.round((xs + 0.5) / st - 0.5), 0, h.shape[1] - 1).astype(int)
        L = lab[fy, fx]
        m = P0[i].copy()
        for c in range(1, n + 1):
            sel = m & (L == c)
            if sel.any() and y[sel].sum() < (~y[sel]).sum():
                m[sel] = False
        Porc[i] = m
        pv = S.probs(i, h, st)
        best = max((((pv > t) & y).sum() / max(((pv > t) | y).sum(), 1), t) for t in np.arange(0.05, 0.96, 0.05))
        Pthr[i] = pv > best[1]
    b = S.score(P0); o = S.score(Porc); tq = S.score(Pthr)
    print(s, "pp", round(b[0], 4), "comp-oracle", round(o[0], 4), fmt(boot(b[1], o[1])), "| thr-oracle", round(tq[0], 4), fmt(boot(b[1], tq[1])))
