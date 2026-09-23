"""Headroom: component-level keep/drop oracle on the blur14+fill output. python oracle.py"""
import numpy as np
from vbase import queries, load_raw, pp, components, dom_cells, Dom, doc_macro, boot, fmt

for split in ("train", "val"):
    exs, qs = queries(split)
    base, orac, ncomp, nlab = {}, {}, [], []
    for i, e, _ in qs:
        h = pp(load_raw(e)); lab, n = components(h)
        D = Dom(e); m = D.mask(h)
        L = lab[dom_cells(D.ys, D.xs, h.shape)]; L[~m] = 0
        pos = np.bincount(L, D.y, n + 1); tot = np.bincount(L, minlength=n + 1)
        drop = (tot > 0) & (pos < 0.5 * tot); drop[0] = False
        base[i] = D.iou(m); orac[i] = D.iou(m & ~drop[L])
        ncomp.append(n); nlab.append(int((tot[1:] > 0).sum()))
    b, o = doc_macro(base, exs), doc_macro(orac, exs)
    print(split, "base", round(b[0], 4), "oracle", round(o[0], 4), fmt(boot(b[1], o[1])),
          "| comps/query median", np.median(ncomp), "max", max(ncomp), "labelled median", np.median(nlab), "total labelled", sum(nlab), flush=True)
