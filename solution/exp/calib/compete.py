"""Cross-query competition within an image: a pixel goes to query q only if p_q beats every non-alias query's p by margin.
Label-free but transductive (uses other requests on the same drawing). python compete.py SRC SPLITS"""
import sys
import numpy as np
from explore_p4 import SPL, G, base, report
import pp as P

pre = P.chain(P.median(28), P.fill_holes(1e6)) if "--pp" in sys.argv else (lambda g, st, gd=None: g)


def box_med(g, st, box):
    x0, y0, x1, y1 = [v // st for v in box]
    return float(np.median(g[y0:y1 + 1, x0:x1 + 1]))


def compete(margin, alias_t=0.5):
    out = {}
    for s, S in SPL.items():
        by = {}
        for i, e, _ in S.qs:
            by.setdefault(e["image"], []).append(i)
        Pq = {}
        for img, ids in by.items():
            H = {i: pre(G[s][i][0], G[s][i][1]) for i in ids}
            st = G[s][ids[0]][1]
            for i in ids:
                others = [j for j in ids if j != i and not (box_med(H[j], st, S.exs[i]["query_box"]) > alias_t
                                                               and box_med(H[i], st, S.exs[j]["query_box"]) > alias_t)]
                h = H[i]
                if others:
                    mx = np.max([H[j] for j in others], 0)
                    h = np.where(h + margin < mx, np.minimum(h, 0.49), h)
                Pq[i] = S.probs(i, h, st)
        out[s] = S.score(Pq)
    return out


if __name__ == "__main__":
    ref = {s: SPL[s].score({i: SPL[s].probs(i, pre(g, st), st) for i, (g, st) in G[s].items()}) for s in SPL}
    report("pre", ref)
    for k in base:
        base[k] = ref[k]
    for m in (0.0, 0.1, 0.3):
        for a in (0.5, 0.2):
            report(f"compete m{m} alias{a}", compete(m, a))
