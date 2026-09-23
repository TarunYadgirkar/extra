"""Sweep post-processing ops on full-image prob grids. python explore_p4.py SRC SPLITS [--guide]"""
import sys
import numpy as np
from maps import Scorer, load_grid
from common import ROOT, boot, fmt
import pp as P

SRC = sys.argv[1] if len(sys.argv) > 1 else "p4"
SPLITS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["val"]
SPL = {s: Scorer(s) for s in SPLITS}
G, GD = {}, {}
for s, S in SPL.items():
    G[s] = {}
    for i, e, _ in S.qs:
        try:
            G[s][i] = load_grid(SRC, e)
        except FileNotFoundError:
            pass
    S.qs = [q for q in S.qs if q[0] in G[s]]
if "--guide" in sys.argv:
    from features import load_gray
    for s, S in SPL.items():
        for i, e, _ in S.qs:
            if e["image"] not in GD:
                GD[e["image"]] = P.ink_guide(load_gray(ROOT / "dataset" / e["image"]), G[s][i][1])


def run(fn, t=0.5):
    out = {}
    for s, S in SPL.items():
        Pq = {i: S.probs(i, fn(g, st, GD.get(S.exs[i]["image"])), st) for i, (g, st) in G[s].items()}
        out[s] = S.score(Pq, t)
    return out


base = run(lambda g, st, gd=None: g)


def report(name, r):
    print(f"{name:28s}", " | ".join(f"{s} {r[s][0]:.4f} {fmt(boot(base[s][1], r[s][1]))}" for s in r), flush=True)


if __name__ == "__main__":
    print(SRC, {s: (len(G[s]), round(v[0], 4)) for s, v in base.items()})
    for t in (0.4, 0.6):
        report(f"thr {t}", run(lambda g, st, gd=None: g, t))
    C = [(f"median{k}", P.median(k)) for k in (12, 20, 28, 44, 60)]
    C += [(f"blur{k}", P.blur(k)) for k in (4, 8, 14, 20)]
    C += [(f"fill{a}", P.fill_holes(a)) for a in (2000, 10000, 40000, 1e6)]
    C += [(f"fillb{a}", P.fill_holes_b(a)) for a in (40000, 1e6)]
    C += [("med28+fill1e6", P.chain(P.median(28), P.fill_holes(1e6))), ("fill1e6+med28", P.chain(P.fill_holes(1e6), P.median(28))),
          ("med44+fill1e6", P.chain(P.median(44), P.fill_holes(1e6))), ("blur8+fill1e6", P.chain(P.blur(8), P.fill_holes(1e6))),
          ("med20+fill40k", P.chain(P.median(20), P.fill_holes(40000)))]
    if GD:
        C += [(f"guided{r},{e}", P.guided(r, e)) for r in (16, 32, 64) for e in (1e-3, 1e-2)]
    for name, fn in C:
        report(name, run(fn))
