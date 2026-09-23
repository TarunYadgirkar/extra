"""Query-box-scaled post-processing. python explore_adapt.py SRC SPLITS"""
from explore_p4 import SPL, G, GD, report
import pp as P


def run2(fn, t=0.5):
    out = {}
    for s, S in SPL.items():
        Pq = {i: S.probs(i, fn(g, st, GD.get(S.exs[i]["image"]), S.exs[i]["query_box"]), st) for i, (g, st) in G[s].items()}
        out[s] = S.score(Pq, t)
    return out


def lift(op):
    return lambda g, st, gd, box: op(g, st, gd)


if __name__ == "__main__":
    C = [(f"med{k}", lift(P.median(k))) for k in (28, 44, 60, 80)]
    C += [(f"blur{k}", lift(P.blur(k))) for k in (14, 20, 28)]
    C += [(f"med box*{c}", P.box_scaled(P.median, c, 12, 200)) for c in (0.4, 0.6, 0.8, 1.0, 1.3)]
    C += [(f"blur box*{c}", P.box_scaled(P.blur, c, 4, 60)) for c in (0.15, 0.25, 0.35)]
    C += [(f"minarea rel{f}", P.min_area_rel(f)) for f in (0.25, 1.0)]
    C += [("med box*.8+minrel.5", lambda g, st, gd, b: P.min_area_rel(0.5)(P.box_scaled(P.median, 0.8, 12, 200)(g, st, gd, b), st, gd, b))]
    for name, fn in C:
        report(name, run2(fn))
