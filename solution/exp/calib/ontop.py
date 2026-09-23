"""Variants on top of blur14+fill: threshold, hysteresis, cross-query competition (transductive)."""
import numpy as np
from explore_p4 import SPL, G
from explore_adapt import lift
from compete import box_med
from common import doc_macro, boot, fmt
import pp as P

BEST = P.chain4(lift(P.blur(14)), lift(P.fill_holes(1e6)))
H = {s: {i: BEST(g, st, None, SPL[s].exs[i]["query_box"]) for i, (g, st) in G[s].items()} for s in SPL}
ST = 4


def score(fn_map, t=0.5):
    return {s: S.score({i: S.probs(i, fn_map(s, i), ST) for i in G[s]}, t) for s, S in SPL.items()}


def rep(name, r, ref):
    print(f"{name:26s}", " | ".join(f"{s} {r[s][0]:.4f} {fmt(boot(ref[s][1], r[s][1]))}" for s in r), flush=True)


ref = score(lambda s, i: H[s][i])
rep("blur14+fill", ref, ref)
for t in (0.4, 0.45, 0.55, 0.6):
    rep(f"thr {t}", score(lambda s, i: H[s][i], t), ref)
for lo, hi in ((0.3, 0.8), (0.4, 0.9), (0.5, 0.9), (0.5, 0.97)):
    rep(f"hyst {lo}>{hi}", score(lambda s, i: P.hyst(lo, hi)(H[s][i], ST)), ref)


def compete_map(margin, alias_t):
    cache = {}
    for s, S in SPL.items():
        by = {}
        for i in G[s]:
            by.setdefault(S.exs[i]["image"], []).append(i)
        for ids in by.values():
            for i in ids:
                others = [j for j in ids if j != i and not (box_med(H[s][j], ST, S.exs[i]["query_box"]) > alias_t
                                                               and box_med(H[s][i], ST, S.exs[j]["query_box"]) > alias_t)]
                h = H[s][i]
                if others:
                    mx = np.max([H[s][j] for j in others], 0)
                    h = np.where(h + margin < mx, np.minimum(h, 0.49), h)
                cache[(s, i)] = h
    return lambda s, i: cache[(s, i)]


for m in (0.0, 0.1, 0.3):
    for a in (0.2, 0.5):
        rep(f"compete m{m} a{a}", score(compete_map(m, a)), ref)
