"""Nested selection over a restricted family (smoothing x hole fill), plus per-fold deltas of fixed candidates."""
import sys
import numpy as np
from explore_p4 import SPL, G
from explore_adapt import lift
from pp import chain4
from common import doc_macro, boot, fmt
import pp as P

C = {"none": lift(lambda g, st, gd=None: g), "fill": lift(P.fill_holes(1e6))}
for k in (20, 28, 36, 44, 60):
    C[f"med{k}"] = lift(P.median(k))
for k in (8, 11, 14, 17, 20):
    C[f"blur{k}"] = lift(P.blur(k))
for n in list(C):
    if n not in ("none", "fill"):
        C[f"{n}+fill"] = chain4(C[n], C["fill"])
        C[f"fill+{n}"] = chain4(C["fill"], C[n])
IOU = {s: {} for s in SPL}
for s, S in SPL.items():
    for i, (g, st) in G[s].items():
        e = S.exs[i]; y = S.dom[i][2]
        for name, fn in C.items():
            p = S.probs(i, fn(g, st, None, e["query_box"]), st) > 0.5
            IOU[s][(name, i)] = (p & y).sum() / (p | y).sum() if y.any() else np.nan


def dm(s, name, ids):
    return doc_macro({i: IOU[s][(name, i)] for i in ids}, SPL[s].exs)


fold = {i: f for i, _, f in SPL["train"].qs}
allq = list(fold); vq = list(G["val"])
base, vbase = dm("train", "none", allq)[1], dm("val", "none", vq)[1]
for n in sorted(C, key=lambda n: -dm("train", n, allq)[0])[:12]:
    pf = [dm("train", n, [i for i in fold if fold[i] == k])[0] - dm("train", "none", [i for i in fold if fold[i] == k])[0] for k in range(4)]
    print(f"{n:16s} cv {dm('train', n, allq)[0]:.4f} {fmt(boot(base, dm('train', n, allq)[1]))} per-fold {np.round(pf, 4)} | val {dm('val', n, vq)[0]:.4f} {fmt(boot(vbase, dm('val', n, vq)[1]))}")
per_doc = {}
for k in range(4):
    inner = [i for i in fold if fold[i] != k]
    best = max(C, key=lambda n: dm("train", n, inner)[0])
    print("fold", k, "picks", best)
    per_doc.update(dm("train", best, [i for i in fold if fold[i] == k])[1])
print("nested cv", round(np.mean(list(per_doc.values())), 4), fmt(boot(base, per_doc)))
