"""Nested selection of a post-processing op: per fold, pick the candidate with best doc-macro on the other 3 folds'
OOF maps, score it on the held-out fold. Val uses the candidate picked on all train. python nested.py [--guide]"""
import sys
import numpy as np
from explore_p4 import SPL, G, GD
from explore_adapt import lift
from common import doc_macro, boot, fmt
import pp as P


def chain4(*fs):
    def f(g, st, gd, box):
        for h in fs:
            g = h(g, st, gd, box)
        return g
    return f


C = {"none": lift(lambda g, st, gd=None: g)}
for k in (20, 28, 44, 60):
    C[f"med{k}"] = lift(P.median(k))
for k in (8, 14, 20):
    C[f"blur{k}"] = lift(P.blur(k))
for c in (0.25, 0.4, 0.6):
    C[f"medbox{c}"] = P.box_scaled(P.median, c, 12, 200)
for c in (0.1, 0.15, 0.25):
    C[f"blurbox{c}"] = P.box_scaled(P.blur, c, 4, 60)
C["fill"] = lift(P.fill_holes(1e6))
for f in (0.25, 0.5, 1.0):
    C[f"minrel{f}"] = P.min_area_rel(f)
for sm in ("med28", "med44", "blur14", "blurbox0.15", "medbox0.4"):
    C[f"{sm}+fill"] = chain4(C[sm], C["fill"])
    for f in (0.5, 1.0):
        C[f"{sm}+fill+minrel{f}"] = chain4(C[sm], C["fill"], C[f"minrel{f}"])
if GD:
    for r in (16, 24, 32):
        C[f"guided{r}"] = lift(P.guided(r, 1e-2))
        C[f"guided{r}+fill+minrel0.5"] = chain4(C[f"guided{r}"], C["fill"], C["minrel0.5"])

IOU = {s: {} for s in SPL}
for s, S in SPL.items():
    for i, (g, st) in G[s].items():
        e = S.exs[i]; y = S.dom[i][2]
        for name, fn in C.items():
            p = S.probs(i, fn(g, st, GD.get(e["image"]), e["query_box"]), st) > 0.5
            IOU[s][(name, i)] = (p & y).sum() / (p | y).sum() if y.any() else np.nan
    print("scored", s, flush=True)


def dm(s, name, ids):
    return doc_macro({i: IOU[s][(name, i)] for i in ids}, SPL[s].exs)


tr = SPL["train"]; fold = {i: f for i, _, f in tr.qs}
full = {n: dm("train", n, list(fold))[0] for n in C}
va = {n: dm("val", n, list(G["val"]))[0] for n in C} if "val" in SPL else {}
for n in sorted(C, key=lambda n: -full[n]):
    b = boot(dm("train", "none", list(fold))[1], dm("train", n, list(fold))[1])
    print(f"{n:28s} train(all folds) {full[n]:.4f} {fmt(b)} | val {va.get(n, np.nan):.4f}")
per_doc = {}
for k in range(4):
    inner = [i for i in fold if fold[i] != k]; outer = [i for i in fold if fold[i] == k]
    best = max(C, key=lambda n: dm("train", n, inner)[0])
    print("fold", k, "picks", best, flush=True)
    per_doc.update(dm("train", best, outer)[1])
base = dm("train", "none", list(fold))[1]
print("cross-fitted selection cv (not strict nested CV)", round(np.mean(list(per_doc.values())), 4), "vs base", round(np.mean(list(base.values())), 4), fmt(boot(base, per_doc)))
if va:
    pick = max(C, key=lambda n: full[n])
    print("val pick", pick, round(va[pick], 4), "base", round(va["none"], 4),
          fmt(boot(dm("val", "none", list(G["val"]))[1], dm("val", pick, list(G["val"]))[1])))
