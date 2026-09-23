"""Nested per-query threshold selection from label-free map stats.

For each query: IoU curve over candidate thresholds (after pp). A regressor predicts IoU(t) - IoU(0.5) from
map stats + t; pick argmax if predicted gain > margin. Trained on train folds != k, applied to fold k; val uses all train.
"""
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from explore1 import SPL, G, fill_holes, median
from common import boot, fmt

src_pp = "pp" if "--raw" not in sys.argv else "raw"
pp = (lambda g, st: median(42)(fill_holes(100000)(g, st), st)) if src_pp == "pp" else (lambda g, st: g)
TS = np.round(np.arange(0.1, 0.91, 0.1), 2)
K5 = list(TS).index(0.5)


def stats(h, e, st):
    x0, y0, x1, y1 = [v // st for v in e["query_box"]]
    qb = h[y0:y1 + 1, x0:x1 + 1]
    fr = [np.mean(h > t) for t in (0.1, 0.3, 0.5, 0.7, 0.9, 0.97)]
    hi = h[h > 0.5]
    qs = np.percentile(hi, [10, 25, 50, 75]) if len(hi) > 10 else np.zeros(4)
    qarea = qb.size
    return np.r_[np.log(np.array(fr) + 1e-6), qs, np.mean(qb), np.percentile(qb, 10), np.log(fr[2] * h.size / qarea + 1e-3)]


D = {}
for s, S in SPL.items():
    for i, (g, st) in G[s].items():
        h = pp(g, st); ys, xs, y = S.dom[i]
        pv = S.probs(i, h, st)
        tp = np.array([((pv > t) & y).sum() for t in TS]); fp = np.array([((pv > t) & ~y).sum() for t in TS])
        fn = y.sum() - tp
        D[(s, i)] = (stats(h, S.exs[i], st), tp, fp, fn, y.any())

folds = {i: f for i, _, f in SPL["train"].qs}


def design(keys):
    X, Yg = [], []
    for k in keys:
        f, tp, fp, fn, ok = D[k]
        if not ok:
            continue
        iou = tp / np.maximum(tp + fp + fn, 1)
        for j, t in enumerate(TS):
            X.append(np.r_[f, t]); Yg.append(iou[j] - iou[K5])
    return np.array(X), np.array(Yg)


def choose(model, keys, margin):
    out = {}
    for k in keys:
        f = D[k][0]
        g = model.predict(np.c_[np.tile(f, (len(TS), 1)), TS])
        j = int(np.argmax(g))
        out[k] = j if g[j] > margin else K5
    return out


def score_choice(s, ch):
    S = SPL[s]; per = {}
    for (ss, i), j in ch.items():
        _, tp, fp, fn, ok = D[(ss, i)]
        per[i] = tp[j] / (tp[j] + fp[j] + fn[j]) if ok else np.nan
    from common import doc_macro
    return doc_macro(per, S.exs)


tr_keys = [("train", i) for i in folds]
va_keys = [("val", i) for i, _, _ in SPL["val"].qs]
base_tr = score_choice("train", {k: K5 for k in tr_keys}); base_va = score_choice("val", {k: K5 for k in va_keys})
print(src_pp, "base", round(base_tr[0], 4), round(base_va[0], 4))
P = dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=40, l2_regularization=1.0)
for margin in (0.0, 0.02, 0.05, 0.1):
    ch = {}
    for k in range(4):
        X, Y = design([kk for kk in tr_keys if folds[kk[1]] != k])
        m = HistGradientBoostingRegressor(**P, random_state=0).fit(X, Y)
        ch.update(choose(m, [kk for kk in tr_keys if folds[kk[1]] == k], margin))
    r = score_choice("train", ch)
    X, Y = design(tr_keys)
    m = HistGradientBoostingRegressor(**P, random_state=0).fit(X, Y)
    rv = score_choice("val", choose(m, va_keys, margin))
    print(f"margin {margin}: cv {r[0]:.4f} {fmt(boot(base_tr[1], r[1]))} | val {rv[0]:.4f} {fmt(boot(base_va[1], rv[1]))}", flush=True)
