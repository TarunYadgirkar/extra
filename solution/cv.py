"""Dev-only: document-grouped CV on train real queries + val score, for model selection."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from train_head import load, fit
from devdata import examples, domain, score

THS = (0.4, 0.5, 0.6, 0.7)


def doc_macro(prob, Y, Q, exs, t):
    docs = {}
    for i in np.unique(Q):
        m = Q == i; p = prob[m] > t; y = Y[m]
        if y.any():
            docs.setdefault(exs[i]["document_id"], []).append((p & y).sum() / (p | y).sum())
    return np.mean([np.mean(v) for v in docs.values()])


def cv(tag, params=None, rounds=400, folds=4, cols=None):
    X, Y, Q, names = load("train", tag)
    Xu, Yu, Qu, _ = load("train_u", tag)
    if cols is not None:
        X, Xu = X[:, cols], Xu[:, cols]
    exs = examples("train"); kinds = [e["kind"] for e in exs]
    docs = sorted({e["document_id"] for e in exs})
    rng = np.random.default_rng(0); rng.shuffle(docs)
    fold_of = {d: i % folds for i, d in enumerate(docs)}
    qfold = np.array([fold_of[e["document_id"]] for e in exs])
    prob = np.zeros(len(Yu))
    for k in range(folds):
        m = fit(X[qfold[Q] != k], Y[qfold[Q] != k], Q[qfold[Q] != k], kinds, params=params, rounds=rounds)
        te = qfold[Qu] == k
        prob[te] = m.predict_proba(Xu[te])[:, 1]
    cvs = [doc_macro(prob, Yu, Qu, exs, t) for t in THS]
    m = fit(X, Y, Q, kinds, params=params, rounds=rounds)
    Xv, Yv, Qv, _ = load("val", tag)
    if cols is not None:
        Xv = Xv[:, cols]
    pv = m.predict_proba(Xv)[:, 1]
    vexs = examples("val")
    vals = [score([(e, domain(e), pv[Qv == i] > t) for i, e in enumerate(vexs)])["doc_macro"] for t in THS]
    return np.round(cvs, 4), np.round(vals, 4), m


if __name__ == "__main__":
    tag = sys.argv[1]
    for name, params, rounds in [("base", {}, 400), ("deep", dict(max_leaf_nodes=127, min_samples_leaf=100), 600),
                                 ("reg", dict(l2_regularization=10, min_samples_leaf=500, max_features=0.5), 600)]:
        c, v, _ = cv(tag, params, rounds)
        print(name, "cv", c, "val", v, flush=True)
