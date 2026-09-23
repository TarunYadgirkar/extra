"""Dev-only: cv.py protocol on v3 rows + texture columns.

Usage: cv_tx.py TXTAG GROUPS [--deep]   GROUPS: comma list of prefixes to keep, e.g. s0_,s1_,qs,b,qb,ncc,q_  or 'all'
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv as cvmod  # noqa: E402
from train_head import load as load_rows  # noqa: E402

BASE = "v3"


def make_loader(txtag, groups):
    def load(split, tag):
        X, Y, Q, names = load_rows(split, BASE)
        T, Ty, _, tnames = load_rows(split, txtag)
        assert len(Ty) == len(Y) and (Ty == Y).all(), f"row misalignment in {split}"
        keep = [i for i, n in enumerate(tnames) if groups == ["all"] or any(n.startswith(g) for g in groups)]
        out = np.empty((len(Y), X.shape[1] + len(keep)), np.float32)
        out[:, :X.shape[1]] = X
        del X
        out[:, -len(keep):] = T[:, keep]
        return out, Y, Q, names + [tnames[i] for i in keep]
    return load


def cv_oof(tag, params, rounds, name, folds=4):
    """cv.cv with out-of-fold probs saved to cache/tx_oof_NAME.npy (paired bootstrap via exp/backbone/cv_oof.py)."""
    from devdata import examples, domain, score, ROOT
    X, Y, Q, _ = cvmod.load("train", tag)
    Xu, Yu, Qu, _ = cvmod.load("train_u", tag)
    exs = examples("train"); kinds = [e["kind"] for e in exs]
    docs = sorted({e["document_id"] for e in exs})
    rng = np.random.default_rng(0); rng.shuffle(docs)
    fold_of = {d: i % folds for i, d in enumerate(docs)}
    qfold = np.array([fold_of[e["document_id"]] for e in exs])
    prob = np.zeros(len(Yu), np.float32)
    for k in range(folds):
        tr = qfold[Q] != k
        m = cvmod.fit(X[tr], Y[tr], Q[tr], kinds, params=params, rounds=rounds)
        te = qfold[Qu] == k
        prob[te] = m.predict_proba(Xu[te])[:, 1]
    np.save(ROOT / "cache" / f"tx_oof_{name}.npy", prob)
    cvs = [cvmod.doc_macro(prob, Yu, Qu, exs, t) for t in cvmod.THS]
    del Xu
    m = cvmod.fit(X, Y, Q, kinds, params=params, rounds=rounds)
    del X
    Xv, Yv, Qv, _ = cvmod.load("val", tag)
    pv = m.predict_proba(Xv)[:, 1]
    np.save(ROOT / "cache" / f"tx_pval_{name}.npy", pv)
    vexs = examples("val")
    vals = [score([(e, domain(e), pv[Qv == i] > t) for i, e in enumerate(vexs)])["doc_macro"] for t in cvmod.THS]
    return np.round(cvs, 4), np.round(vals, 4), m


if __name__ == "__main__":
    txtag, groups = sys.argv[1], sys.argv[2].split(",")
    cvmod.load = make_loader(txtag, groups)
    name = txtag + "_" + sys.argv[2].replace(",", "+") + ("_deep" if "--deep" in sys.argv else "")
    params, rounds = ({}, 400) if "--deep" not in sys.argv else (dict(max_leaf_nodes=127, min_samples_leaf=100), 600)
    c, v, m = cv_oof(txtag, params, rounds, name)
    print(txtag, sys.argv[2], "cv", c, "val", v, flush=True)
    if "--save" in sys.argv:
        import joblib
        from devdata import ROOT
        joblib.dump(m, ROOT / "cache" / f"head_{name}.joblib")
