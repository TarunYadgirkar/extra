"""cv.py protocol that also saves out-of-fold probs for paired per-document comparison.
Usage: cv_oof.py TAG [--noval]     -> cache/bb_oof_TAG.npy
       cv_oof.py --compare TAG_A TAG_B"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train_head import load, fit
from devdata import examples, domain, score, ROOT
from cv import THS, doc_macro


def oof(tag, folds=4):
    X, Y, Q, _ = load("train", tag)
    Xu, Yu, Qu, _ = load("train_u", tag)
    exs = examples("train"); kinds = [e["kind"] for e in exs]
    docs = sorted({e["document_id"] for e in exs})
    rng = np.random.default_rng(0); rng.shuffle(docs)
    fold_of = {d: i % folds for i, d in enumerate(docs)}
    qfold = np.array([fold_of[e["document_id"]] for e in exs])
    prob = np.zeros(len(Yu), np.float32)
    for k in range(folds):
        tr = qfold[Q] != k
        m = fit(X[tr], Y[tr], Q[tr], kinds)
        te = qfold[Qu] == k
        prob[te] = m.predict_proba(Xu[te])[:, 1]
    np.save(ROOT / "cache" / f"bb_oof_{tag}.npy", prob)
    print(tag, "cv", np.round([doc_macro(prob, Yu, Qu, exs, t) for t in THS], 4), flush=True)
    return X, Y, Q, kinds


def per_doc(prob, Y, Q, exs, t=0.5):
    docs = {}
    for i in np.unique(Q):
        m = Q == i; p = prob[m] > t; y = Y[m]
        if y.any():
            docs.setdefault(exs[i]["document_id"], []).append((p & y).sum() / (p | y).sum())
    return {d: np.mean(v) for d, v in docs.items()}


def compare(a, b):
    _, Yu, Qu, _ = load("train_u", a)
    exs = examples("train")
    for t in THS:
        da = per_doc(np.load(ROOT / "cache" / f"bb_oof_{a}.npy"), Yu, Qu, exs, t)
        db = per_doc(np.load(ROOT / "cache" / f"bb_oof_{b}.npy"), Yu, Qu, exs, t)
        diff = np.array([db[d] - da[d] for d in da])
        rng = np.random.default_rng(0)
        boot = rng.choice(diff, (5000, len(diff))).mean(1)
        print(f"t={t} {b}-{a}: mean {diff.mean():+.4f}  95%CI [{np.quantile(boot, .025):+.4f},{np.quantile(boot, .975):+.4f}]"
              f"  better {int((diff > 0.005).sum())} worse {int((diff < -0.005).sum())} of {len(diff)}")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
        sys.exit()
    tag = sys.argv[1]
    X, Y, Q, kinds = oof(tag)
    if "--noval" not in sys.argv:
        m = fit(X, Y, Q, kinds)
        del X
        Xv, Yv, Qv, _ = load("val", tag)
        pv = m.predict_proba(Xv)[:, 1]
        vexs = examples("val")
        print(tag, "val", np.round([score([(e, domain(e), pv[Qv == i] > t) for i, e in enumerate(vexs)])["doc_macro"] for t in THS], 4), flush=True)
