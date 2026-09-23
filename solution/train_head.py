"""Dev-only: fit the pixel head on train rows, report val doc-macro IoU."""
import sys, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib
sys.path.insert(0, str(Path(__file__).parent))
from devdata import examples, domain, score, ROOT


def load(split, tag):
    d = np.load(ROOT / "cache" / f"rows_{split}_{tag}.npz")
    return d["X"], d["Y"], d["Q"], list(d["names"])


def fit(X, Y, Q, kinds=None, real_only=False, params=None, rounds=400):
    keep = np.ones(len(Y), bool)
    if real_only:
        keep = np.array([kinds[q] == "real" for q in Q])
    X, Y, Q = X[keep], Y[keep], Q[keep]
    cnt = np.bincount(Q)
    w = 1.0 / cnt[Q]
    p = dict(learning_rate=0.06, max_iter=rounds, max_leaf_nodes=63, min_samples_leaf=200,
             l2_regularization=1.0, max_features=0.7, random_state=0)
    if params:
        p.update(params)
    return HistGradientBoostingClassifier(**p).fit(X, Y, sample_weight=w / w.mean())


def val_report(model, tag, ths=np.arange(0.2, 0.85, 0.05)):
    X, Y, Q, _ = load("val", tag)
    prob = model.predict_proba(X)[:, 1]
    exs = examples("val")
    best = None
    for t in ths:
        rows = [(e, domain(e), prob[Q == i] > t) for i, e in enumerate(exs)]
        r = score(rows)
        print(f"t={t:.2f} doc={r['doc_macro']:.4f} q={r['query_mean']:.4f} P={r['precision']:.3f} R={r['recall']:.3f}")
        if best is None or r["doc_macro"] > best[1]["doc_macro"]:
            best = (t, r)
    return prob, best


if __name__ == "__main__":
    tag = sys.argv[1]
    real_only = "--real" in sys.argv
    X, Y, Q, names = load("train", tag)
    kinds = [e["kind"] for e in examples("train")]
    m = fit(X, Y, Q, kinds, real_only)
    prob, best = val_report(m, tag)
    print("best", best[0], best[1]["doc_macro"])
    joblib.dump(m, ROOT / "cache" / f"head_{tag}{'_real' if real_only else ''}.joblib")
