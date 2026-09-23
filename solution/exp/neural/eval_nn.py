"""Dev: score cached nn logit maps with the cv.py protocol, alone and stacked into the v3 GBM rows.

python eval_nn.py TAG [--stack] [--base v3]
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nn_common import ROOT, S  # noqa: E402
sys.path.insert(0, str(ROOT / "solution"))
from devdata import examples, domain, score  # noqa: E402
from pixfeat import sample_grid  # noqa: E402
from cv import doc_macro, THS  # noqa: E402
from train_head import load as load_rows, fit  # noqa: E402
from build_rows import PER_CLASS, UNIFORM  # noqa: E402

PRED = ROOT / "cache" / "nn_pred"


def row_index(split, mode="balanced"):
    """Replays build_rows.run sampling: returns list of (query index, domain index array) in row order."""
    rng = np.random.default_rng(0)
    exs = examples(split)
    by_img = {}
    for e in exs:
        by_img.setdefault(e["image"], []).append(e)
    out = []
    for group in by_img.values():
        for e in group:
            d = domain(e)
            idx = np.arange(len(d["ys"]))
            if mode == "uniform":
                if e["kind"] != "real":
                    continue
                idx = rng.choice(idx, min(UNIFORM, len(idx)), replace=False)
            elif split == "train":
                pos, neg = idx[d["pos"]], idx[~d["pos"]]
                idx = np.concatenate([rng.choice(pos, min(PER_CLASS, len(pos)), replace=False),
                                      rng.choice(neg, min(PER_CLASS, len(neg)), replace=False)])
            out.append((exs.index(e), idx))
    return out


def column(tag, split, mode="balanced"):
    exs = examples(split)
    cols = []
    for qi, idx in row_index(split, mode):
        e = exs[qi]; d = domain(e)
        g = np.load(PRED / tag / f"{e['id']}.npy").astype(np.float32)
        cols.append(sample_grid(g, S, d["ys"][idx].astype(np.float32), d["xs"][idx].astype(np.float32)))
    return np.concatenate(cols)


def load(split, tags):
    """tags: '+'-joined row tags (e.g. v3+tx_v1), columns concatenated, rows must align."""
    parts = tags.split("+")
    X, Y, Q, names = load_rows(split, parts[0])
    for t in parts[1:]:
        T, Ty, _, tn = load_rows(split, t)
        assert np.array_equal(Ty, Y), f"row misalignment {split} {t}"
        X = np.c_[X, T]; names = names + tn
    return X, Y, Q, names


def per_doc(prob, Y, Q, exs, t=0.5):
    docs = {}
    for i in np.unique(Q):
        m = Q == i; p = prob[m] > t; y = Y[m]
        if y.any():
            docs.setdefault(exs[i]["document_id"], []).append((p & y).sum() / (p | y).sum())
    return {d: np.mean(v) for d, v in docs.items()}


def bootstrap(pa, pb, Yu, Qu, exs, t=0.5, n=5000):
    """Paired per-doc bootstrap of doc-macro(b) - doc-macro(a): mean delta and 95% CI."""
    da, db = per_doc(pa, Yu, Qu, exs, t), per_doc(pb, Yu, Qu, exs, t)
    diff = np.array([db[d] - da[d] for d in da])
    rng = np.random.default_rng(0)
    bs = diff[rng.integers(len(diff), size=(n, len(diff)))].mean(1)
    return round(float(diff.mean()), 4), np.round(np.quantile(bs, [0.025, 0.975]), 4)


def sig(z):
    return 1 / (1 + np.exp(-z))


def cv_split():
    exs = examples("train")
    docs = sorted({e["document_id"] for e in exs})
    rng = np.random.default_rng(0); rng.shuffle(docs)
    fo = {d: i % 4 for i, d in enumerate(docs)}
    return exs, np.array([fo[e["document_id"]] for e in exs])


def alone(tag):
    exs, _ = cv_split()
    _, Yu, Qu, _ = load("train_u", tag_base)
    pu = sig(column(tag, "train", "uniform"))
    cvs = [doc_macro(pu, Yu, Qu, exs, t) for t in THS]
    vexs = examples("val")
    _, Yv, Qv, _ = load("val", tag_base)
    pv = sig(column(tag, "val"))
    assert len(pv) == len(Yv)
    vals = [score([(e, domain(e), pv[Qv == i] > t) for i, e in enumerate(vexs)])["doc_macro"] for t in THS]
    return np.round(cvs, 4), np.round(vals, 4)


def stacked(tag, params=None, rounds=400):
    exs, qfold = cv_split()
    kinds = [e["kind"] for e in exs]
    X, Y, Q, _ = load("train", tag_base)
    Xu, Yu, Qu, _ = load("train_u", tag_base)
    Xv, Yv, Qv, _ = load("val", tag_base)
    X = np.c_[X, column(tag, "train")]; Xu = np.c_[Xu, column(tag, "train", "uniform")]; Xv = np.c_[Xv, column(tag, "val")]
    prob = np.zeros(len(Yu))
    for k in range(4):
        m = fit(X[qfold[Q] != k], Y[qfold[Q] != k], Q[qfold[Q] != k], kinds, params=params, rounds=rounds)
        te = qfold[Qu] == k
        prob[te] = m.predict_proba(Xu[te])[:, 1]
    np.save(ROOT / "cache" / f"nn_oof_{tag}_on_{tag_base}.npy", prob)
    cvs = [doc_macro(prob, Yu, Qu, exs, t) for t in THS]
    del Xu
    m = fit(X, Y, Q, kinds, params=params, rounds=rounds)
    del X
    pv = m.predict_proba(Xv)[:, 1]
    vexs = examples("val")
    vals = [score([(e, domain(e), pv[Qv == i] > t) for i, e in enumerate(vexs)])["doc_macro"] for t in THS]
    return np.round(cvs, 4), np.round(vals, 4), m


def check_rows():
    for split, mode in (("train", "balanced"), ("train", "uniform"), ("val", "balanced")):
        ri = row_index(split, mode)
        _, Y, Q, _ = load(split + ("_u" if mode == "uniform" else ""), tag_base)
        exs = examples(split)
        Qr = np.concatenate([np.full(len(i), q) for q, i in ri])
        Yr = np.concatenate([domain(exs[q])["pos"][i] for q, i in ri])
        print(split, mode, "rows match:", np.array_equal(Qr, Q) and np.array_equal(Yr, Y), flush=True)


tag_base = "v3+tx_v1"
BASE_OOF = {"v3": "bb_oof_v3.npy", "v3+tx_v1": "bb_oof_bb_tx.npy"}
if __name__ == "__main__":
    if "--base" in sys.argv:
        tag_base = sys.argv[sys.argv.index("--base") + 1]
    if sys.argv[1] == "check":
        check_rows(); sys.exit()
    tag = sys.argv[1]
    c, v = alone(tag)
    print(tag, "alone cv", c, "val", v, flush=True)
    if "--stack" in sys.argv:
        c, v, m = stacked(tag)
        print(tag, "stack cv", c, "val", v, flush=True)
        exs, _ = cv_split()
        _, Yu, Qu, _ = load("train_u", tag_base)
        pa = np.load(ROOT / "cache" / BASE_OOF[tag_base])
        print("base oof cv@0.5", round(doc_macro(pa, Yu, Qu, exs, 0.5), 4), "stack-base delta, 95% CI",
              bootstrap(pa, np.load(ROOT / "cache" / f"nn_oof_{tag}_on_{tag_base}.npy"), Yu, Qu, exs), flush=True)
