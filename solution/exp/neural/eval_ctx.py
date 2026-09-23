"""Dev: score an auto-context (ctx=1 resid=1) run. Final pixel logit = tx_v1 GBM pixel logit + upsampled s14 net delta.

python eval_ctx.py valprobs        -> cache/nn_gbm/pval_tx_v1.npy (tx_v1 'all' model on every known val pixel)
python eval_ctx.py TAG [FOLD]      -> cv (all folds, or one) + val (if the 'all' run exists), plus paired bootstrap vs tx_v1
"""
import sys
from pathlib import Path
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_nn as E  # noqa: E402
from gbm_maps import OUT, logit  # noqa: E402

GBM14 = OUT / "tx_v1"


def delta_col(tag, split, mode, keep=None):
    exs = E.examples(split)
    cols = []
    for q, idx in E.row_index(split, mode):
        e = exs[q]; d = E.domain(e)
        if keep is not None and not keep[q]:
            cols.append(np.full(len(idx), np.nan, np.float32)); continue
        z = np.load(E.PRED / tag / f"{e['id']}.npy").astype(np.float32)
        g = np.clip(np.load(GBM14 / f"{e['id']}.npy").astype(np.float32), -10, 10)
        cols.append(E.sample_grid(z - g, E.S, d["ys"][idx].astype(np.float32), d["xs"][idx].astype(np.float32)))
    return np.concatenate(cols)


def valprobs():
    X, _, _, _ = E.load("val", "v3+tx_v1")
    p = joblib.load(OUT / "tx_v1_fall.joblib").predict_proba(X)[:, 1].astype(np.float32)
    np.save(OUT / "pval_tx_v1.npy", p)


def main(tag, folds):
    exs, qfold = E.cv_split()
    _, Yu, Qu, _ = E.load_rows("train_u", "v3")
    base = np.load(E.ROOT / "cache" / "bb_oof_bb_tx.npy")
    keep = np.isin(qfold, folds)
    sel = keep[Qu]
    dz = delta_col(tag, "train", "uniform", keep)
    p = E.sig(logit(base) + np.nan_to_num(dz))
    for t in E.THS:
        print(f"{tag} folds {folds} t={t} cv {E.doc_macro(p[sel], Yu[sel], Qu[sel], exs, t):.4f} "
              f"base {E.doc_macro(base[sel], Yu[sel], Qu[sel], exs, t):.4f}", flush=True)
    print("delta vs tx_v1 @0.5, 95% CI", E.bootstrap(base[sel], p[sel], Yu[sel], Qu[sel], exs), flush=True)
    np.save(E.ROOT / "cache" / f"nn_oof_{tag}_resid.npy", p)
    vexs = E.examples("val")
    if all((E.PRED / tag / f"{e['id']}.npy").exists() for e in vexs):
        _, Yv, Qv, _ = E.load_rows("val", "v3")
        pv0 = np.load(OUT / "pval_tx_v1.npy")
        pv = E.sig(logit(pv0) + delta_col(tag, "val", "balanced"))
        for t in E.THS:
            v = E.score([(e, E.domain(e), pv[Qv == i] > t) for i, e in enumerate(vexs)])["doc_macro"]
            v0 = E.score([(e, E.domain(e), pv0[Qv == i] > t) for i, e in enumerate(vexs)])["doc_macro"]
            print(f"{tag} t={t} val {v:.4f} base {v0:.4f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "valprobs":
        valprobs()
    else:
        main(sys.argv[1], [int(sys.argv[2])] if len(sys.argv) > 2 else [0, 1, 2, 3])
