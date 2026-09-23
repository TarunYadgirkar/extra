"""Dev: dense out-of-fold tx_v1 GBM logit maps on the stride-14 grid (auto-context input for the neural head).

python gbm_maps.py fit      -> cache/nn_gbm/tx_v1_f{0..3,all}.joblib (same rows/params/folds as cv.py)
python gbm_maps.py maps     -> cache/nn_gbm/tx_v1/{id}.npy: train queries from their fold's model, val from 'all'
"""
import sys, time
from pathlib import Path
import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "texture"))
import eval_nn as E  # noqa: E402
from nn_common import ROOT, S  # noqa: E402
from features import load_gray  # noqa: E402
from pixfeat import ImageContext, features_at  # noqa: E402
from txfeat import TexContext, tex_query, tex_features_at  # noqa: E402

OUT = ROOT / "cache" / "nn_gbm"
BASE = "v3+tx_v1"


def fit_all():
    OUT.mkdir(parents=True, exist_ok=True)
    exs, qfold = E.cv_split()
    kinds = [e["kind"] for e in exs]
    X, Y, Q, _ = E.load("train", BASE)
    for k in (0, 1, 2, 3, "all"):
        p = OUT / f"tx_v1_f{k}.joblib"
        if p.exists():
            continue
        m = qfold[Q] != k if k != "all" else np.ones(len(Y), bool)
        t = time.time()
        joblib.dump(E.fit(X[m], Y[m], Q[m], kinds), p)
        print("fit", k, f"{time.time()-t:.0f}s", flush=True)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def dense_maps():
    out = OUT / "tx_v1"; out.mkdir(parents=True, exist_ok=True)
    exs, qfold = E.cv_split()
    jobs = [(e, str(f)) for e, f in zip(exs, qfold)] + [(e, "all") for e in E.examples("val")]
    models = {k: joblib.load(OUT / f"tx_v1_f{k}.joblib") for k in ("0", "1", "2", "3", "all")}
    by = {}
    for e, k in jobs:
        by.setdefault(e["image"], []).append((e, k))
    t = time.time()
    for n, (img, group) in enumerate(by.items()):
        if all((out / f"{e['id']}.npy").exists() for e, _ in group):
            continue
        stem = Path(img).stem[:16]
        gray = load_gray(ROOT / "dataset" / img)
        dino = [(np.load(ROOT / "cache" / d / f"{stem}.npy"), s) for d, s in (("feat_s_0.5", 28.0), ("feat_s_1.0", 14.0))]
        ctx = ImageContext(gray, dino); tctx = TexContext(gray)
        gh, gw = dino[1][0].shape[:2]
        gy, gx = np.mgrid[0:gh, 0:gw]
        ys = (gy.ravel() * S + S // 2).astype(np.float32); xs = (gx.ravel() * S + S // 2).astype(np.float32)
        for e, k in group:
            f, _ = features_at(ctx, e["query_box"], ys, xs)
            tx, _ = tex_features_at(tctx, tex_query(tctx, e["query_box"], False), ys, xs)
            p = models[k].predict_proba(np.concatenate([f, tx], 1))[:, 1]
            np.save(out / f"{e['id']}.npy", logit(p).reshape(gh, gw).astype(np.float16))
        print(n, len(by), f"{time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    {"fit": fit_all, "maps": dense_maps}[sys.argv[1]]()
