"""Dev: full-image tx_v1f probability grids (4px, float16) from cached DINO-S grids, CPU only.

python gen_maps.py val            -> cache/cal_p4/{id}.npy with the tx_v1f all-train head
python gen_maps.py fit            -> cache/cal_head_f{0..3}.joblib (tx_v1f rows, cv.py folds/params)
python gen_maps.py train          -> cache/cal_p4/{id}.npy for train real queries from their held-out fold head
"""
import sys, time
from pathlib import Path
import joblib
import numpy as np
from common import CACHE, TXF, ROOT, SOL, examples, cv_split
sys.path.insert(0, str(SOL / "exp" / "texture"))
from features import load_gray  # noqa: E402
from pixfeat import DOWN, ImageContext, features_at  # noqa: E402
from txfeat import TexContext, tex_query, tex_features_at  # noqa: E402
from infer import TEX_GROUPS  # noqa: E402

OUT = CACHE / "cal_p4"
SCALES = [("feat_s_0.5", 28.0), ("feat_s_1.0", 14.0)]


def prob_grid(head, ctx, tctx, box, chunk=400_000):
    h, w = ctx.shape
    gh, gw = -(-h // DOWN), -(-w // DOWN)
    gy, gx = np.mgrid[0:gh, 0:gw]
    ys = (gy.ravel() * DOWN + DOWN // 2).astype(np.float32)
    xs = (gx.ravel() * DOWN + DOWN // 2).astype(np.float32)
    cache, prob = {}, np.empty(len(ys), np.float32)
    qc = tex_query(tctx, box, False)
    keep = None
    for i in range(0, len(ys), chunk):
        f, _ = features_at(ctx, box, ys[i:i + chunk], xs[i:i + chunk], cache)
        t, names = tex_features_at(tctx, qc, ys[i:i + chunk], xs[i:i + chunk])
        if keep is None:
            keep = [j for j, n in enumerate(names) if n.startswith(TEX_GROUPS)]
        prob[i:i + chunk] = head.predict_proba(np.concatenate([f, t[:, keep]], 1))[:, 1]
    return prob.reshape(gh, gw)


def fit_folds():
    sys.path.insert(0, str(SOL / "exp" / "texture"))
    import cv as cvmod
    from cv_tx import make_loader
    load = make_loader("tx_v2", list(TEX_GROUPS))
    exs, qfold = cv_split()
    kinds = [e["kind"] for e in exs]
    X, Y, Q, _ = load("train", "x")
    for k in range(4):
        p = CACHE / f"cal_head_f{k}.joblib"
        if p.exists():
            continue
        t = time.time(); m = qfold[Q] != k
        joblib.dump(cvmod.fit(X[m], Y[m], Q[m], kinds), p)
        print("fit", k, f"{time.time()-t:.0f}s", flush=True)


def maps(split):
    OUT.mkdir(parents=True, exist_ok=True)
    if split == "val":
        jobs = [(e, "all") for e in examples("val")]
    else:
        exs, qfold = cv_split()
        jobs = [(e, str(f)) for e, f in zip(exs, qfold) if e["kind"] == "real"]
    heads = {"all": CACHE / f"head_{TXF}.joblib", **{str(k): CACHE / f"cal_head_f{k}.joblib" for k in range(4)}}
    by = {}
    for e, k in jobs:
        by.setdefault(e["image"], []).append((e, k))
    loaded, t0 = {}, time.time()
    for n, (img, group) in enumerate(by.items()):
        todo = [(e, k) for e, k in group if not (OUT / f"{e['id']}.npy").exists()]
        if not todo:
            continue
        stem = Path(img).stem[:16]
        gray = load_gray(ROOT / "dataset" / img)
        ctx = ImageContext(gray, [(np.load(CACHE / d / f"{stem}.npy"), s) for d, s in SCALES])
        tctx = TexContext(gray)
        for e, k in todo:
            if k not in loaded:
                loaded[k] = joblib.load(heads[k])
            g = prob_grid(loaded[k], ctx, tctx, e["query_box"])
            np.save(OUT / f"{e['id']}.npy", g.astype(np.float16))
        print(split, n + 1, len(by), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    {"fit": fit_folds, "val": lambda: maps("val"), "train": lambda: maps("train")}[sys.argv[1]]()
