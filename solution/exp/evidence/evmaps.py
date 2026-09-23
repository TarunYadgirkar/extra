"""Dev-only: full-image 4px val probability grids for an ev_ head (cached DINO-S grids, CPU), then score them
with and without the deployed blur14+fill post-processing against the deployed tx_v1f grids (cache/cal_p4).

python evmaps.py NAME          -> cache/ev_p4_NAME/{id}.npy, then prints val doc-macro (raw, pp) + paired bootstrap
python evmaps.py NAME --score  -> scoring only
"""
import sys, time
from pathlib import Path
import joblib
import numpy as np
HERE = Path(__file__).resolve().parent
SOL = HERE.parents[1]
sys.path.insert(0, str(SOL)); sys.path.insert(0, str(SOL / "exp" / "calib")); sys.path.insert(0, str(HERE))
from devdata import examples, domain, score, ROOT  # noqa: E402
from features import load_gray  # noqa: E402
from pixfeat import DOWN, ImageContext, features_at  # noqa: E402
from texfeat import TexContext, tex_query, tex_features_at  # noqa: E402
from infer import TEX_GROUPS  # noqa: E402
from evfeat import EvContext, ev_query, ev_features_at  # noqa: E402
import evcv  # noqa: E402
import pp as P  # noqa: E402

C = ROOT / "cache"
SCALES = [("feat_s_0.5", 28.0), ("feat_s_1.0", 14.0)]


def prob_grid(head, cfg, ctx, tctx, ectx, box, chunk=400_000):
    h, w = ctx.shape
    gh, gw = -(-h // DOWN), -(-w // DOWN)
    gy, gx = np.mgrid[0:gh, 0:gw]
    ys = (gy.ravel() * DOWN + DOWN // 2).astype(np.float32)
    xs = (gx.ravel() * DOWN + DOWN // 2).astype(np.float32)
    cache, prob = {}, np.empty(len(ys), np.float32)
    qc, eq = tex_query(tctx, box), ev_query(ectx, box)
    keep = None
    for i in range(0, len(ys), chunk):
        f, _ = features_at(ctx, box, ys[i:i + chunk], xs[i:i + chunk], cache)
        t, names = tex_features_at(tctx, qc, ys[i:i + chunk], xs[i:i + chunk])
        if keep is None:
            keep = [j for j, n in enumerate(names) if n.startswith(TEX_GROUPS)]
        e, _ = ev_features_at(eq, ys[i:i + chunk], xs[i:i + chunk])
        X, _ = evcv.design(cfg, np.concatenate([f, t[:, keep]], 1), e)
        prob[i:i + chunk] = head.predict_proba(X)[:, 1]
    return prob.reshape(gh, gw)


def maps(name):
    out = C / f"ev_p4_{name}"; out.mkdir(exist_ok=True)
    head = joblib.load(C / f"ev_head_{name}.joblib")["model"]
    cfg = evcv.CONFIGS[name]
    by = {}
    for e in examples("val"):
        by.setdefault(e["image"], []).append(e)
    t0 = time.time()
    for n, (img, group) in enumerate(by.items()):
        todo = [e for e in group if not (out / f"{e['id']}.npy").exists()]
        if not todo:
            continue
        stem = Path(img).stem[:16]
        gray = load_gray(ROOT / "dataset" / img)
        ctx = ImageContext(gray, [(np.load(C / d / f"{stem}.npy"), s) for d, s in SCALES])
        tctx, ectx = TexContext(gray), EvContext(gray)
        for e in todo:
            np.save(out / f"{e['id']}.npy", prob_grid(head, cfg, ctx, tctx, ectx, e["query_box"]).astype(np.float16))
        print(n + 1, len(by), f"{time.time()-t0:.0f}s", flush=True)


def val_score(grid_dir, post):
    rows = []
    for e in examples("val"):
        g = np.load(grid_dir / f"{e['id']}.npy").astype(np.float32)
        if post:
            g = P.fill_holes(1e6)(P.blur(14)(g, 4), 4)
        d = domain(e)
        gh, gw = g.shape
        # bilinear upsample to native exactly as infer.py, evaluated only at known pixels
        import cv2
        full = cv2.resize(g, (gw * DOWN, gh * DOWN), interpolation=cv2.INTER_LINEAR)
        rows.append((e, d, full[d["ys"], d["xs"]] > 0.5))
    return score(rows)


def per_doc(r):
    docs = {}
    ex = {e["id"]: e["document_id"] for e in examples("val")}
    for qid, iou, *_ in r["per"]:
        docs.setdefault(ex[qid], []).append(iou)
    return {d: np.mean(v) for d, v in docs.items()}


def report(name):
    for post in (False, True):
        a = val_score(C / "cal_p4", post)
        b = val_score(C / f"ev_p4_{name}", post)
        da, db = per_doc(a), per_doc(b)
        diff = np.array([db[k] - da[k] for k in da])
        bs = np.random.default_rng(0).choice(diff, (5000, len(diff))).mean(1)
        print(f"post={post} tx_v1f {a['doc_macro']:.4f} {name} {b['doc_macro']:.4f} delta {diff.mean():+.4f} "
              f"[{np.quantile(bs, .025):+.4f},{np.quantile(bs, .975):+.4f}] better {(diff > 0.005).sum()} worse {(diff < -0.005).sum()}",
              flush=True)
        qa = {q: i for q, i, *_ in a["per"]}
        for q, i, *_ in b["per"]:
            if abs(i - qa[q]) > 0.02:
                print(f"   {q} {qa[q]:.3f} -> {i:.3f}")


if __name__ == "__main__":
    if "--score" not in sys.argv:
        maps(sys.argv[1])
    report(sys.argv[1])
