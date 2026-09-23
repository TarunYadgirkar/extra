"""Inference with v3 features + texture columns.

python solution/exp/texture/infer_tx.py --inputs X --data-root D --output-dir O --head H --groups all
"""
import argparse, json, sys, time
from pathlib import Path
import cv2
import joblib
import numpy as np
from PIL import Image
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from features import load_model, load_gray, dense_features  # noqa: E402
from pixfeat import DOWN, ImageContext, features_at  # noqa: E402
from infer import SCALES, INPUT_KEYS  # noqa: E402
from txfeat import TexContext, tex_query, tex_features_at  # noqa: E402


def predict_mask(head, ctx, tctx, box, threshold, groups, extra=False, chunk=400_000):
    h, w = ctx.shape
    gh, gw = -(-h // DOWN), -(-w // DOWN)
    gy, gx = np.mgrid[0:gh, 0:gw]
    ys = (gy.ravel() * DOWN + DOWN // 2).astype(np.float32)
    xs = (gx.ravel() * DOWN + DOWN // 2).astype(np.float32)
    cache, prob = {}, np.empty(len(ys), np.float32)
    qc = tex_query(tctx, box, extra)
    keep = None
    for i in range(0, len(ys), chunk):
        f, _ = features_at(ctx, box, ys[i:i + chunk], xs[i:i + chunk], cache)
        t, tnames = tex_features_at(tctx, qc, ys[i:i + chunk], xs[i:i + chunk])
        if keep is None:
            keep = [j for j, n in enumerate(tnames) if groups == ["all"] or any(n.startswith(g) for g in groups)]
        prob[i:i + chunk] = head.predict_proba(np.concatenate([f, t[:, keep]], 1))[:, 1]
    full = cv2.resize(prob.reshape(gh, gw), (gw * DOWN, gh * DOWN), interpolation=cv2.INTER_LINEAR)[:h, :w]
    return full > threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--groups", default="all")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--backbone", default="s")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--extra", action="store_true")
    a = ap.parse_args()
    groups = a.groups.split(",")
    rows = json.loads(Path(a.inputs).read_text())["examples"]
    if a.limit:
        rows = rows[:a.limit]
    for r in rows:
        extra = set(r) - INPUT_KEYS
        if extra:
            raise SystemExit(f"{r.get('id')}: unexpected input fields {sorted(extra)}")
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    head = joblib.load(a.head)
    model = load_model(a.backbone)
    by_img = {}
    for r in rows:
        by_img.setdefault(r["image"], []).append(r)
    t0 = time.time(); done = 0
    for img, group in by_img.items():
        gray = load_gray(Path(a.data_root) / img)
        dino = [dense_features(model, gray, s) for s in SCALES]
        ctx = ImageContext(gray, dino)
        t1 = time.time()
        tctx = TexContext(gray)
        print(f"texctx {time.time()-t1:.1f}s", flush=True)
        for r in group:
            if gray.shape != (r["height"], r["width"]):
                raise SystemExit(f"{r['id']}: image size mismatch")
            mask = predict_mask(head, ctx, tctx, r["query_box"], a.threshold, groups, a.extra)
            tmp = out / f".{r['id']}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(tmp)
            tmp.rename(out / f"{r['id']}.png")
            done += 1
            print(f"{done}/{len(rows)} {r['id']} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
