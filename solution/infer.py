"""Hatch matching inference: python solution/infer.py --inputs X --data-root D --output-dir O"""
import argparse, json, sys, time
from pathlib import Path
import cv2
import joblib
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
from features import load_model, load_gray, dense_features
from pixfeat import DOWN, ImageContext, features_at

HERE = Path(__file__).parent
SCALES = (0.5, 1.0)
INPUT_KEYS = {"id", "image", "width", "height", "query_box", "context_boxes"}


def predict_mask(head, ctx, box, threshold, chunk=400_000):
    h, w = ctx.shape
    gh, gw = -(-h // DOWN), -(-w // DOWN)
    gy, gx = np.mgrid[0:gh, 0:gw]
    ys = (gy.ravel() * DOWN + DOWN // 2).astype(np.float32)
    xs = (gx.ravel() * DOWN + DOWN // 2).astype(np.float32)
    cache, prob = {}, np.empty(len(ys), np.float32)
    for i in range(0, len(ys), chunk):
        f, _ = features_at(ctx, box, ys[i:i + chunk], xs[i:i + chunk], cache)
        prob[i:i + chunk] = head.predict_proba(f)[:, 1]
    full = cv2.resize(prob.reshape(gh, gw), (gw * DOWN, gh * DOWN), interpolation=cv2.INTER_LINEAR)[:h, :w]
    return full > threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--head", default=str(HERE / "weights" / "head.joblib"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--backbone", default="s")
    a = ap.parse_args()
    rows = json.loads(Path(a.inputs).read_text())["examples"]
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
        for r in group:
            if gray.shape != (r["height"], r["width"]):
                raise SystemExit(f"{r['id']}: image size mismatch")
            mask = predict_mask(head, ctx, r["query_box"], a.threshold)
            tmp = out / f".{r['id']}.png"
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(tmp)
            tmp.rename(out / f"{r['id']}.png")
            done += 1
            print(f"{done}/{len(rows)} {r['id']} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
