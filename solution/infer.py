"""Hatch matching inference: python solution/infer.py --inputs X --data-root D --output-dir O"""
import argparse, json, sys, time
from pathlib import Path
import cv2
import joblib
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
sys.path.insert(0, str(Path(__file__).parent))
from features import load_model, load_gray, dense_features
from pixfeat import DOWN, ImageContext, features_at
from texfeat import TexContext, tex_query, tex_features_at

HERE = Path(__file__).parent
SCALES = (0.5, 1.0)
INPUT_KEYS = {"id", "image", "width", "height", "query_box", "context_boxes"}
TEX_GROUPS = ("s0_", "s1_", "qs", "b6", "b16", "qb", "ncc", "q_")


def predict_probabilities(head, features):
    if not isinstance(head, HistGradientBoostingClassifier):
        return head.predict_proba(features)[:, 1]
    # Use sklearn's fitted bins and native tree evaluator; requires the pinned sklearn version.
    values = head._preprocess_X(features, reset=False)
    binned = head._bin_mapper.transform(values)
    raw = np.zeros((len(values), head.n_trees_per_iteration_),
                   dtype=head._baseline_prediction.dtype, order="F")
    raw += head._baseline_prediction
    head._predict_iterations(binned, head._predictors, raw, True, _openmp_effective_n_threads())
    return head._loss.predict_proba(raw)[:, 1]


def postprocess(prob, shape, threshold):
    h, w = shape
    gh, gw = prob.shape
    # Labels cover whole regions, including enclosed text and symbols.
    g = cv2.GaussianBlur(prob, (0, 0), 14 / DOWN)
    lab, n = ndimage.label(g <= threshold)
    fill = np.ones(n + 1, bool); fill[0] = False
    fill[np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])] = False
    g = np.where(fill[lab], threshold + 0.01, g)
    full = cv2.resize(g, (gw * DOWN, gh * DOWN), interpolation=cv2.INTER_LINEAR)[:h, :w]
    return full > threshold


def predict_mask(head, ctx, tctx, box, threshold, chunk=400_000):
    h, w = ctx.shape
    gh, gw = -(-h // DOWN), -(-w // DOWN)
    gy, gx = np.mgrid[0:gh, 0:gw]
    ys = (gy.ravel() * DOWN + DOWN // 2).astype(np.float32)
    xs = (gx.ravel() * DOWN + DOWN // 2).astype(np.float32)
    cache, prob = {}, np.empty(len(ys), np.float32)
    qc = tex_query(tctx, box)
    keep = None
    for i in range(0, len(ys), chunk):
        f, _ = features_at(ctx, box, ys[i:i + chunk], xs[i:i + chunk], cache)
        t, names = tex_features_at(tctx, qc, ys[i:i + chunk], xs[i:i + chunk])
        if keep is None:
            keep = [j for j, n in enumerate(names) if n.startswith(TEX_GROUPS)]
        prob[i:i + chunk] = predict_probabilities(head, np.concatenate([f, t[:, keep]], 1))
    return postprocess(prob.reshape(gh, gw), (h, w), threshold)


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
        tctx = TexContext(gray)
        for r in group:
            if gray.shape != (r["height"], r["width"]):
                raise SystemExit(f"{r['id']}: image size mismatch")
            mask = predict_mask(head, ctx, tctx, r["query_box"], a.threshold)
            tmp = out / f".{r['id']}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(tmp)
            tmp.rename(out / f"{r['id']}.png")
            done += 1
            print(f"{done}/{len(rows)} {r['id']} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
