"""Inference for the neural head.

Auto-context (net trained with ctx=1 resid=1), the recommended path:
  python solution/exp/neural/infer_nn.py --inputs X --data-root D --output-dir O \
      --net solution/exp/neural/weights/nn_c1_fall.pt --gbm cache/nn_gbm/tx_v1_fall.joblib
  final logit = tx_v1 GBM logit on the stride-4 grid + bilinear-upsampled s14 net correction.
Standalone net (ctx=0): omit --gbm; mask = sigmoid(net) > threshold.
"""
import argparse, json, sys, time
from pathlib import Path
import cv2
import joblib
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "texture"))
from nn_common import S, PCA_PATH, image_input, LiteCtx, query_sims, query_cells  # noqa: E402
from features import load_model, load_gray, dense_features  # noqa: E402
from pixfeat import DOWN, ImageContext, features_at, sample_grid  # noqa: E402
from infer import SCALES, INPUT_KEYS  # noqa: E402
from train_nn import NORM_PATH, ranks, device, build_net  # noqa: E402


class NNHead:
    def __init__(self, path):
        ck = torch.load(path, map_location="cpu")
        self.net = build_net(ck["cfg"])
        self.net.load_state_dict(ck["state"]); self.net.eval().to(device())
        self.pca, self.norm = dict(np.load(PCA_PATH)), dict(np.load(NORM_PATH))

    def image(self, gray, f28, f14):
        """f28/f14: DINOv2-S grids from dense_features at 0.5x and 1x."""
        x = (image_input(gray, f14, f28, self.pca).astype(np.float32) - self.norm["mu"]) / self.norm["sd"]
        return x, LiteCtx([(f28, 28.0), (f14, 14.0)])

    @torch.no_grad()
    def logits(self, img, box, gbm14=None):
        x, ctx = img
        gh, gw = x.shape[:2]
        sims, qs = query_sims(ctx, box, gh, gw)
        sims = sims.astype(np.float32)
        extra = []
        if gbm14 is not None:
            q = np.quantile(gbm14, np.linspace(0, 1, 101))
            extra = [np.stack([np.clip(gbm14, -10, 10) / 4, np.interp(gbm14, q, np.linspace(0, 1, 101))], -1).astype(np.float32)]
        s = np.concatenate([sims, ranks(sims, qs)] + extra, -1)
        y0, y1, x0, x1 = query_cells(box, gh, gw)
        qt = x[y0:y1, x0:x1].reshape(-1, x.shape[-1])
        dev = device()
        tt = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32))[None].to(dev)
        qm = torch.ones(1, len(qt), dtype=torch.bool, device=dev)
        return self.net(tt(x), tt(s), tt(qt), qm)[0].cpu().numpy()


def grid_points(h, w):
    gh, gw = -(-h // DOWN), -(-w // DOWN)
    gy, gx = np.mgrid[0:gh, 0:gw]
    return gh, gw, (gy.ravel() * DOWN + DOWN // 2).astype(np.float32), (gx.ravel() * DOWN + DOWN // 2).astype(np.float32)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def gbm_logits(head, ctx, tctx, box, ys, xs, cache, chunk=400_000):
    """tx_v1 GBM (v3 + all texture cols, same order as training rows) logits at native points."""
    from txfeat import tex_query, tex_features_at
    qc = cache.setdefault("tex_q", tex_query(tctx, box, False))
    out = np.empty(len(ys), np.float32)
    for i in range(0, len(ys), chunk):
        f, _ = features_at(ctx, box, ys[i:i + chunk], xs[i:i + chunk], cache)
        t, _ = tex_features_at(tctx, qc, ys[i:i + chunk], xs[i:i + chunk])
        out[i:i + chunk] = logit(head.predict_proba(np.concatenate([f, t], 1))[:, 1])
    return out


def ctx_prob(nn, nimg, head, ctx, tctx, box):
    h, w = ctx.shape
    cache = {}
    gh14, gw14 = nimg[0].shape[:2]
    gy, gx = np.mgrid[0:gh14, 0:gw14]
    g14 = gbm_logits(head, ctx, tctx, box, (gy.ravel() * S + S // 2).astype(np.float32),
                     (gx.ravel() * S + S // 2).astype(np.float32), cache).reshape(gh14, gw14)
    delta = nn.logits(nimg, box, g14) - np.clip(g14, -10, 10)
    gh, gw, ys, xs = grid_points(h, w)
    z = gbm_logits(head, ctx, tctx, box, ys, xs, cache) + sample_grid(delta, S, ys, xs)
    z = cv2.resize(z.reshape(gh, gw), (gw * DOWN, gh * DOWN), interpolation=cv2.INTER_LINEAR)[:h, :w]
    return 1 / (1 + np.exp(-z))


def nn_prob(logit, h, w):
    """Bilinear s14 logits -> native probabilities (same sampling convention as pixfeat.sample_grid)."""
    gh, gw = logit.shape
    up = cv2.resize(logit, (gw * S, gh * S), interpolation=cv2.INTER_LINEAR)[:h, :w]
    return 1 / (1 + np.exp(-up))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--gbm", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    rows = json.loads(Path(a.inputs).read_text())["examples"]
    for r in rows:
        extra = set(r) - INPUT_KEYS
        if extra:
            raise SystemExit(f"{r.get('id')}: unexpected input fields {sorted(extra)}")
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    nn = NNHead(a.net)
    head = joblib.load(a.gbm) if a.gbm else None
    model = load_model("s")
    by_img = {}
    for r in rows:
        by_img.setdefault(r["image"], []).append(r)
    t0 = time.time(); done = 0
    for img, group in by_img.items():
        gray = load_gray(Path(a.data_root) / img)
        (f28, _), (f14, _) = [dense_features(model, gray, s) for s in SCALES]
        nimg = nn.image(gray, f28, f14)
        if head is not None:
            from txfeat import TexContext
            ctx = ImageContext(gray, [(f28, 28.0), (f14, 14.0)])
            tctx = TexContext(gray)
        for r in group:
            if gray.shape != (r["height"], r["width"]):
                raise SystemExit(f"{r['id']}: image size mismatch")
            if head is not None:
                prob = ctx_prob(nn, nimg, head, ctx, tctx, r["query_box"])
            else:
                prob = nn_prob(nn.logits(nimg, r["query_box"]), *gray.shape)
            tmp = out / f".{r['id']}.png"
            Image.fromarray((prob > a.threshold).astype(np.uint8) * 255).save(tmp)
            tmp.rename(out / f"{r['id']}.png")
            done += 1
            print(f"{done}/{len(rows)} {r['id']} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
