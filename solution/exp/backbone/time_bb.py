"""Inference cost per backbone grid on one full-size drawing. Usage: time_bb.py NAME:SCALE ..."""
import sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from features_bb import load_model, load_gray, dense_features
from pixfeat import ImageContext, features_at, DOWN
from devdata import examples, ROOT

e = examples("val")[0]
gray = load_gray(ROOT / "dataset" / e["image"])
h, w = gray.shape
gy, gx = np.mgrid[0:-(-h // DOWN), 0:-(-w // DOWN)]
ys = (gy.ravel() * DOWN + DOWN // 2).astype(np.float32)
xs = (gx.ravel() * DOWN + DOWN // 2).astype(np.float32)
for spec in sys.argv[1:]:
    grids, t_ext = [], 0.0
    for part in spec.split("+"):
        name, scale = part.split(":")
        m = load_model(name)
        dense_features(m, gray[:1000, :1000], float(scale))
        t = time.time()
        grids.append(dense_features(m, gray, float(scale)))
        torch.mps.synchronize()
        t_ext += time.time() - t
    t = time.time(); ctx = ImageContext(gray, grids); t_ctx = time.time() - t
    t = time.time(); cache = {}
    for i in range(0, len(ys), 400_000):
        features_at(ctx, e["query_box"], ys[i:i + 400_000], xs[i:i + 400_000], cache)
    t_q = time.time() - t
    print(f"{spec}: extract {t_ext:.1f}s  context {t_ctx:.1f}s  per-query features {t_q:.1f}s", flush=True)
