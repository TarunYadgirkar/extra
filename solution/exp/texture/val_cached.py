"""Dev-only: run the infer_tx.py predict path on val using cached DINO grids (no GPU), write PNGs for evaluate.py.

Usage: val_cached.py HEAD OUTDIR [--groups G] [--extra] [--threshold T]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import joblib  # noqa: E402
from devdata import ROOT  # noqa: E402
from features import load_gray  # noqa: E402
from pixfeat import ImageContext  # noqa: E402
from txfeat import TexContext  # noqa: E402
from infer_tx import predict_mask  # noqa: E402

SCALES = [("feat_s_0.5", 28.0), ("feat_s_1.0", 14.0)]

ap = argparse.ArgumentParser()
ap.add_argument("head"); ap.add_argument("outdir")
ap.add_argument("--groups", default="all"); ap.add_argument("--extra", action="store_true")
ap.add_argument("--threshold", type=float, default=0.5)
a = ap.parse_args()
rows = json.loads((ROOT / "validation-inputs.json").read_text())["examples"]
head = joblib.load(a.head)
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
by_img = {}
for r in rows:
    by_img.setdefault(r["image"], []).append(r)
t0 = time.time()
for img, group in by_img.items():
    gray = load_gray(ROOT / "dataset" / img)
    stem = Path(img).stem[:16]
    ctx = ImageContext(gray, [(np.load(ROOT / "cache" / d / f"{stem}.npy"), s) for d, s in SCALES])
    tctx = TexContext(gray)
    for r in group:
        m = predict_mask(head, ctx, tctx, r["query_box"], a.threshold, a.groups.split(","), a.extra)
        Image.fromarray(m.astype(np.uint8) * 255).save(out / f"{r['id']}.png")
        print(r["id"], f"{time.time()-t0:.0f}s", flush=True)
