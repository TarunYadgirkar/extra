"""Dev-only: deployed solution/infer.py predict path (texture groups + blur/fill post-processing) on val from cached
fine-tuned DINO grids, no GPU. Usage: val_cached_ft.py FEATTAG HEAD OUTDIR   (FEATTAG e.g. ft1 -> cache/feat_ft1_{0.5,1.0})"""
import json, sys, time
from pathlib import Path
import joblib
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from devdata import ROOT  # noqa: E402
from features import load_gray  # noqa: E402
from pixfeat import ImageContext  # noqa: E402
from texfeat import TexContext  # noqa: E402
from infer import predict_mask  # noqa: E402

feat, head, out = sys.argv[1], joblib.load(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
rows = json.loads((ROOT / "validation-inputs.json").read_text())["examples"]
by_img = {}
for r in rows:
    by_img.setdefault(r["image"], []).append(r)
t0 = time.time()
for img, group in by_img.items():
    gray = load_gray(ROOT / "dataset" / img)
    stem = Path(img).stem[:16]
    ctx = ImageContext(gray, [(np.load(ROOT / "cache" / f"feat_{feat}_{s}" / f"{stem}.npy"), 14 / s) for s in (0.5, 1.0)])
    tctx = TexContext(gray)
    for r in group:
        m = predict_mask(head, ctx, tctx, r["query_box"], 0.5)
        Image.fromarray(m.astype(np.uint8) * 255).save(out / f"{r['id']}.png")
        print(r["id"], f"{time.time()-t0:.0f}s", flush=True)
