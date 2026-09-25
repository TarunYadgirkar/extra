"""Dev-only feature cache. Usage: cache_feats.py SPLIT MODEL SCALE"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from features import load_model, load_gray, dense_features
from devdata import examples, domain, ROOT

split, size, scale = sys.argv[1], sys.argv[2], float(sys.argv[3])
out = ROOT / "cache" / f"feat_{size}_{scale}"
out.mkdir(parents=True, exist_ok=True)
model = load_model(size, finetuned=False)  # frozen grids: the fine-tuned ones come from exp/ft/extract_ft.py
by_img = {}
for e in examples(split):
    by_img.setdefault(e["image"], []).append(e)
for n, (img, exs) in enumerate(by_img.items()):
    stem = Path(img).stem[:16]
    p = out / f"{stem}.npy"
    if p.exists():
        continue
    t = time.time()
    f, stride = dense_features(model, load_gray(ROOT / "dataset" / img), scale)
    np.save(p, f)
    print(n, len(by_img), f.shape, f"{time.time()-t:.1f}s", flush=True)
