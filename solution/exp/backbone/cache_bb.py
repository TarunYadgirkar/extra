"""Feature cache for train+val. Usage: cache_bb.py MODEL SCALE -> cache/bb_feat_{MODEL}_{SCALE}/"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from features_bb import load_model, load_gray, dense_features
from devdata import examples, ROOT

name, scale = sys.argv[1], float(sys.argv[2])
out = ROOT / "cache" / f"bb_feat_{name}_{scale}"
out.mkdir(parents=True, exist_ok=True)
model = load_model(name)
imgs = list(dict.fromkeys(e["image"] for s in ("val", "train") for e in examples(s)))
for n, img in enumerate(imgs):
    p = out / f"{Path(img).stem[:16]}.npy"
    if p.exists():
        continue
    t = time.time()
    f, stride = dense_features(model, load_gray(ROOT / "dataset" / img), scale)
    np.save(p.with_suffix(".tmp.npy"), f)
    p.with_suffix(".tmp.npy").rename(p)
    print(n, len(imgs), f.shape, stride, f"{time.time()-t:.1f}s", flush=True)
