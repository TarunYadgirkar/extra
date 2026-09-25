"""Dev: write val PNGs from cached 4px prob grids with optional post-processing, using the deployed post-processing. Cached float16 grids may differ from fresh inference.
python val_pp.py OUTDIR [--none]"""
import json, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from common import CACHE, ROOT
from infer import postprocess

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rows = json.loads((ROOT / "validation-inputs.json").read_text())["examples"]
for r in rows:
    g = np.load(CACHE / "cal_p4" / f"{r['id']}.npy").astype(np.float32)
    if "--none" in sys.argv:
        gh, gw = g.shape
        full = cv2.resize(g, (gw * 4, gh * 4), interpolation=cv2.INTER_LINEAR)[:r["height"], :r["width"]]
        mask = full > 0.5
    else:
        mask = postprocess(g, (r["height"], r["width"]), 0.5)
    Image.fromarray(mask.astype(np.uint8) * 255).save(out / f"{r['id']}.png")
