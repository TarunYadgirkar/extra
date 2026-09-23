"""Dev: write val PNGs from cached 4px prob grids with optional post-processing, exactly as infer.py would.
python val_pp.py OUTDIR [--none]"""
import json, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from common import CACHE, ROOT
import pp as P

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rows = json.loads((ROOT / "validation-inputs.json").read_text())["examples"]
for r in rows:
    g = np.load(CACHE / "cal_p4" / f"{r['id']}.npy").astype(np.float32)
    if "--none" not in sys.argv:
        g = P.fill_holes(1e6)(P.blur(14)(g, 4), 4)
    gh, gw = g.shape
    full = cv2.resize(g, (gw * 4, gh * 4), interpolation=cv2.INTER_LINEAR)[:r["height"], :r["width"]]
    Image.fromarray((full > 0.5).astype(np.uint8) * 255).save(out / f"{r['id']}.png")
