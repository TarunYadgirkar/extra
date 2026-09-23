import sys, time
import numpy as np
from maps import Scorer, load_grid
src = sys.argv[1] if len(sys.argv) > 1 else "s14"
for split in ("train", "val"):
    t = time.time(); S = Scorer(split)
    P = {}
    for i, e, _ in S.qs:
        try:
            g, s = load_grid(src, e)
        except FileNotFoundError:
            continue
        P[i] = S.probs(i, g, s)
    print(split, len(P), [round(S.score(P, th)[0], 4) for th in (0.4, 0.5, 0.6, 0.7)], f"{time.time()-t:.0f}s", flush=True)
