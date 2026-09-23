"""Dev: per-query val IoU (row level, t=0.5) of ev_pval_NAME vs tx_v1f; prints changed queries."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from devdata import examples, ROOT  # noqa: E402
C = ROOT / "cache"
Y, Q = np.load(C / "hd_val_Y.npy"), np.load(C / "hd_val_Q.npy")
ex = examples("val")
pa = np.load(C / "tx_pval_tx_v2_s0_+s1_+qs+b6+b16+qb+ncc+q_.npy")
for name in sys.argv[1:]:
    pb = np.load(C / f"ev_pval_{name}.npy")
    print(name)
    for i in range(len(ex)):
        m = Q == i; y = Y[m]
        if not y.any():
            continue
        ia = ((pa[m] > .5) & y).sum() / ((pa[m] > .5) | y).sum(); ib = ((pb[m] > .5) & y).sum() / ((pb[m] > .5) | y).sum()
        if abs(ia - ib) > 0.02 or i in (3, 45, 28, 56):
            print(f"  q{i} {ex[i]['document_id'][:10]} {ia:.3f} -> {ib:.3f}")
