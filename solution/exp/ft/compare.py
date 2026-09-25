"""Paired per-document bootstrap between two cv OOF prob files (cache/tx_oof_*.npy), same as exp/backbone/cv_oof.py --compare.
Usage: compare.py NAME_A NAME_B"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backbone"))
from devdata import examples, ROOT  # noqa: E402
from train_head import load  # noqa: E402
from cv import THS  # noqa: E402
from cv_oof import per_doc  # noqa: E402

a, b = sys.argv[1], sys.argv[2]
_, Yu, Qu, _ = load("train_u", "v3")
exs = examples("train")
for t in THS:
    da = per_doc(np.load(ROOT / "cache" / f"tx_oof_{a}.npy"), Yu, Qu, exs, t)
    db = per_doc(np.load(ROOT / "cache" / f"tx_oof_{b}.npy"), Yu, Qu, exs, t)
    diff = np.array([db[d] - da[d] for d in da])
    boot = np.random.default_rng(0).choice(diff, (5000, len(diff))).mean(1)
    print(f"t={t} {b}-{a}: mean {diff.mean():+.4f}  95%CI [{np.quantile(boot, .025):+.4f},{np.quantile(boot, .975):+.4f}]"
          f"  better {int((diff > 0.005).sum())} worse {int((diff < -0.005).sum())} of {len(diff)}")
