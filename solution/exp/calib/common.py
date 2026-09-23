"""Shared dev helpers for calibration/post-processing experiments (never used at inference)."""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SOL = HERE.parents[1]
ROOT = SOL.parent
sys.path.insert(0, str(SOL))
from devdata import examples, domain  # noqa: E402

CACHE = ROOT / "cache"
TXF = "tx_v2_s0_+s1_+qs+b6+b16+qb+ncc+q_"
FOLDS = 4


def cv_split():
    exs = examples("train")
    docs = sorted({e["document_id"] for e in exs})
    rng = np.random.default_rng(0); rng.shuffle(docs)
    fold_of = {d: i % FOLDS for i, d in enumerate(docs)}
    return exs, np.array([fold_of[e["document_id"]] for e in exs])


def rows_yq(split):
    d = np.load(CACHE / f"rows_{split}_v3.npz")
    return d["Y"], d["Q"]


def iou(p, y):
    u = (p | y).sum()
    return (p & y).sum() / u if u else np.nan


def doc_macro(per_q, exs):
    """per_q: {query index: iou or nan}. Mirrors evaluate doc-macro (queries with no positives dropped)."""
    docs = {}
    for i, v in per_q.items():
        if not np.isnan(v):
            docs.setdefault(exs[i]["document_id"], []).append(v)
    return float(np.mean([np.mean(v) for v in docs.values()])), {d: np.mean(v) for d, v in docs.items()}


def boot(da, db, n=5000):
    """Paired per-doc bootstrap of mean(db - da)."""
    ks = sorted(da)
    diff = np.array([db[k] - da[k] for k in ks])
    rng = np.random.default_rng(0)
    bs = diff[rng.integers(len(diff), size=(n, len(diff)))].mean(1)
    return diff.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), int((diff > 1e-9).sum()), int((diff < -1e-9).sum())


def fmt(b):
    return f"{b[0]:+.4f} [{b[1]:+.4f},{b[2]:+.4f}] better {b[3]} worse {b[4]}"
