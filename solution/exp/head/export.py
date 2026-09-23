"""Dev-only: turn an hcv.py head (cache/hd_head_NAME.joblib, class hcv.Avg) into a pure-sklearn drop-in for infer.py.

Usage: export.py NAME[,NAME2...] OUT.joblib -> VotingClassifier(soft) over the HGB members, each config weighted
equally (members split their config's weight). A single config that drops or derives columns is wrapped in a
Pipeline with a stateless ColumnTransformer. Checks the exported model against the hcv.Avg objects on val rows.
"""
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hcv  # noqa: E402


def export(avgs):
    ms = [m for a in avgs for m in a.ms]
    wts = [1 / len(a.ms) for a in avgs for _ in a.ms]
    vc = VotingClassifier([(f"m{i}", m) for i, m in enumerate(ms)], voting="soft", weights=wts)
    vc.estimators_ = list(ms)
    vc.le_ = LabelEncoder().fit(ms[0].classes_)
    vc.classes_ = vc.le_.classes_
    if all(not a.c["drop"] and not a.c["absd"] for a in avgs):
        return vc
    assert len(avgs) == 1, "column transforms are only supported for a single config"
    avg, c = avgs[0], avgs[0].c
    parts = [("keep", "passthrough", list(avg.cols))]
    if c["absd"]:
        parts.append(("abs", FunctionTransformer(np.abs), hcv.ABS_I))
    ct = ColumnTransformer(parts).fit(np.zeros((2, len(hcv.NAMES)), np.float32))
    return Pipeline([("cols", ct), ("head", vc)])


if __name__ == "__main__":
    sys.modules["__main__"].Avg = hcv.Avg  # hcv.py run as a script pickles Avg under __main__
    avgs = [joblib.load(hcv.C / f"hd_head_{n}.joblib") for n in sys.argv[1].split(",")]
    out = export(avgs)
    X = np.ascontiguousarray(hcv.load("val")[0][:200_000])
    ref = np.mean([a.predict_proba(X)[:, 1] for a in avgs], 0)
    d = np.abs(out.predict_proba(X)[:, 1] - ref).max()
    assert d < 1e-6, d
    joblib.dump(out, sys.argv[2])
    print("wrote", sys.argv[2], type(out).__name__, "max diff", d)
