# Exact binned inference

The deployed head is a 400-tree `HistGradientBoostingClassifier`. Its default inference compares floating-point feature values with tree thresholds repeatedly. `solution/infer.py` instead maps each feature to the head's fitted bins once, then uses scikit-learn's native binned tree traversal. Tree order, missing-value routing, baseline and probability transformation stay unchanged. No retraining or threshold selection occurs.

This uses private scikit-learn APIs and requires the pinned version in `solution/requirements.txt`. Regression tests compare the shipped head against public `predict_proba` at fitted bin boundaries, adjacent floating-point values and missing values. Other estimator types retain their public prediction path.

Run from the repository root:

```sh
OMP_NUM_THREADS=4 .venv/bin/python -m unittest discover -s solution/tests -v
nice -n 15 env OMP_NUM_THREADS=4 .venv/bin/python solution/exp/runtime/benchmark.py --output scratch/runtime.json
```

The paired benchmark uses one query per drawing, alternates method order, checks every output pixel, and excludes backbone extraction and image-context construction from timing. It requires the existing DINO feature caches. It does not read annotation labels or fit a model. Full inference timing must separately include initialization, backbone loading, extraction and PNG writing.
