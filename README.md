# Hatch Matching

Reproducible tooling for the TruTec-AI Hatch Matching Challenge. Python 3.11
and 3.12 are supported; direct project and development dependencies are pinned
exactly in `pyproject.toml`.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Official challenge tooling

Pin the official repository to the revision published with the challenge
release. The sync helper clones that exact revision and verifies the evaluator,
input builder, submission validator, and release metadata:

```bash
export HATCH_CHALLENGE_REPO='OFFICIAL_REPOSITORY_URL'
export HATCH_CHALLENGE_REVISION='OFFICIAL_RELEASE_COMMIT'
python - <<'PY'
import os
from pathlib import Path

from scripts.sync_challenge import sync

sync(
    os.environ["HATCH_CHALLENGE_REPO"],
    os.environ["HATCH_CHALLENGE_REVISION"],
    Path("challenge"),
)
PY
```

Use the dataset URL and checksum from the synced
`challenge/data/release.json`:

```bash
export HATCH_DATASET_URL='OFFICIAL_DATASET_URL'
curl --fail --location "$HATCH_DATASET_URL" --output dataset.tar.gz
mkdir -p dataset
tar --extract --file dataset.tar.gz --directory dataset
```

Verify the downloaded archive against the checksum in the official release
metadata before using it.

## Train, calibrate, and infer

```bash
hatch-train --config configs/train-b2.yaml
hatch-calibrate --config configs/search.yaml
hatch-infer \
  --inputs validation-inputs.json \
  --data-root dataset \
  --output-dir predictions \
  --config configs/final.yaml
```

The model, data, and command implementations are added by subsequent project
tasks; Task 1 reserves these stable console-script names.

## Official evaluation

Build label-free inputs, run inference, validate the submission, and evaluate
with the pinned official tools:

```bash
python challenge/make_inputs.py \
  --manifest dataset/val.json \
  --output validation-inputs.json
hatch-infer \
  --inputs validation-inputs.json \
  --data-root dataset \
  --output-dir predictions \
  --config configs/final.yaml
python challenge/validate_submission.py --predictions predictions
python challenge/evaluate.py \
  --manifest dataset/val.json \
  --data-root dataset \
  --predictions predictions \
  --output metrics.json
```

## Reproducibility audit

The commands that completed on a clean CPU-only environment, the measured
pytest wall time, the existing capped-1600 baseline metrics path, and the
unscored neural ensemble are recorded in
[docs/reproducibility.md](docs/reproducibility.md).
