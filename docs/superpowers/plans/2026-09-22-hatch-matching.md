# Hatch Matching Challenge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and honestly evaluate a reproducible query-conditioned hatch segmenter targeting more than 75% public-validation document-macro IoU.

**Architecture:** A query-conditioned SegFormer model produces dense logits from overlapping drawing tiles, augmented by explicit Gabor texture similarity and foreground evidence. Document-grouped fold models, test-time transforms, calibrated channel fusion, and native-resolution postprocessing form the final ensemble.

**Tech Stack:** Python 3.11, PyTorch, Transformers, OpenCV, scikit-image, scikit-learn, Albumentations, Optuna, Pillow, NumPy, pytest

## Global Constraints

- Inference receives only label-free manifests, drawings, query boxes, and context boxes.
- Accuracy takes priority over compute cost; multi-GPU training and justified ensembles are allowed.
- Inference must never import or read labels, expected counts, or private data.
- Every prediction is a native-size mode `L` PNG containing only `0` and `255`.
- External model artifacts and dependencies are pinned and disclosed.
- Public-validation use and every reported metric are disclosed without editing evaluator output.
- The official evaluator is the scoring authority.

## File Map

- `pyproject.toml`: pinned package, test, and command configuration.
- `scripts/sync_challenge.py`: fetch and verify the official challenge utilities.
- `hatchmatch/contracts.py`: label-free request parsing and path safety.
- `hatchmatch/output.py`: atomic native-resolution PNG writing.
- `hatchmatch/features.py`: foreground, Gabor, and query texture features.
- `hatchmatch/data.py`: training-only labels, grouped folds, and tile sampling.
- `hatchmatch/model.py`: query-conditioned SegFormer network.
- `hatchmatch/losses.py`: masked focal, Dice, and boundary losses.
- `hatchmatch/train.py`: distributed fold training and checkpoint metadata.
- `hatchmatch/predict.py`: tiled model/texture inference and TTA.
- `hatchmatch/calibrate.py`: out-of-fold fusion and postprocessing search.
- `inference.py`: challenge-compatible entry point.
- `scripts/run_experiments.py`: baseline, fold, ablation, and ensemble orchestration.
- `scripts/build_submission.py`: artifacts, checksums, manifest, and report generation.
- `tests/`: unit and integration coverage for each boundary above.

---

### Task 1: Reproducible project and official tooling

**Files:**
- Create: `pyproject.toml`
- Create: `scripts/sync_challenge.py`
- Create: `tests/test_sync_challenge.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `sync(repo: str, revision: str, destination: Path) -> None`
- Produces: console scripts `hatch-train`, `hatch-calibrate`, and `hatch-infer`

- [ ] **Step 1: Write the failing tooling test**

```python
from pathlib import Path
from scripts.sync_challenge import REQUIRED_FILES, validate_checkout

def test_validate_checkout_rejects_missing_files(tmp_path: Path) -> None:
    missing = validate_checkout(tmp_path)
    assert missing == sorted(REQUIRED_FILES)
```

- [ ] **Step 2: Run the test and confirm the missing module failure**

Run: `python -m pytest tests/test_sync_challenge.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.sync_challenge'`.

- [ ] **Step 3: Add pinned packaging and sync implementation**

```python
# scripts/sync_challenge.py
from pathlib import Path
import subprocess

REQUIRED_FILES = {"evaluate.py", "make_inputs.py", "validate_submission.py", "data/release.json"}

def validate_checkout(root: Path) -> list[str]:
    return sorted(path for path in REQUIRED_FILES if not (root / path).is_file())

def sync(repo: str, revision: str, destination: Path) -> None:
    subprocess.run(["git", "clone", "--filter=blob:none", repo, str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", revision], check=True)
    missing = validate_checkout(destination)
    if missing:
        raise RuntimeError(f"challenge checkout is missing: {', '.join(missing)}")
```

Create `pyproject.toml` with `requires-python = ">=3.11,<3.13"`, exact dependency versions resolved by the package manager, pytest settings, and the three console scripts. Document setup, dataset download, training, inference, and evaluation commands in `README.md`.

- [ ] **Step 4: Install and verify**

Run: `python -m pip install -e '.[dev]' && python -m pytest tests/test_sync_challenge.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml scripts/sync_challenge.py tests/test_sync_challenge.py README.md
git commit -m "build: add reproducible hatch matching project"
```

### Task 2: Label-free contracts and binary outputs

**Files:**
- Create: `hatchmatch/__init__.py`
- Create: `hatchmatch/contracts.py`
- Create: `hatchmatch/output.py`
- Create: `tests/test_contracts.py`
- Create: `tests/test_output.py`

**Interfaces:**
- Produces: `Request`, `Box`, `load_requests(path, data_root) -> list[Request]`
- Produces: `write_binary_png(mask, path, expected_size) -> None`

- [ ] **Step 1: Write failing contract tests**

```python
import json
import numpy as np
from PIL import Image
from hatchmatch.contracts import load_requests
from hatchmatch.output import write_binary_png

def test_loader_rejects_labels(tmp_path):
    manifest = {"schema": "hatch-matching-challenge/v1", "examples": [{
        "id": "x", "document_id": "d", "kind": "real", "image": "x.png",
        "width": 4, "height": 3, "query_box": [0, 0, 1, 1],
        "context_boxes": [], "labels": {"positive_mask": "leak.png"}}]}
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(manifest))
    try:
        load_requests(path, tmp_path)
    except ValueError as exc:
        assert "labels" in str(exc)
    else:
        raise AssertionError("labels must be rejected")

def test_writer_emits_native_binary_l_png(tmp_path):
    path = tmp_path / "mask.png"
    write_binary_png(np.array([[False, True], [True, False]]), path, (2, 2))
    image = Image.open(path)
    assert image.mode == "L"
    assert set(image.getdata()) == {0, 255}
```

- [ ] **Step 2: Confirm failures**

Run: `python -m pytest tests/test_contracts.py tests/test_output.py -v`

Expected: FAIL because `hatchmatch.contracts` does not exist.

- [ ] **Step 3: Implement strict parsing and atomic output**

Use frozen dataclasses for `Box` and `Request`; reject unknown schema, duplicate IDs, any `labels` key, paths escaping `data_root`, invalid half-open boxes, image-size mismatches, and missing images. Convert Boolean masks to `uint8 * 255`, validate `(height, width)`, save to a sibling temporary PNG, reopen it to verify mode/size/values, then replace the destination atomically.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_contracts.py tests/test_output.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hatchmatch tests/test_contracts.py tests/test_output.py
git commit -m "feat: enforce label-free input and binary output contracts"
```

### Task 3: Classical texture baseline

**Files:**
- Create: `hatchmatch/features.py`
- Create: `hatchmatch/baseline.py`
- Create: `tests/test_features.py`
- Create: `tests/test_baseline.py`

**Interfaces:**
- Produces: `texture_channels(gray: np.ndarray, query_box: Box) -> np.ndarray`
- Produces: `baseline_probability(gray, query_box) -> np.ndarray`

- [ ] **Step 1: Write a synthetic hatch test**

```python
import numpy as np
from hatchmatch.baseline import baseline_probability
from hatchmatch.contracts import Box

def test_baseline_prefers_matching_orientation():
    image = np.full((128, 192), 255, np.uint8)
    image[:, 8:64:6] = 0
    for offset in range(-64, 192, 6):
        yy = np.arange(128)
        xx = yy + offset
        valid = (xx >= 96) & (xx < 176)
        image[yy[valid], xx[valid]] = 0
    score = baseline_probability(image, Box(8, 8, 56, 120))
    assert score[32:96, 8:64].mean() > score[32:96, 112:168].mean() + 0.2
```

- [ ] **Step 2: Confirm the test fails**

Run: `python -m pytest tests/test_features.py tests/test_baseline.py -v`

Expected: FAIL because the baseline module does not exist.

- [ ] **Step 3: Implement multi-scale texture channels**

Convert to float ink intensity, apply CLAHE, then compute foreground density, distance transform, local variance, and Gabor energy for 12 orientations, 4 wavelengths, and 3 scales. Summarize the query with robust median/MAD statistics; map per-pixel Mahalanobis-like distance to `[0, 1]`. Implement rotations through filter orientation, not image rotation, to preserve native coordinates.

- [ ] **Step 4: Verify baseline and record official score**

Run: `python -m pytest tests/test_features.py tests/test_baseline.py -v && python scripts/run_experiments.py baseline --config configs/baseline.yaml`

Expected: tests PASS and `runs/baseline/metrics.json` contains `summary.document_macro_iou`.

- [ ] **Step 5: Commit**

```bash
git add hatchmatch/features.py hatchmatch/baseline.py tests/test_features.py tests/test_baseline.py configs/baseline.yaml
git commit -m "feat: add multiscale texture baseline"
```

### Task 4: Training data and document-grouped folds

**Files:**
- Create: `hatchmatch/data.py`
- Create: `hatchmatch/augment.py`
- Create: `tests/test_data.py`
- Create: `tests/test_augment.py`

**Interfaces:**
- Produces: `HatchTileDataset`
- Produces: `make_group_folds(examples, count, seed) -> list[Fold]`
- Produces each item: `{"image", "query", "texture", "target", "known"}`

- [ ] **Step 1: Write leakage and mask tests**

```python
from hatchmatch.data import make_group_folds

def test_group_folds_never_split_documents():
    examples = [{"id": f"q{i}", "document_id": f"d{i // 2}"} for i in range(20)]
    for fold in make_group_folds(examples, count=5, seed=20260922):
        train_docs = {examples[i]["document_id"] for i in fold.train}
        valid_docs = {examples[i]["document_id"] for i in fold.valid}
        assert train_docs.isdisjoint(valid_docs)
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest tests/test_data.py tests/test_augment.py -v`

Expected: FAIL because `hatchmatch.data` does not exist.

- [ ] **Step 3: Implement grouped sampling**

Use `StratifiedGroupKFold` with `document_id` as groups and real/generated kind as the stratification signal. Sample positive, hard-negative, blank, and random known-centered tiles at a `4:3:2:1` ratio. Unknown and query/context pixels receive zero loss weight. Apply shared scale, rotation, perspective, blur, noise, JPEG, brightness, and line-dropout transforms to drawing/target/known; independently degrade the query appearance without changing its pattern class.

- [ ] **Step 4: Run tests and inspect fold manifests**

Run: `python -m pytest tests/test_data.py tests/test_augment.py -v && python -m hatchmatch.data --manifest dataset/train.json --folds 5 --output runs/folds.json`

Expected: PASS and every document ID occurs in one validation fold.

- [ ] **Step 5: Commit**

```bash
git add hatchmatch/data.py hatchmatch/augment.py tests/test_data.py tests/test_augment.py
git commit -m "feat: add grouped hatch training dataset"
```

### Task 5: Query-conditioned dense model and losses

**Files:**
- Create: `hatchmatch/model.py`
- Create: `hatchmatch/losses.py`
- Create: `tests/test_model.py`
- Create: `tests/test_losses.py`

**Interfaces:**
- Produces: `QuerySegFormer.forward(image, query, texture) -> Tensor[B,1,H,W]`
- Produces: `masked_segmentation_loss(logits, target, known) -> Tensor`

- [ ] **Step 1: Write shape, conditioning, and unknown-mask tests**

```python
import torch
from hatchmatch.losses import masked_segmentation_loss
from hatchmatch.model import QuerySegFormer

def test_model_shape_and_query_conditioning():
    model = QuerySegFormer(backbone="nvidia/mit-b2", pretrained=False).eval()
    image = torch.rand(2, 3, 128, 128)
    texture = torch.rand(2, 4, 128, 128)
    q1, q2 = torch.rand(2, 3, 96, 96), torch.zeros(2, 3, 96, 96)
    with torch.no_grad():
        assert model(image, q1, texture).shape == (2, 1, 128, 128)
        assert not torch.equal(model(image, q1, texture), model(image, q2, texture))

def test_loss_ignores_unknown_pixels():
    logits = torch.tensor([[[[0., 10.]]]])
    target = torch.tensor([[[[0., 0.]]]])
    known = torch.tensor([[[[1., 0.]]]])
    assert masked_segmentation_loss(logits, target, known).item() < 1.0
```

- [ ] **Step 2: Confirm failures**

Run: `python -m pytest tests/test_model.py tests/test_losses.py -v`

Expected: FAIL because model and loss modules do not exist.

- [ ] **Step 3: Implement the network and objective**

Use a pretrained MiT encoder for drawing and shared-weight query encoding. Pool query features into per-level prototypes, compute normalized dot-product similarity against drawing features, and inject similarities through FiLM into an FPN decoder. Concatenate resized classical channels before the final two convolution blocks. Loss is `0.5 * focal + 0.4 * soft_dice + 0.1 * boundary`, each reduced only over `known == 1`; return a differentiable zero when no known pixel exists.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_model.py tests/test_losses.py -v`

Expected: PASS on CPU.

- [ ] **Step 5: Commit**

```bash
git add hatchmatch/model.py hatchmatch/losses.py tests/test_model.py tests/test_losses.py
git commit -m "feat: add query-conditioned hatch segmenter"
```

### Task 6: Distributed fold training and checkpoint integrity

**Files:**
- Create: `hatchmatch/train.py`
- Create: `hatchmatch/checkpoints.py`
- Create: `configs/train-b2.yaml`
- Create: `configs/train-b4.yaml`
- Create: `tests/test_checkpoints.py`

**Interfaces:**
- Produces: `train_fold(config, fold) -> Path`
- Produces: `load_verified_checkpoint(path, expected_sha256) -> dict`

- [ ] **Step 1: Write checkpoint verification test**

```python
import hashlib, torch
from hatchmatch.checkpoints import load_verified_checkpoint

def test_verified_checkpoint(tmp_path):
    path = tmp_path / "model.pt"
    torch.save({"state_dict": {}, "config": {"fold": 0}}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_verified_checkpoint(path, digest)["config"]["fold"] == 0
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest tests/test_checkpoints.py -v`

Expected: FAIL because checkpoint utilities do not exist.

- [ ] **Step 3: Implement training**

Use `torchrun`, DDP, automatic mixed precision, gradient accumulation, gradient clipping at `1.0`, AdamW, cosine decay, EMA weights, and deterministic seeding. Validate each epoch on the held-out document fold; early-stop on document-macro IoU computed only from known pixels. Save best EMA weights with architecture, dependency versions, fold IDs, dataset checksum, git SHA, and SHA-256 sidecar. Define B2 and B4 configs with 5 folds and three independent seeds.

- [ ] **Step 4: Run smoke training and full folds**

Run: `torchrun --standalone --nproc_per_node=1 -m hatchmatch.train --config configs/train-b2.yaml --fold 0 --max-steps 2`

Expected: checkpoint and metadata are written, then verified by `tests/test_checkpoints.py`.

Run: `python scripts/run_experiments.py train-folds --configs configs/train-b2.yaml configs/train-b4.yaml`

Expected: all configured fold/seed checkpoints have valid SHA-256 sidecars.

- [ ] **Step 5: Commit**

```bash
git add hatchmatch/train.py hatchmatch/checkpoints.py configs tests/test_checkpoints.py
git commit -m "feat: add distributed grouped-fold training"
```

### Task 7: Native-resolution tiled prediction and TTA

**Files:**
- Create: `hatchmatch/predict.py`
- Create: `tests/test_predict.py`

**Interfaces:**
- Produces: `predict_probability(request, models, config) -> np.ndarray`
- Produces: `iter_tiles(height, width, size, overlap) -> Iterator[Tile]`

- [ ] **Step 1: Write seam and OOM fallback tests**

```python
import numpy as np
from hatchmatch.predict import blend_tiles, iter_tiles

def test_blended_constant_tiles_have_no_seams():
    tiles = list(iter_tiles(513, 769, size=256, overlap=64))
    values = [np.ones((tile.height, tile.width), np.float32) for tile in tiles]
    merged = blend_tiles((513, 769), tiles, values)
    np.testing.assert_allclose(merged, 1.0, atol=1e-6)
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest tests/test_predict.py -v`

Expected: FAIL because prediction utilities do not exist.

- [ ] **Step 3: Implement deterministic tiled inference**

Pad reflectively, use overlap equal to one quarter of tile size, blend with a clipped Hann window, and crop to native dimensions. Batch tiles by shape. On CUDA OOM, clear only unused cache, halve the batch size, and retry the same tile batch until size one; re-raise at size one. Average fold/seed checkpoints and horizontal, vertical, and 90-degree TTA only after reversing transforms in native coordinates. Return float32 probabilities matching `(height, width)`.

- [ ] **Step 4: Verify**

Run: `python -m pytest tests/test_predict.py -v`

Expected: PASS, including odd dimensions and images smaller than a tile.

- [ ] **Step 5: Commit**

```bash
git add hatchmatch/predict.py tests/test_predict.py
git commit -m "feat: add tiled ensemble prediction"
```

### Task 8: Out-of-fold calibration and postprocessing search

**Files:**
- Create: `hatchmatch/calibrate.py`
- Create: `configs/search.yaml`
- Create: `tests/test_calibrate.py`

**Interfaces:**
- Produces: `FusionCalibrator.fit(channels, target, known) -> FusionCalibrator`
- Produces: `postprocess(probability, foreground, config) -> np.ndarray`

- [ ] **Step 1: Write calibration isolation and postprocessing tests**

```python
import numpy as np
from hatchmatch.calibrate import postprocess

def test_postprocess_removes_isolated_pixel():
    probability = np.zeros((32, 32), np.float32)
    probability[2, 2] = 1.0
    probability[12:20, 12:20] = 1.0
    mask = postprocess(probability, np.ones_like(probability), {
        "threshold": .5, "min_component": 8, "close_radius": 0})
    assert not mask[2, 2]
    assert mask[15, 15]
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest tests/test_calibrate.py -v`

Expected: FAIL because calibration utilities do not exist.

- [ ] **Step 3: Implement leakage-safe fusion**

Fit logistic fusion on out-of-fold channels only: neural probability, classical texture probability, foreground, local variance, and query compatibility. Use balanced pixel subsampling capped per document. Search global threshold, minimum component area, closing radius, and foreground floor with Optuna; objective is document-macro IoU with a hard penalty when pooled recall is below `0.95`. Persist the study database, sampled parameters, fold provenance, and final JSON configuration.

- [ ] **Step 4: Run tests and search**

Run: `python -m pytest tests/test_calibrate.py -v && python scripts/run_experiments.py calibrate --config configs/search.yaml`

Expected: PASS and `runs/calibration/best.json` records objective, recall, study seed, and parameters.

- [ ] **Step 5: Commit**

```bash
git add hatchmatch/calibrate.py configs/search.yaml tests/test_calibrate.py
git commit -m "feat: calibrate fusion and postprocessing"
```

### Task 9: Challenge-compatible inference entry point

**Files:**
- Create: `inference.py`
- Create: `configs/final.yaml`
- Create: `tests/test_inference.py`

**Interfaces:**
- Consumes: `load_requests`, `predict_probability`, `postprocess`, `write_binary_png`
- Produces: CLI `python inference.py --inputs PATH --data-root PATH --output-dir PATH`

- [ ] **Step 1: Write end-to-end CLI test**

```python
import json, subprocess, sys
import numpy as np
from PIL import Image

def test_inference_cli_writes_every_request(tmp_path):
    Image.fromarray(np.full((16, 20), 255, np.uint8)).save(tmp_path / "drawing.png")
    manifest = {"schema": "hatch-matching-challenge/v1", "examples": [{
        "id": "q1", "document_id": "d1", "kind": "real", "image": "drawing.png",
        "width": 20, "height": 16, "query_box": [1, 1, 8, 8], "context_boxes": []}]}
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps(manifest))
    out = tmp_path / "predictions"
    subprocess.run([sys.executable, "inference.py", "--inputs", str(inputs),
                    "--data-root", str(tmp_path), "--output-dir", str(out),
                    "--config", "configs/test.yaml"], check=True)
    assert Image.open(out / "q1.png").size == (20, 16)
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest tests/test_inference.py -v`

Expected: FAIL because `inference.py` does not exist.

- [ ] **Step 3: Implement CLI and metadata**

Parse only the required three challenge arguments plus optional `--config` and `--device`. Load checksummed checkpoints, process every request, atomically write masks, and fail the run if any request fails. Write `run-metadata.json` containing elapsed wall time, count, hardware, package versions, git SHA, config hash, and checkpoint hashes; never inspect labels.

- [ ] **Step 4: Verify official contract**

Run: `python -m pytest tests/test_inference.py -v && python challenge/make_inputs.py --manifest dataset/val.json --output validation-inputs.json && python inference.py --inputs validation-inputs.json --data-root dataset --output-dir predictions`

Expected: 57 valid PNGs and no access to label paths.

- [ ] **Step 5: Commit**

```bash
git add inference.py configs/final.yaml configs/test.yaml tests/test_inference.py
git commit -m "feat: add challenge-compatible inference CLI"
```

### Task 10: Ablations, final ensemble, and score gate

**Files:**
- Create: `scripts/run_experiments.py`
- Create: `tests/test_experiments.py`
- Create: `runs/.gitignore`

**Interfaces:**
- Produces: immutable experiment directories containing config, logs, predictions, and unedited metrics

- [ ] **Step 1: Write result-provenance test**

```python
from scripts.run_experiments import validate_run

def test_run_requires_unedited_metrics_and_provenance(tmp_path):
    missing = validate_run(tmp_path)
    assert set(missing) == {"config.json", "metrics.json", "provenance.json"}
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest tests/test_experiments.py -v`

Expected: FAIL because the experiment runner does not exist.

- [ ] **Step 3: Implement orchestration**

Support `baseline`, `train-folds`, `predict-oof`, `calibrate`, `ablate`, and `final` subcommands. Hash every config and refuse to overwrite a completed run. Run ablations for texture channels, neural model, calibration, morphology, backbone size, fold count, seeds, and each TTA. Promote only additions that improve real-document out-of-fold IoU and do not collapse any document. Select the final ensemble before one last public-validation run.

- [ ] **Step 4: Run full verification and final evaluation**

Run: `python -m pytest -v && python scripts/run_experiments.py ablate --config configs/final.yaml && python scripts/run_experiments.py final --config configs/final.yaml`

Expected: all tests PASS; official `metrics.json` is retained unedited. If `document_macro_iou <= 0.75`, inspect per-document errors, add a hypothesis-driven experiment, and repeat Tasks 3–10 without claiming the target was beaten.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_experiments.py tests/test_experiments.py runs/.gitignore configs/final.yaml
git commit -m "feat: orchestrate ablations and final ensemble"
```

### Task 11: Submission artifacts and technical report

**Files:**
- Create: `scripts/build_submission.py`
- Create: `tests/test_submission.py`
- Create: `submission.example.json`
- Create: `docs/technical-report.md`

**Interfaces:**
- Produces: checksummed predictions archive, `submission.json`, metrics, report, and artifact manifest

- [ ] **Step 1: Write checksum-manifest test**

```python
import hashlib
from scripts.build_submission import artifact_record

def test_artifact_record(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"abc")
    record = artifact_record(path)
    assert record["bytes"] == 3
    assert record["sha256"] == hashlib.sha256(b"abc").hexdigest()
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest tests/test_submission.py -v`

Expected: FAIL because the submission builder does not exist.

- [ ] **Step 3: Implement artifact generation**

Generate SHA-256 and byte length for checkpoints, prediction archive, metrics, config, and environment lock. Populate the official submission schema with repository URL, immutable commit, commands, dataset/evaluator revisions, hardware, timing definition, data/pretraining disclosures, and artifact records. Build three public examples showing drawing, query, prediction, positive/negative labels, and unknown regions distinctly. The report must state limitations and keep historical reference metrics separate.

- [ ] **Step 4: Validate release bundle**

Run: `python scripts/build_submission.py --run runs/final --output dist && python challenge/validate_submission.py --submission dist/submission.json && python -m pytest -v`

Expected: validator exits 0, all tests PASS, and `dist/manifest.json` verifies every artifact.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_submission.py tests/test_submission.py submission.example.json docs/technical-report.md
git commit -m "docs: add reproducible challenge submission bundle"
```

### Task 12: Final reproducibility audit

**Files:**
- Modify: `README.md`
- Create: `docs/reproducibility.md`

**Interfaces:**
- Produces: a clean-machine sequence from clone through validated predictions

- [ ] **Step 1: Run clean-environment audit**

Run: `python -m venv /tmp/hatch-clean && /tmp/hatch-clean/bin/python -m pip install -e '.[dev]' && /tmp/hatch-clean/bin/python -m pytest -v`

Expected: installation succeeds and all tests PASS.

- [ ] **Step 2: Run label-free inference audit**

Run: `/tmp/hatch-clean/bin/python inference.py --inputs validation-inputs.json --data-root dataset --output-dir /tmp/hatch-predictions`

Expected: all requests complete using only paths reachable from the label-free manifest.

- [ ] **Step 3: Run official evaluator and compare immutable output**

Run: `/tmp/hatch-clean/bin/python challenge/evaluate.py --manifest dataset/val.json --data-root dataset --predictions /tmp/hatch-predictions --output /tmp/metrics.json && sha256sum /tmp/metrics.json`

Expected: output schema and eligible counts match the recorded final run; metric differences are investigated and resolved before release.

- [ ] **Step 4: Document exact reproduction commands**

Add the successful commands, hardware, wall time, peak-memory measurement method, artifact hashes, and expected output paths to `docs/reproducibility.md` and link it from `README.md`.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/reproducibility.md
git commit -m "docs: record clean reproduction workflow"
```
