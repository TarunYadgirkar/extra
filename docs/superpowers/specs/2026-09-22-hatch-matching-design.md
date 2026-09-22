# Hatch Matching Challenge Design

## Objective

Build a reproducible system for the TruTec Hatch Matching Challenge that targets more than 75% public-validation document-macro IoU while preserving private-test generalization. Recall should remain near or above 95%, reviewed-blank false positives should remain low, and all outputs must satisfy the challenge's native-resolution binary PNG contract.

The score is an optimization target, not a guaranteed outcome. The implementation will report the best honestly measured result even if it does not exceed 75%.

## Constraints

- Inference receives only the label-free request manifest, source drawings, query boxes, and supplied context boxes.
- Inference must never read labels, expected counts, or private data.
- Predictions must cover the full native drawing and use accepted binary PNG encodings.
- Model selection may use public training data and public validation results, with that use disclosed.
- The primary execution target is one CUDA-capable GPU. A slower CPU path must remain functional.
- External pretrained weights and all other artifacts must be pinned, checksummed, and disclosed.

## Considered Approaches

### Classical texture matching

Use oriented filters, local frequency descriptors, normalized correlation, and morphology. This is inexpensive and interpretable, but construction scans vary in scale, rotation, line quality, occlusion, and surrounding symbols. A classical-only system is likely too brittle.

### Learned segmentation

Train a query-conditioned segmentation network end to end. This could model the task directly, but only 192 real training queries are available. Generated CAD helps, yet the domain gap and document grouping create substantial overfitting risk.

### Hybrid matching — selected

Combine pretrained dense visual features with explicit texture and line evidence. Learned features supply invariance to scan degradation and local appearance; classical evidence preserves sensitivity to repeated hatch geometry and provides blank/text rejection. A lightweight calibration layer combines independently testable signals without requiring a large end-to-end model.

## Architecture

### 1. Data and evaluation layer

Vendor or pin the challenge utilities and dataset release metadata. A data loader validates manifest fields, resolves paths under the supplied data root, and exposes drawing images and support rectangles without exposing labels to inference code.

Training and analysis code may load labels through a separate module that cannot be imported by the inference entry point. Development reports use the official evaluator unchanged and retain its complete JSON output.

### 2. Query representation

The query crop is normalized in several representations:

- grayscale and contrast-normalized pixels;
- edge magnitude and oriented line responses;
- local frequency/orientation summaries;
- dense features from a pinned pretrained visual encoder.

Foreground estimation downweights paper background and border artifacts. Query descriptors are computed at multiple scales and rotations supported by the data analysis rather than assuming an exact pixel-scale match.

### 3. Drawing candidate generation

The drawing is processed in overlapping native-coordinate tiles with enough halo to avoid seam artifacts. Cheap foreground and texture tests reject obvious blank regions before expensive feature extraction.

For each remaining location, the system produces:

- learned-feature similarity to the query;
- orientation/frequency compatibility;
- edge-density and ink-structure compatibility;
- blank/text rejection evidence.

Tile results are blended into full-resolution score maps. Coordinates are tracked explicitly so no implicit resizing can violate the output contract.

### 4. Calibration and mask generation

A small regularized calibrator combines the score channels. Calibration examples are sampled by document, with held-out real document groups used for local cross-validation. Generated CAD is used for representation learning and robustness augmentation, not as evidence of real-domain generalization.

The calibrated map is thresholded using global parameters selected for document-macro IoU subject to a recall floor. Connected-component and morphology rules remove isolated detections, bridge hatch-line gaps, and suppress components incompatible with the query's estimated structure. Query and context regions are not specially cleared in the prediction because they are excluded by scoring, but they are excluded from calibration measurements.

### 5. Inference interface

`inference.py` accepts:

```text
--inputs PATH
--data-root PATH
--output-dir PATH
```

It creates exactly one native-size binary PNG per request. Device selection defaults to CUDA when available and otherwise uses CPU. Batch and tile sizes adapt to available memory without changing prediction semantics.

## Development Strategy

1. Establish a contract-correct baseline using classical multi-scale matching.
2. Add dense pretrained features and measure the change on document-grouped development folds.
3. Add calibration and rejection channels only when ablations improve held-out real-document performance.
4. Tune a small, declared parameter set against public validation.
5. Freeze the final configuration and generate reproducibility and submission artifacts.

The implementation will keep experiment configurations and results machine-readable. Every promoted change must include an ablation against the current baseline, with document-level scores inspected to avoid gains dominated by one drawing.

## Error Handling

- Reject malformed manifests, unsafe paths, missing images, invalid rectangles, and duplicate IDs with actionable messages.
- Fail the run if any output is missing or has the wrong dimensions or encoding.
- Catch GPU out-of-memory errors at tile-batch boundaries, reduce the batch size deterministically, and retry; other inference errors remain fatal.
- Write predictions atomically so interrupted runs do not leave apparently valid partial files.
- Record model and configuration hashes in run metadata.

## Testing and Verification

Unit tests cover coordinate transforms, tiling/blending, rotation and scale handling, binary encoding, support exclusion in metric analysis, and deterministic calibration. Integration tests run the public label-free interface on representative examples and validate outputs with the challenge validator/evaluator.

Quality gates are:

- exact interface and PNG contract compliance;
- no label-related imports or paths reachable from inference;
- deterministic predictions within fixed hardware/backend settings;
- document-grouped cross-validation reports and feature ablations;
- official public-validation metrics, including eligible counts, precision, recall, blank false-positive rate, and document-macro IoU;
- CPU smoke test and timed GPU run.

## Deliverables

- inference and training/analysis source code;
- pinned environment and model artifact manifest;
- tests and reproducible commands;
- validation predictions and unedited evaluator output;
- submission manifest;
- concise technical report with three required visual examples, limitations, timing, hardware, and data-use disclosure.

## Scope Boundaries

The project will not use validation labels as inference inputs, search for private test material, or claim a score not produced by the official evaluator. Distributed training, a hosted service, and a general drawing-understanding product are out of scope unless measurement shows they are necessary for the challenge result.
