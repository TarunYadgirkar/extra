# Technical specification

This document defines the data, inference, prediction, and scoring contracts. `evaluate.py` is the executable scoring authority; record its revision with submitted results.

## 1. Manifest and coordinates

A dataset manifest has this envelope:

```json
{"schema": "hatch-matching-challenge/v1", "examples": []}
```

| Example field | Contract |
|---|---|
| `id` | Opaque query identifier and prediction filename stem. |
| `document_id` | Opaque group used for document-macro scoring. |
| `kind` | `real` or `generated_cad`. |
| `image` | Drawing path relative to `--data-root`. |
| `width`, `height` | Original drawing dimensions in pixels. |
| `query_box` | Required native rectangle `[x0,y0,x1,y1]`. |
| `context_boxes` | Additional explicitly supplied input rectangles, or `[]`. |
| `labels` | Training/evaluation annotation descriptors; excluded from inference requests. |

Coordinates are integers with half-open bounds: `x0 <= x < x1`, `y0 <= y < y1`. Rectangles satisfy `0 <= x0 < x1 <= width` and `0 <= y0 < y1 <= height`. Coordinates are not normalized; the evaluator does not resize, expand, or repair them.

`make_inputs.py` produces label-free requests. Do not consume annotation paths, annotation rectangles, or expected-count files during inference. The query rectangle refers to the same source drawing. Additional context is only what the request explicitly supplies.

## 2. Known, positive, negative, and support domains

Supported label keys are `positive_mask`, `negative_mask`, `known_mask`, `blank_mask`, and their rectangle counterparts `positive_boxes`, `negative_boxes`, `known_boxes`, `blank_boxes`. Mask paths are relative to `--data-root`. Box lists denote unions of native rectangles. Unexpected label keys are rejected.

Let `Omega = {0,...,width-1} x {0,...,height-1}`. Let `P0`, `N0`, and `B0` denote explicitly annotated positive, negative, and blank regions. Nonzero mask pixels belong to the corresponding region.

The known domain `K0` is the supplied known mask/box domain when present. Otherwise it is the union of explicitly annotated positive, negative, and blank regions. Let `S` be the union of the query rectangle and all supplied context rectangles. Scoring uses:

```text
K = K0 minus S
P = P0 intersect K
N = K minus P
B = B0 intersect K
U = Omega minus K
```

When an explicit known domain is supplied, every positive, negative, and blank label must lie within it. Positive labels cannot overlap explicit negative or blank labels. Contradictory annotations and examples with no known scoring pixels fail validation. Unknown pixels and supplied support pixels contribute to none of TP, FP, FN, or TN. A region is not negative merely because it is outside a positive mask or visually empty.

Dense known-domain masks and scoped rectangle annotations have intentional, different extents. No annotation is expanded to cover an entire drawing. Generated CAD remains identified separately and is training-only.

## 3. Binary prediction contract

Write `OUTPUT_DIR/<id>.png` for every request, at exactly its original `(width,height)`. Accepted binary PNG encodings are mode `1`, or mode `L` with values contained in `{0,1}` or contained in `{0,255}`. The same encoding and native-dimension rules apply to annotation masks. Nonzero is selected; zero is unselected.

Probability maps, RGB masks, palette masks, other value encodings, and incorrect dimensions are rejected. The scorer does not resize, interpolate, threshold-search, or repair predictions. A missing prediction fails evaluation instead of dropping the example. Predictions cover the complete native drawing; annotation domains determine which pixels are scored.

## 4. Inference interface

The submitted entry point must accept:

```text
--inputs PATH       Label-free request manifest
--data-root PATH    Root for relative drawing paths
--output-dir PATH   Destination of <id>.png predictions
```

A reproducible invocation has this form:

```bash
python your_inference.py --inputs validation-inputs.json --data-root dataset --output-dir predictions
```

The same interface applies to public validation and organizer-held private requests. It must not require target masks, evaluation annotations, expected pixel counts, or private labels. Disclose additional non-label data dependencies with the submission.

## 5. Confusion counts and aggregation

For binary prediction `M`:

```text
TP = |M intersect P|
FN = |P minus M|
FP = |M intersect N|
TN = |N minus M|
```

When `|P| > 0`, query IoU is `TP / (TP + FP + FN)`.

The primary result, `metrics.summary.document_macro_iou`, first averages eligible nonempty-target query IoUs within each document, then averages those document means. A document with no eligible nonempty-target query does not contribute to this mean. Pooling all pixels or averaging every query directly is a different metric.

Supporting fields include `mean_query_iou`, `global_precision`, `global_recall`, `empty_target_examples`, `positive_only_examples`, `empty_target_false_positive_rate`, and `blank_false_positive_rate`. Pooled precision is `sum(TP)/(sum(TP)+sum(FP))`; pooled recall is `sum(TP)/(sum(TP)+sum(FN))`. Zero denominators produce `null`, not an invented perfect or zero score.

An empty-target query has query IoU `null`; its false-positive rate is `FP/|K|`. Empty queries do not receive automatic perfect IoU. Positive-only examples have no annotated negative domain and cannot establish rejection performance outside their known positives. Retain the reported eligible counts when comparing results.

Reviewed-blank false-positive rate uses only explicitly labeled blank pixels: selected pixels in `B` divided by eligible pixels in `B`. Other-pattern negatives are not automatically blank. If no eligible blank domain is supplied, the blank rate is `null`; absent blank labels do not imply zero blank error.

## 6. Integrity and partition separation

`checksums.json` binds distributed files. Separate `expected-counts.json` records positive, negative, known, and unknown pixel totals for dataset/evaluator integrity checks only. It is never an inference input.

Public training contains 448 queries (192 real, 256 generated CAD); public validation contains 57 real queries; private test contains 40 real queries. Real document-group counts are 62, 13, 14. Known aliases remain together. Private test is participant-heldout, not claimed internally untouched.

Public-validation results, organizer private-test results, and historical reference-suite measurements are distinct. The documented 37-query reference is not a score on this challenge split.
