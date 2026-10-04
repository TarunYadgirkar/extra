# Technical report

## Input and output contract

Inference reads a label-free request manifest and the original drawings. It
writes one native-resolution binary PNG per request id.

Native classical inference is separate from the capped-1600 evaluation. It
scores each original drawing with `baseline_probability` and does not resize
the page to a working-resolution cap. `configs/test.yaml` selects that
classical mode:

```bash
python inference.py --inputs validation-inputs.json --data-root dataset --output-dir predictions --config configs/test.yaml
```

The native experiment config is `configs/baseline.yaml` (`max_dimension: null`):

```bash
python scripts/run_experiments.py baseline --config configs/baseline.yaml
```

That native command was not the source of `runs/baseline-eval-1600/metrics.json`.

Annotation masks, annotation rectangles, expected counts, and private labels
are not inference inputs. Unknown pixels and supplied query or context
supports are unscored. The organizer-held private test is not displayed and
was not used.

## Reproducibility

The repository is `https://github.com/TarunYadgirkar/extra`. Direct
dependencies are pinned in `pyproject.toml`. The dataset release is version
1.0, URL
`https://github.com/TruTec-AI/hatch-matching-challenge/releases/download/dataset-v1/trutec-hatch-dataset-v1.zip`,
422,368,602 bytes, SHA-256
`d5c1c845898669fe3941c363db3f328ed9d50c828acf7daeed7c9b2ae74b2229`. The
evaluator revision is `65b98e480f2a10b82f974f4feb519ee4e012d66c`.

`scripts/build_submission.py` packages a run only when `config.json` and
`metrics.json` still match `provenance.json`. It copies those metrics bytes
unchanged, writes SHA-256 and byte length for the prediction archive,
metrics, config, environment lock, and any supplied checkpoints, and asks the
official `challenge/validate_submission.py` to accept `submission.json`. A
request for `runs/final` fails because that sealed ensemble does not exist.

## Measured results

The 73.73% figure is a historical reference on a different 37-query suite. It is not a score on this challenge's public-validation split.

That cached reference is 37 real queries across 12 documents
(`document_macro_iou` 0.7373200409764458 in the challenge reference metrics).
It is not the public-validation split and it is not the private test.

The capped-1600 classical baseline document-macro IoU is 0.4779169321.

That score was produced by the capped classical experiment:

```bash
python scripts/run_experiments.py baseline --config configs/baseline-eval-1600.yaml
```

`configs/baseline-eval-1600.yaml` sets `max_dimension: 1600`. Native classical
inference is the separate path described above.

The unedited evaluator file `runs/baseline-eval-1600/metrics.json` records
the full value 0.4779169321376195. Its SHA-256 is
`04e1ac742653c468b52f8470bf8829273589d8b6412433ccad674de7d931667c`. The run
scored 57 public-validation queries across 13 documents. Supporting
evaluator figures from that same file are mean query IoU
0.43592916774392726, pooled precision 0.4827817373038098, and pooled recall
0.5652035245486439. Empty-target queries were 0. Reviewed-blank pixels were
0, so the blank false-positive rate is null. Two examples are positive-only.
Document `doc-d4324654d593e7af` scored 0. This is the explicitly capped
1,600-pixel classical baseline, not a native-resolution rerun.

That metrics file has no `provenance.json` seal. The builder therefore
refuses to package it until the bytes are sealed without editing them. No
second public-validation document-macro IoU is claimed.

The submitted public-validation score is document-macro IoU 0.6283198947626735.

That score is the unedited output of `challenge/evaluate.py` on
`runs/segment-v2/metrics.json` (46605 bytes, SHA-256
`4b5fcf5a96ac2686209171cd86ec2ba3d94e130b62784ec1e325d4da885cec5b`). It covers
57 queries and 13 documents. Mean query IoU is 0.5528167717705091. Pooled
precision is 0.5457175109355157 and pooled recall is 0.7497639252815699.
Empty-target queries were 0. Reviewed-blank pixels were 0. Two examples are
positive-only. Three documents scored 1.0. `doc-2f69e1183867e83c` scored
0.03489977904040404.

The masks came from:

```bash
python challenge/make_inputs.py --manifest dataset/val.json --output validation-inputs.json
python scripts/segment_predict.py --inputs validation-inputs.json --data-root dataset --output-dir predictions --config configs/segment.yaml
```

`configs/segment.yaml` sets `tone_threshold: 0.7`, `alpha: 0.75`, and
`max_side: 2000`. Flat gray fills are matched at native resolution by gray
level. Other patterns use multi-scale Gabor energy at a 2,000-pixel working
side, with the prototype taken from pattern pixels inside the query.
Threshold constants were chosen on real training queries before either
public-validation run.

An earlier public-validation run of the same segmenter downsampled tone fills
and scored 0.452178232295848. Inspecting those empty gray-hex masks showed
that downsampling mixed black lines into the fill. The submitted run keeps
tone matching at native resolution. The threshold numbers were not refit on
validation. Both evaluator outputs are unedited. The submitted file is the
second one.

The neural ensemble has not been trained or scored here.

There is no sealed `runs/final`, no `best.pt` ensemble, and no GPU run.
`configs/final.yaml` names `nvidia/mit-b2` at revision
`3bb39e8739149c3777d0325349b2a6c32c6413db`, but those checkpoints were not
trained or scored in this workspace. The 0.75 aspiration is not claimed.

## Visual examples

Existing prediction PNGs are not available to overlay on reviewed labels.
Doing so would require inventing labels. The report examples are representative fixtures until real validation predictions exist.

`tests/test_submission.py` builds three 4-by-4 fixture drawings and
`scripts/build_submission.py` renders each as four panels: grayscale drawing,
magenta query support `(255, 0, 255)`, green selected prediction
`(0, 255, 0)` on black, and a label panel. On the label panel, positive is
red `(255, 0, 0)`, negative is blue `(0, 0, 255)`, query support is yellow
`(255, 255, 0)`, and unknown is gray `(128, 128, 128)`. Positive overrides
negative, which overrides query support. This report does not display private-test material. The fixture panels are not public-validation
predictions and are not a measured score.

## Timing and hardware

No GPU was available. The submitted segmenter run used 4 CPU threads on an
Intel Xeon host with no CUDA device. `runs/segment-v2/timing.json` records
wall-clock inference of 112.520262125 seconds for 57 queries, including
loading, preprocessing, inference, and file writing. Peak resident memory was
927.171875 MiB from `resource.getrusage.ru_maxrss`. `time_spent_hours` in that
file is that same wall clock converted to hours, not a separately metered
development duration. The capped-1600 metrics file does not record processor
model, wall-clock time, or peak memory. When a sealed run has no nonnegative
`runtime_seconds` and `time_spent_hours`, `scripts/build_submission.py` exits
2 and writes nothing. The builder does not substitute zeros.
`submission.example.json` still uses 0 for those two fields so the example
file itself passes the validator. Those zeros are schema fillers, not
measurements.

## Data use

The classical baseline and the segmenter use the public dataset and no neural
pretrained weights. Public-validation labels are not inference inputs. They
were read by the official evaluator after prediction. After the first
segmenter evaluation, those labels showed empty masks on gray fills, which
led to the native-resolution tone fix. Threshold constants were not refit on
validation scores. No private-test labels were read.
Development of this repository was assisted by an automated coding agent.
That assistance did not train or score the neural ensemble.

## Limitations and failure cases

The capped-1600 baseline pooled recall is about 0.565. The submitted
segmenter pooled recall is about 0.750, still below the 95% aspiration.
Sparse stipple on `doc-2f69e1183867e83c` scores about 0.035 because the query
is a few gray dots and the same statistics appear in reviewed negatives.
The historical 37-query reference does not measure this split. Fixture
drawings do not show model behavior. The 0.75 aspiration is not claimed.
