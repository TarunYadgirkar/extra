# Technical report

## Input and output contract

Inference reads a label-free request manifest and the original drawings. It
writes one native-resolution binary PNG per request id. The command accepted
by the official submission schema is:

```bash
python inference.py --inputs validation-inputs.json --data-root dataset --output-dir predictions --config configs/baseline-eval-1600.yaml
```

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

The only measured challenge score is the capped-1600 classical baseline document-macro IoU 0.4779169321.

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

No GPU was available. The capped-1600 metrics file does not record processor
model, wall-clock time, or peak memory. Timing coverage for a future sealed
bundle is wall clock including loading, preprocessing, inference, file
writing, and initialization. When a sealed run has no timing record,
`scripts/build_submission.py` writes null `runtime_seconds`,
`time_spent_hours`, and `peak_memory_mb`. Timing was not measured. The
recorded nulls are not measurements. The official validator accepts null
peak memory and rejects null `runtime_seconds` and `time_spent_hours`, so an
untimed bundle is not a validator-clean submission.
`submission.example.json` still uses 0 for those two fields so the example
file itself passes the validator. Those zeros are schema fillers, not
measurements.

## Data use

The classical baseline uses the public dataset and no neural pretrained
weights. Public-validation labels were read by the official evaluator after
prediction to produce the capped-1600 metrics. They are not inference inputs
and were not used to select an ensemble. No private-test labels were read.
Development of this repository was assisted by an automated coding agent.
That assistance did not train or score the neural ensemble.

## Limitations and failure cases

The capped working resolution can drop fine hatch. Pooled recall is about
0.565, below the 95% aspiration, and one public-validation document scored
0. The fixed threshold is 0.5. The historical 37-query reference does not
measure this split. Fixture drawings do not show model behavior. A submission
bundle cannot be completed from `runs/final` until an unedited sealed metrics
file exists.
