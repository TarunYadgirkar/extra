# Metrics and measured reference

## Challenge scoring

The primary metric is **document-macro IoU over nonempty-target queries**: average eligible query IoUs within each document, then average document means. Query-macro IoU and pooled precision/recall use different weighting.

An empty-target query has IoU `null`; its false-positive rate is false positives divided by known pixels. Empty targets do not receive automatic perfect IoU. Zero denominators are `null`. Report eligible query/document counts alongside scores.

Only known pixels are scored, after excluding supplied query and context boxes. Unknown pixels are unscored. Reviewed-blank false-positive rate uses explicit blank domains only; other-pattern negatives are not automatically blank. See [SPECIFICATION.md](SPECIFICATION.md).

## Measured reference

These cached measurements come from a **separate, previously exposed development suite of 37 real queries across 12 documents**. Supplied supports were excluded; only known regions were scored. They are not fresh held-out results and are not measurements on the new challenge validation or private-test partitions.

| Metric | Measured reference |
|---|---:|
| Document-macro IoU | 73.73% |
| Query-macro IoU | 74.09% |
| Pooled pixel recall | 94.19% |
| Pooled pixel precision | 54.31% |
| Reviewed-blank false positives | 0 / 6,488,288 pixels |
| Empty-target queries | 0 |

[Machine-readable counts](data/reference-metrics.json) retain exact values and denominators. With no empty-target queries, this reference does not establish empty-target rejection performance. The zero reviewed-blank result applies only to labeled blank pixels in that historical suite.

## Aspirational target

Aim for **75%+ public-validation document-macro IoU**, with recall near or above **95%**, while improving precision and keeping reviewed-blank false positives low. These are aspirations based on historical context, not measured challenge results, guaranteed attainability, or application acceptance thresholds.

Submit unedited evaluator output and disclose development use of validation data. Keep public validation, organizer private test, and historical reference measurements separate.
