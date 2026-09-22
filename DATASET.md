# Dataset v1

[`data/release.json`](data/release.json) records exact split counts, download size and SHA-256. The collection combines reviewed real queries with explicitly identified generated CAD training examples. The private test is removed before packaging; it is not a hidden folder in the public ZIP.

## Splits

- **Train:** real drawings plus generated CAD examples.
- **Validation:** real drawings with labels, for model selection and error analysis.
- **Private test:** real document groups retained by TruTec. Neither images nor labels are distributed; submit runnable inference code.

Split by document/alias group before selecting queries. Multiple patterns, crops and aliases stay together. CAD variants remain in training. The test is held out from challenge participants; this does **not** claim the organizers have never used these drawings in development. No claim of project-level generalization follows from this split.

## Manifest

`train.json` and `val.json` contain `{ "schema": "hatch-matching-challenge/v1", "examples": [...] }`.

| Field | Meaning |
|---|---|
| `id` | Query ID; output filename `<id>.png` |
| `image` | Image path relative to dataset directory |
| `width`, `height` | Original dimensions |
| `document_id` | Split and document-macro group |
| `kind` | `real` or `generated_cad` |
| `query_box` | Exclusive-right/bottom `[x0,y0,x1,y1]` |
| `context_boxes` | Additional inputs, excluded from scoring |
| `labels` | Explicit mask paths and/or rectangles |

Optional label fields: `positive_mask`, `known_mask`, `negative_mask`, `blank_mask`, `positive_boxes`, `known_boxes`, `negative_boxes`, `blank_boxes`. Paths are dataset-relative. Masks are binary PNGs. Rectangles use native pixels; right/bottom edges are excluded. Known pixels outside positives count as negatives only inside the explicitly defined known domain.

**Unknown stays unknown.** Query/context rectangles are removed from every scored domain. Partial real annotations cover reviewed windows, not certified complete surfaces. Generated labels cover the attested visible composite and preserve unknown/occluded regions. A positive-only example cannot measure full-page precision.

## Reproducibility

Keep validation separate from training; disclose all data used. Labels must never be inference inputs. Predictions cover the full source resolution even when some pixels are unscored. `checksums.json` binds distributed images and masks; the downloader verifies both archive and files. Internal model details and private test materials are not part of the public schema.
