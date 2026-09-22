# Submission requirements

Email **[ryan@trutec.ai](mailto:ryan@trutec.ai)** with the repository and artifact links below. Include `submission.json`, based on [submission.example.json](submission.example.json).

## Reproducibility

Provide:

1. Repository URL and immutable full commit SHA.
2. Environment specification fixing interpreter, dependencies, installation steps, and required system dependencies.
3. Exact inference command supporting `--inputs`, `--data-root`, and `--output-dir` under [SPECIFICATION.md](SPECIFICATION.md).
4. Required executable artifacts and parameter files, each with a download location, byte length, and SHA-256 checksum. Declare additional data or network dependencies.
5. Dataset release version and evaluator revision.

The organizer must be able to supply label-free requests and original drawings through the same interface. Annotation masks, annotation rectangles, expected counts, and private labels must not be read during inference.

## Evaluation artifacts

Submit native-resolution prediction PNGs for every public validation ID and the **unedited** output of:

```bash
python evaluate.py --manifest dataset/val.json --data-root dataset --predictions predictions --output metrics.json
```

Supply checksums for prediction archives and metrics files. Do not omit failed examples or hand-edit metrics. Disclose all training, adaptation, and external data, including pretrained artifacts. State whether validation labels or results influenced development, selection, or parameter choices.

Report elapsed inference time, processed query count, and hardware: processor, accelerators if used, and memory. Define timing coverage, including loading, preprocessing, inference, file writing, and initialization. If peak memory is measured, report its measurement method; distinguish measurements from estimates.

## Short technical report

Describe the submitted input/output contract, reproducibility, measured results, limitations, and failure cases. Include **three visual examples** with the original drawing, query region, prediction, and available annotation overlay. Mark positive, negative, and unscored pixels distinctly. Identify public example IDs; do not display private-test material.

Report document-macro IoU, supporting evaluator metrics, and eligible counts. Keep historical reference numbers separate from measurements on the released validation set.

## Validate and send

Complete [submission.example.json](submission.example.json), save it as `submission.json`, and run:

```bash
python validate_submission.py --submission submission.json
```

Send the manifest, repository/commit, report, and artifact links to **ryan@trutec.ai**. Dataset redistribution is governed by [DATA_TERMS.md](DATA_TERMS.md). The organizer retains private labels and evaluates private requests separately.
