# Reproducibility audit

This record is the clean-environment audit run on this host. It is a CPU-only
check of installation, the full pytest suite, and one label-free inference
request. It is not a scored neural-ensemble run.

## Hardware

- CPU only. `nvidia-smi` is not installed. `torch.cuda.is_available()` is
  false, and the inference metadata selected device `cpu`.
- Architecture: `x86_64`.
- Processor: Intel(R) Xeon(R) Processor, 4 CPUs, KVM hypervisor.
- Kernel platform reported by the inference run:
  `Linux-6.12.94+-x86_64-with-glibc2.39`.
- Peak memory was not measured.

## Clean environment

This host has no `python` binary until a virtualenv exists. `python3 -m venv`
also failed until `python3.12-venv` was installed, because `ensurepip` was
missing. The successful setup, from `/workspace`, was:

```bash
sudo apt-get update -qq && sudo apt-get install -y -qq python3.12-venv strace
rm -rf /tmp/hatch-clean
/usr/bin/python3 -m venv /tmp/hatch-clean
/tmp/hatch-clean/bin/python -m pip install --upgrade pip
/tmp/hatch-clean/bin/python -m pip install -e '/workspace[dev]'
```

Installation exited 0. The editable install is `hatch-matching` 0.1.0.
`pip` in the virtualenv ended at 26.2.1. The interpreter is Python 3.12.3.

## Full test suite

Wall time was measured by recording `date +%s.%N` immediately before and after
the pytest process. The difference was 44.996 seconds. Peak memory was not
measured. Pytest's own timer is the separate line below.

```bash
cd /workspace
/tmp/hatch-clean/bin/python -m pytest -v
```

Result: exit 0. The session header was
`platform linux -- Python 3.12.3, pytest-9.1.1`. The summary was
`222 passed in 43.90s`.

## Label-free inference, one official-schema request

`configs/final.yaml` was not used for this prediction. Invoking it exits 2
because the named `best.pt` checkpoints are missing. The 57 native
public-validation drawings were not run. The request below is the first
example of `validation-inputs.json`, kept in the official schema
`hatch-matching-inputs/v1`, with fields `id`, `image`, `width`, `height`,
`query_box`, and `context_boxes` only.

```bash
/tmp/hatch-clean/bin/python - <<'PY'
import json
from pathlib import Path
src = json.loads(Path("validation-inputs.json").read_text())
one = {"schema": src["schema"], "examples": [src["examples"][0]]}
Path("/tmp/hatch-one-request.json").write_text(
    json.dumps(one, indent=2) + "\n", encoding="utf-8"
)
PY
strace -f -e trace=openat,open -o /tmp/hatch-one-opens.log \
  /tmp/hatch-clean/bin/python /workspace/inference.py \
  --inputs /tmp/hatch-one-request.json \
  --data-root /workspace/dataset \
  --output-dir /tmp/hatch-one-predictions \
  --config /workspace/configs/test.yaml
```

The request id is `q-054ce5dd82203aa7`. The drawing is
`dataset/assets/abb125d94a45636a9bf1ea7b504a276688d8e7eaf13ad638bd8dd2a81f01c2da.png`
at 7200 by 4800. `configs/test.yaml` is classical mode with threshold 0.5
and schema `hatchmatch-inference/v1`. Its SHA-256, recorded in
`run-metadata.json`, is
`2ff340f7c41c0886523eebcc2dc9f37a504b21222f9a67553274c678dc11bacc`.

Result: exit 0. Outputs:

- `/tmp/hatch-one-predictions/q-054ce5dd82203aa7.png` (mode `L`, 7200 by 4800,
  pixel extrema 0 and 255, 185875 bytes, SHA-256
  `2f167c72445c8a4d390b3a359a4881cc8a0130569a4079fe26a3f38c0bee5928`)
- `/tmp/hatch-one-predictions/run-metadata.json`

The metadata reports `count` 1, `cuda_available` false, device `cpu`, empty
`checkpoint_hashes`, and `elapsed_wall_seconds` 125.96036594999896. The outer
`date` interval around the traced process was 132.863 seconds. That interval
is inference timing, not the pytest wall time above.

`strace` recorded 13342 `open` and `openat` lines. The only path under
`dataset/` was the request drawing, opened twice. These files exist on disk
and were not opened:

- `dataset/val.json`
- `dataset/train.json`
- `dataset/expected-counts.json`
- `dataset/assets/8653a10df4fc3044a1ae2a9e8d0c51b0bc973ca33cdd3c5165989bdd038b1aa6.png`
  (the `known_mask` named for this id in `dataset/val.json`)
- `dataset/assets/56bc73d3f48314e151645e8e3b55593a56a75bfdd8f077e504bfe15e298c4962.png`
  (the `positive_mask` named for this id in `dataset/val.json`)

## Capped-1600 baseline score

The existing evaluator file is `runs/baseline-eval-1600/metrics.json`.
That directory is gitignored; the file is already present in this workspace.
Its SHA-256 is
`04e1ac742653c468b52f8470bf8829273589d8b6412433ccad674de7d931667c`
(46716 bytes). `summary.document_macro_iou` is 0.4779169321376195, the
capped-1600 classical baseline score reported as 0.4779169321. The same
summary records 57 examples, 13 documents, mean query IoU
0.43592916774392726, pooled precision 0.4827817373038098, and pooled recall
0.5652035245486439.

This audit did not run `challenge/evaluate.py`. The one-request PNG above is
not that metrics file and is not a challenge score.

## Neural ensemble

The neural ensemble remains unscored. There is no GPU run and no sealed
`runs/final` score from this audit. `configs/final.yaml` names fifteen
`runs/train-b2/fold-*/seed-*/best.pt` checkpoints. The following command
exits 2 and writes no output directory:

```bash
/tmp/hatch-clean/bin/python /workspace/inference.py \
  --inputs /tmp/hatch-one-request.json \
  --data-root /workspace/dataset \
  --output-dir /tmp/hatch-final-should-not-exist \
  --config /workspace/configs/final.yaml
```

Stderr begins with `error: refusing to skip missing checkpoint file(s):`.
