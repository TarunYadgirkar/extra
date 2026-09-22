# Task 5 Report: Query-conditioned dense model and losses

## Status

Implemented the Task 5 model and losses without adding a training loop.

The model uses one shared MiT encoder instance for drawing and query inputs,
per-level pooled query prototypes, channel-normalized dot-product similarity,
FiLM conditioning in a top-down FPN, and four-channel texture fusion before two
final convolution blocks. Native output dimensions are restored explicitly, so
non-multiple input sizes are supported.

The objective combines masked focal, soft Dice, and boundary losses with
weights `0.5`, `0.4`, and `0.1`. Unknown logits and targets are replaced before
any loss calculation. Boundary comparisons are included only when both
adjacent endpoints are known. Empty known regions produce differentiable zero.

Offline construction supports canonical NVIDIA MiT B0-B5 configurations when
`pretrained=False`. Loading pretrained weights requires an explicit
40-character Hugging Face revision so the future artifact cannot be loaded
from a mutable branch or tag.

## RED evidence

Command:

```text
python3 -m pytest tests/test_model.py tests/test_losses.py -v
```

Observed before production modules were created:

```text
collected 0 items / 2 errors
ModuleNotFoundError: No module named 'hatchmatch.model'
ModuleNotFoundError: No module named 'hatchmatch.losses'
2 errors in 0.99s
```

The failures were the expected missing-feature failures.

## GREEN evidence

Focused verification:

```text
python3 -m pytest tests/test_model.py tests/test_losses.py -v
17 passed in 3.97s
```

Full verification:

```text
python3 -m pytest -v
91 passed in 4.69s
```

Coverage added for:

- B2 offline construction, query conditioning, and arbitrary `65x79` output
- shared drawing/query encoder object identity
- immutable revision enforcement for pretrained weights
- texture-channel conditioning
- gradients through image, query, texture, shared encoder, and FiLM paths
- exact focal, soft Dice, boundary, and weighted composite behavior
- unknown-logit and unknown-target invariance for every loss component
- differentiable zero with zero known pixels for every loss component

## Self-review

- Confirmed only one encoder module is registered and reused for both inputs.
- Confirmed similarity normalizes both drawing features and query prototypes.
- Confirmed every decoder interpolation uses explicit target dimensions.
- Confirmed unknown values are removed before focal, Dice, or boundary math.
- Confirmed boundary masks require two known endpoints.
- Confirmed no training-loop code was added.

## Commits

- `fcb638e` — `feat: add query-conditioned hatch segmenter`
- `de7a897` — `refactor: hard-mask unknown loss inputs`

## Concerns

- The network-loading path was not exercised because this task's tests must run
  offline. It delegates to `SegformerModel.from_pretrained` and requires a
  pinned revision; the final chosen revision and checksum remain release
  configuration work.
