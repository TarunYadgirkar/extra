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
`pretrained=False`. Default pretrained B2 construction uses repository-pinned
revision `3bb39e8739149c3777d0325349b2a6c32c6413db`; other backbones require an
explicit 40-character Hugging Face revision.

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

- The network-loading path is tested through its real call boundary with a
  focused monkeypatch, but artifact download and byte-level validation remain
  intentionally unexercised in offline tests.

## Review fixes: RED evidence

Command:

```text
python3 -m pytest tests/test_model.py tests/test_losses.py -v
```

Observed after adding review regressions and before changing production code:

```text
9 failed, 21 passed, 1 warning in 6.39s
```

The failures demonstrated:

- B1-B5 incorrectly reported `decoder_hidden_size=256` instead of `768`
- default pretrained B2 construction rejected the missing implicit pin
- decoder widths `0` and `1` did not produce the intended validation error
- decoder width `10` was rejected by an artificial divisibility restriction

The normalized-similarity hook test, FiLM/FPN/texture wiring test, and exact
boundary oracle passed during RED. Those tests strengthen regression coverage
for behavior that was already correct rather than claiming a corresponding
production defect.

## Review fixes: GREEN evidence

Focused verification:

```text
python3 -m pytest tests/test_model.py tests/test_losses.py -v
30 passed in 5.50s
```

Full verification:

```text
python3 -m pytest -v
104 passed in 6.62s
```

Review-fix commit:

- `f98d971` — `fix: pin and verify query segmenter architecture`
