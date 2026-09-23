import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from hatchmatch.calibrate import (
    FusionCalibrator,
    FusionChannels,
    document_macro_iou,
    main,
    postprocess,
    search_objective,
)


def test_postprocess_removes_isolated_pixel():
    probability = np.zeros((32, 32), np.float32)
    probability[2, 2] = 1.0
    probability[12:20, 12:20] = 1.0
    mask = postprocess(probability, np.ones_like(probability), {
        "threshold": .5, "min_component": 8, "close_radius": 0})
    assert not mask[2, 2]
    assert mask[15, 15]
    assert mask.dtype == np.bool_
    assert mask.shape == probability.shape


def test_postprocess_applies_foreground_floor_and_closing():
    probability = np.ones((8, 8), np.float32)
    foreground = np.zeros((8, 8), np.float32)
    foreground[2:6, 2:6] = 1.0
    kept = postprocess(
        probability,
        foreground,
        {
            "threshold": 0.5,
            "min_component": 1,
            "close_radius": 0,
            "foreground_floor": 0.5,
        },
    )
    assert kept[3, 3]
    assert not kept[0, 0]

    gapped = np.zeros((8, 8), np.float32)
    gapped[3, 1:3] = 1.0
    gapped[3, 4:6] = 1.0
    foreground = np.ones_like(gapped)
    config = {
        "threshold": 0.5,
        "min_component": 1,
        "foreground_floor": 0.0,
    }
    opened = postprocess(gapped, foreground, {**config, "close_radius": 0})
    closed = postprocess(gapped, foreground, {**config, "close_radius": 1})
    assert not opened[3, 3]
    assert closed[3, 3]


def _channels(
    neural: np.ndarray,
    *,
    document_id: str,
    example_id: str,
    trained_example_ids: frozenset[str],
    classical: np.ndarray | None = None,
    source_fold: int = 0,
    kind: str = "real",
) -> FusionChannels:
    shape = neural.shape
    return FusionChannels(
        neural=neural.astype(np.float32),
        classical=(
            neural.astype(np.float32) if classical is None else classical
        ),
        foreground=np.full(shape, 0.5, np.float32),
        local_variance=np.full(shape, 0.25, np.float32),
        query_compatibility=np.full(shape, 0.4, np.float32),
        document_id=document_id,
        example_id=example_id,
        source_fold=source_fold,
        trained_example_ids=trained_example_ids,
        kind=kind,
    )


def test_logistic_fusion_uses_known_pixels_only():
    neural = np.zeros((20, 20), np.float32)
    target = np.zeros((20, 20), bool)
    known = np.zeros((20, 20), bool)
    neural[:, 0:5] = 0.95
    neural[:, 5:10] = 0.05
    target[:, 0:5] = True
    known[:, 0:10] = True
    neural[:, 10:15] = 0.95
    neural[:, 15:20] = 0.05
    target[:, 15:20] = True

    calibrator = FusionCalibrator(max_pixels_per_document=512, seed=7)
    fitted = calibrator.fit(
        [
            _channels(
                neural,
                document_id="doc",
                example_id="held-out",
                trained_example_ids=frozenset({"other"}),
            )
        ],
        [target],
        [known],
    )

    assert fitted is calibrator
    probe_shape = (4, 4)
    high = calibrator.predict(
        [
            _channels(
                np.full(probe_shape, 0.95, np.float32),
                document_id="doc",
                example_id="probe-high",
                trained_example_ids=frozenset(),
            )
        ]
    )[0]
    low = calibrator.predict(
        [
            _channels(
                np.full(probe_shape, 0.05, np.float32),
                document_id="doc",
                example_id="probe-low",
                trained_example_ids=frozenset(),
            )
        ]
    )[0]
    assert high.shape == probe_shape
    assert high.dtype == np.float32
    assert float(high.mean()) > float(low.mean()) + 0.2


def test_logistic_fusion_balances_and_caps_pixels_per_document():
    def _labeled(size: int, document_id: str, example_id: str):
        neural = np.zeros((size, size), np.float32)
        target = np.zeros((size, size), bool)
        neural[:, : size // 2] = 0.9
        neural[:, size // 2 :] = 0.1
        target[:, : size // 2] = True
        known = np.ones((size, size), bool)
        return (
            _channels(
                neural,
                document_id=document_id,
                example_id=example_id,
                trained_example_ids=frozenset({"train-only"}),
            ),
            target,
            known,
        )

    huge, huge_target, huge_known = _labeled(40, "huge", "huge-query")
    tiny, tiny_target, tiny_known = _labeled(10, "tiny", "tiny-query")
    blank_neural = np.zeros((8, 8), np.float32)
    blank = _channels(
        blank_neural,
        document_id="blank-doc",
        example_id="blank-query",
        trained_example_ids=frozenset({"train-only"}),
    )
    calibrator = FusionCalibrator(max_pixels_per_document=20, seed=1)
    calibrator.fit(
        [huge, tiny, blank],
        [huge_target, tiny_target, np.zeros((8, 8), bool)],
        [huge_known, tiny_known, np.ones((8, 8), bool)],
    )

    assert calibrator.pixels_per_document == {
        "blank-doc": 0,
        "huge": 20,
        "tiny": 20,
    }
    assert calibrator.sampled_positive_count == 20
    assert calibrator.sampled_negative_count == 20


def test_fit_refuses_in_fold_labels():
    neural = np.zeros((8, 8), np.float32)
    neural[:, :4] = 0.9
    target = np.zeros((8, 8), bool)
    target[:, :4] = True
    known = np.ones((8, 8), bool)
    held_out = _channels(
        neural,
        document_id="doc-valid",
        example_id="q-valid",
        trained_example_ids=frozenset({"q-train"}),
    )
    in_fold = _channels(
        neural,
        document_id="doc-train",
        example_id="q-train",
        trained_example_ids=frozenset({"q-train"}),
        source_fold=1,
    )
    calibrator = FusionCalibrator(max_pixels_per_document=32, seed=0)

    with pytest.raises(ValueError, match="in-fold"):
        calibrator.fit(
            [held_out, in_fold],
            [target, target],
            [known, known],
        )

    assert not hasattr(calibrator, "pixels_per_document")


def test_document_macro_iou_averages_queries_then_documents_and_skips_empty():
    # doc-a nonempty IoUs are 1 and 0.5. doc-b is 1. Empty targets are
    # omitted, so the macro is 0.875. Scoring the empty query as perfect
    # would raise doc-a and the macro; a query macro would ignore documents.
    match = np.array([[1, 1, 0, 0]], dtype=bool)
    target = np.array([[1, 1, 0, 0]], dtype=bool)
    broad = np.array([[1, 1, 1, 1]], dtype=bool)
    empty_target = np.zeros((1, 4), dtype=bool)
    empty_prediction = np.ones((1, 4), dtype=bool)
    known = np.ones((1, 4), dtype=bool)
    predictions = [match, broad, empty_prediction, match]
    targets = [target, target, empty_target, target]
    knowns = [known, known, known, known]
    document_ids = ["doc-a", "doc-a", "doc-a", "doc-b"]

    metric = document_macro_iou(predictions, targets, knowns, document_ids)

    assert metric == pytest.approx(0.875)
    assert metric != pytest.approx((1.0 + 0.5 + 1.0) / 3.0)
    assert metric != pytest.approx(((1.0 + 0.5 + 1.0) / 3.0 + 1.0) / 2.0)


def test_document_macro_iou_ignores_unknown_pixels():
    target = np.array([[1, 0]], dtype=bool)
    known = np.array([[1, 0]], dtype=bool)
    low_unknown = np.array([[1, 0]], dtype=bool)
    high_unknown = np.array([[1, 1]], dtype=bool)

    assert document_macro_iou(
        [low_unknown],
        [target],
        [known],
        ["doc"],
    ) == document_macro_iou(
        [high_unknown],
        [target],
        [known],
        ["doc"],
    ) == pytest.approx(1.0)


def test_objective_is_document_macro_iou_when_recall_is_at_least_0_95():
    match = np.array([[1, 1, 0, 0]], dtype=bool)
    target = np.array([[1, 1, 0, 0]], dtype=bool)
    broad = np.array([[1, 1, 1, 1]], dtype=bool)
    known = np.ones((1, 4), dtype=bool)
    predictions = [match, broad, match]
    targets = [target, target, target]
    knowns = [known, known, known]
    document_ids = ["doc-a", "doc-a", "doc-b"]

    objective, recall = search_objective(
        predictions,
        targets,
        knowns,
        document_ids,
        recall_floor=0.95,
    )

    assert recall == pytest.approx(1.0)
    assert objective == pytest.approx(0.875)
    assert objective == pytest.approx(
        document_macro_iou(predictions, targets, knowns, document_ids)
    )


def test_recall_below_0_95_is_penalized_below_every_feasible_iou():
    target = np.zeros((2, 100), dtype=bool)
    target[0] = True
    known = np.ones_like(target)
    low_recall = np.zeros_like(target)
    low_recall[0, :94] = True
    high_recall = np.ones_like(target)
    exact = np.zeros((1, 100), dtype=bool)
    exact[0, :95] = True
    exact_target = np.ones((1, 100), dtype=bool)
    exact_known = np.ones((1, 100), dtype=bool)

    penalized, penalized_recall = search_objective(
        [low_recall],
        [target],
        [known],
        ["doc"],
        recall_floor=0.95,
    )
    feasible, feasible_recall = search_objective(
        [high_recall],
        [target],
        [known],
        ["doc"],
        recall_floor=0.95,
    )
    exact_objective, exact_recall = search_objective(
        [exact],
        [exact_target],
        [exact_known],
        ["doc"],
        recall_floor=0.95,
    )

    assert penalized_recall == pytest.approx(0.94)
    assert feasible_recall == pytest.approx(1.0)
    assert exact_recall == pytest.approx(0.95)
    assert penalized < 0.0
    assert feasible == pytest.approx(0.5)
    assert feasible > penalized
    assert exact_objective == pytest.approx(0.95)


def test_recall_penalty_uses_pooled_not_macro_recall():
    # One perfect pixel and 90/100 elsewhere is macro-recall 0.95, but the
    # pooled recall is 91/101 and must still be penalized.
    perfect = np.array([[1]], dtype=bool)
    partial = np.zeros((1, 100), dtype=bool)
    partial[0, :90] = True
    partial_target = np.ones((1, 100), dtype=bool)
    known_partial = np.ones((1, 100), dtype=bool)

    objective, recall = search_objective(
        [perfect, partial],
        [perfect, partial_target],
        [np.ones((1, 1), dtype=bool), known_partial],
        ["doc-a", "doc-b"],
        recall_floor=0.95,
    )

    assert recall == pytest.approx(91 / 101)
    assert recall < 0.95
    assert objective < 0.0
    assert objective < document_macro_iou(
        [perfect, partial],
        [perfect, partial_target],
        [np.ones((1, 1), dtype=bool), known_partial],
        ["doc-a", "doc-b"],
    )


def _search_config(output_dir: Path, **overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "seed": 20260923,
        "n_trials": 2,
        "recall_floor": 0.95,
        "max_pixels_per_document": 64,
        "parameters": {
            "threshold": {"low": 0.2, "high": 0.8},
            "min_component": {"low": 1, "high": 4},
            "close_radius": {"low": 0, "high": 1},
            "foreground_floor": {"low": 0.0, "high": 0.2},
        },
    }
    config.update(overrides)
    config["output_dir"] = str(output_dir)
    return config


def _separable_channels() -> tuple[FusionChannels, np.ndarray, np.ndarray]:
    neural = np.zeros((8, 8), np.float32)
    neural[:, :4] = 0.92
    target = np.zeros((8, 8), bool)
    target[:, :4] = True
    known = np.ones((8, 8), bool)
    return (
        _channels(
            neural,
            document_id="doc-valid",
            example_id="q-valid",
            trained_example_ids=frozenset({"q-train"}),
            source_fold=0,
        ),
        target,
        known,
    )


def test_search_is_deterministic_and_records_the_real_objective(tmp_path: Path):
    from hatchmatch.calibrate import search_postprocess

    channels, target, known = _separable_channels()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = search_postprocess(
        [channels],
        [target],
        [known],
        _search_config(first_dir),
    )
    second = search_postprocess(
        [channels],
        [target],
        [known],
        _search_config(second_dir),
    )

    assert first["parameters"] == second["parameters"]
    assert first["objective"] == second["objective"]
    assert first["recall"] == second["recall"]
    assert first["study_seed"] == 20260923
    assert first["official_public_validation"] is False
    assert set(first["parameters"]) == {
        "threshold",
        "min_component",
        "close_radius",
        "foreground_floor",
    }
    calibrator = FusionCalibrator.from_dict(first["fusion"])
    fused = calibrator.predict([channels])[0]
    mask = postprocess(fused, channels.foreground, first["parameters"])
    objective, recall = search_objective(
        [mask],
        [target],
        [known],
        [channels.document_id],
        recall_floor=0.95,
    )
    assert first["objective"] == pytest.approx(objective)
    assert first["recall"] == pytest.approx(recall)

    for output_dir in (first_dir, second_dir):
        payload = json.loads((output_dir / "best.json").read_text())
        assert payload["objective"] == first["objective"]
        assert payload["recall"] == first["recall"]
        assert payload["study_seed"] == 20260923
        assert payload["official_public_validation"] is False
        assert (output_dir / "study.db").is_file()
        trials = json.loads((output_dir / "trials.json").read_text())
        assert len(trials["trials"]) == 2
        provenance = json.loads(
            (output_dir / "fold-provenance.json").read_text()
        )
        assert provenance["examples"][0]["example_id"] == "q-valid"
        assert provenance["examples"][0]["out_of_fold"] is True
        assert "q-valid" not in provenance["examples"][0]["trained_example_ids"]


def _save_mask(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8) * 255, mode="L").save(path)


def _write_labeled_example(
    root: Path,
    *,
    example_id: str,
    document_id: str,
) -> None:
    image = np.full((8, 8), 240, np.uint8)
    image[2:6, 2:6] = 20
    Image.fromarray(image, mode="L").save(root / f"{example_id}.png")
    positive = np.zeros((8, 8), bool)
    positive[2:6, 2:6] = True
    known = np.ones((8, 8), bool)
    _save_mask(root / f"{example_id}-positive.png", positive)
    _save_mask(root / f"{example_id}-known.png", known)


def _write_search_fixture(root: Path, *, example_id: str) -> Path:
    _write_labeled_example(root, example_id="q-train", document_id="doc-train")
    _write_labeled_example(root, example_id="q-valid", document_id="doc-valid")
    manifest = {
        "schema": "hatch-matching-challenge/v1",
        "examples": [
            {
                "id": example_id,
                "document_id": f"doc-{example_id.split('-')[1]}",
                "kind": "real",
                "image": f"{example_id}.png",
                "width": 8,
                "height": 8,
                "query_box": [0, 0, 1, 1],
                "context_boxes": [],
                "labels": {
                    "positive_mask": f"{example_id}-positive.png",
                    "known_mask": f"{example_id}-known.png",
                },
            }
            for example_id in ("q-train", "q-valid")
        ],
    }
    training_manifest = root / "train.json"
    training_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    fold_manifest = root / "folds.json"
    fold_manifest.write_text(
        json.dumps(
            {
                "schema": "hatchmatch-group-folds/v1",
                "folds": [
                    {
                        "fold": 0,
                        "train_ids": ["q-train"],
                        "valid_ids": ["q-valid"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    neural = np.full((8, 8), 0.05, np.float32)
    neural[2:6, 2:6] = 0.95
    np.savez(
        root / "q.npz",
        neural=neural,
        classical=neural.copy(),
        foreground=np.ones((8, 8), np.float32),
        local_variance=np.full((8, 8), 0.3, np.float32),
        query_compatibility=neural.copy(),
    )
    oof_manifest = root / "oof.json"
    oof_manifest.write_text(
        json.dumps(
            {
                "schema": "hatchmatch-oof-maps/v1",
                "examples": [{"id": example_id, "fold": 0, "path": "q.npz"}],
            }
        ),
        encoding="utf-8",
    )
    config_path = root / "search.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema": "hatchmatch-calibration-search/v1",
                "seed": 20260923,
                "recall_floor": 0.95,
                "n_trials": 2,
                "max_pixels_per_document": 64,
                "output_dir": "calibration-out",
                "oof_manifest": "oof.json",
                "fold_manifest": "folds.json",
                "training_manifest": "train.json",
                "data_root": ".",
                "parameters": {
                    "threshold": {"low": 0.2, "high": 0.8},
                    "min_component": {"low": 1, "high": 4},
                    "close_radius": {"low": 0, "high": 1},
                    "foreground_floor": {"low": 0.0, "high": 0.2},
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_search_command_fails_when_out_of_fold_maps_are_absent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["--config", "configs/search.yaml"])
    captured = capsys.readouterr()

    assert code == 2
    assert "out-of-fold" in captured.err.lower()
    assert "absent" in captured.err.lower()
    assert not Path("runs/calibration/best.json").exists()


def test_search_command_records_out_of_fold_objective(tmp_path: Path) -> None:
    config_path = _write_search_fixture(tmp_path, example_id="q-valid")

    assert main(["--config", str(config_path)]) == 0

    payload = json.loads((tmp_path / "calibration-out" / "best.json").read_text())
    assert payload["official_public_validation"] is False
    assert payload["study_seed"] == 20260923
    assert {"objective", "recall", "parameters"} <= set(payload)
    assert (tmp_path / "calibration-out" / "study.db").is_file()
    provenance = json.loads(
        (tmp_path / "calibration-out" / "fold-provenance.json").read_text()
    )
    assert provenance["examples"][0]["example_id"] == "q-valid"
    assert provenance["examples"][0]["out_of_fold"] is True


def test_search_command_refuses_in_fold_prediction_maps(tmp_path: Path) -> None:
    config_path = _write_search_fixture(tmp_path, example_id="q-train")

    with pytest.raises(ValueError, match="in-fold"):
        main(["--config", str(config_path)])

    assert not (tmp_path / "calibration-out" / "best.json").exists()


def test_oof_maps_reject_embedded_labels(tmp_path: Path) -> None:
    from hatchmatch.calibrate import load_out_of_fold_examples

    config_path = _write_search_fixture(tmp_path, example_id="q-valid")
    np.savez(
        tmp_path / "q.npz",
        neural=np.zeros((8, 8), np.float32),
        classical=np.zeros((8, 8), np.float32),
        foreground=np.ones((8, 8), np.float32),
        local_variance=np.zeros((8, 8), np.float32),
        query_compatibility=np.zeros((8, 8), np.float32),
        target=np.zeros((8, 8), np.uint8),
    )
    config = yaml.safe_load(config_path.read_text())

    with pytest.raises(ValueError, match="labels"):
        load_out_of_fold_examples(config, tmp_path)


def _polarity_example(
    size: int,
    *,
    positive_on_left: bool,
    document_id: str,
    example_id: str,
    kind: str,
) -> tuple[FusionChannels, np.ndarray, np.ndarray]:
    neural = np.zeros((size, size), np.float32)
    neural[:, : size // 2] = 0.95
    neural[:, size // 2 :] = 0.05
    target = np.zeros((size, size), bool)
    if positive_on_left:
        target[:, : size // 2] = True
    else:
        target[:, size // 2 :] = True
    known = np.ones((size, size), bool)
    channel = _channels(
        neural,
        document_id=document_id,
        example_id=example_id,
        trained_example_ids=frozenset({"train-only"}),
        classical=np.full(neural.shape, 0.5, np.float32),
        kind=kind,
    )
    return channel, target, known


def test_generated_cad_does_not_change_fusion_or_document_objective(
    tmp_path: Path,
) -> None:
    from hatchmatch.calibrate import search_postprocess

    real = _polarity_example(
        16,
        positive_on_left=True,
        document_id="doc-real",
        example_id="q-real",
        kind="real",
    )
    cad = _polarity_example(
        32,
        positive_on_left=False,
        document_id="doc-cad",
        example_id="q-cad",
        kind="generated_cad",
    )
    overrides = {"max_pixels_per_document": 4096, "n_trials": 2}
    real_only = search_postprocess(
        [real[0]],
        [real[1]],
        [real[2]],
        _search_config(tmp_path / "real", **overrides),
    )
    with_cad = search_postprocess(
        [real[0], cad[0]],
        [real[1], cad[1]],
        [real[2], cad[2]],
        _search_config(tmp_path / "both", **overrides),
    )

    assert with_cad["official_public_validation"] is False
    assert with_cad["fusion"]["coefficients"] == real_only["fusion"]["coefficients"]
    assert with_cad["fusion"]["intercept"] == real_only["fusion"]["intercept"]
    assert with_cad["parameters"] == real_only["parameters"]
    assert with_cad["objective"] == real_only["objective"]
    provenance = json.loads((tmp_path / "both" / "fold-provenance.json").read_text())
    assert [item["example_id"] for item in provenance["examples"]] == ["q-real"]

    contaminated = FusionCalibrator(
        max_pixels_per_document=4096,
        seed=20260923,
    )
    contaminated.fit(
        [real[0], cad[0]],
        [real[1], cad[1]],
        [real[2], cad[2]],
    )
    assert real_only["fusion"]["coefficients"][0] > 0
    assert contaminated.to_dict()["coefficients"][0] < 0

    calibrator = FusionCalibrator.from_dict(real_only["fusion"])
    parameters = real_only["parameters"]
    real_mask = postprocess(
        calibrator.predict([real[0]])[0],
        real[0].foreground,
        parameters,
    )
    cad_mask = postprocess(
        calibrator.predict([cad[0]])[0],
        cad[0].foreground,
        parameters,
    )
    real_objective, _ = search_objective(
        [real_mask],
        [real[1]],
        [real[2]],
        ["doc-real"],
    )
    mixed_objective, _ = search_objective(
        [real_mask, cad_mask],
        [real[1], cad[1]],
        [real[2], cad[2]],
        ["doc-real", "doc-cad"],
    )
    assert mixed_objective != pytest.approx(real_objective)
    assert with_cad["objective"] == pytest.approx(real_objective)


def test_search_on_generated_cad_only_writes_no_score(tmp_path: Path) -> None:
    from hatchmatch.calibrate import search_postprocess

    cad = _polarity_example(
        8,
        positive_on_left=False,
        document_id="doc-cad",
        example_id="q-cad",
        kind="generated_cad",
    )
    output_dir = tmp_path / "cad-only"

    with pytest.raises(ValueError, match="real"):
        search_postprocess(
            [cad[0]],
            [cad[1]],
            [cad[2]],
            _search_config(output_dir, n_trials=1),
        )

    assert not (output_dir / "best.json").exists()


def _write_kind_fixture(root: Path, *, include_real: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    examples = []
    oof_examples = []
    valid_ids = []
    specs = [("q-train", "doc-train", "real", False)]
    if include_real:
        specs.append(("q-real", "doc-real", "real", True))
        valid_ids.append("q-real")
    specs.append(("q-cad", "doc-cad", "generated_cad", True))
    valid_ids.append("q-cad")
    for example_id, document_id, kind, held_out in specs:
        _write_labeled_example(
            root,
            example_id=example_id,
            document_id=document_id,
        )
        examples.append(
            {
                "id": example_id,
                "document_id": document_id,
                "kind": kind,
                "image": f"{example_id}.png",
                "width": 8,
                "height": 8,
                "query_box": [0, 0, 1, 1],
                "context_boxes": [],
                "labels": {
                    "positive_mask": f"{example_id}-positive.png",
                    "known_mask": f"{example_id}-known.png",
                },
            }
        )
        if not held_out:
            continue
        neural = np.full((8, 8), 0.05, np.float32)
        if kind == "real":
            neural[2:6, 2:6] = 0.95
        else:
            neural[:, :] = 0.95
            neural[2:6, 2:6] = 0.05
        map_name = f"{example_id}.npz"
        np.savez(
            root / map_name,
            neural=neural,
            classical=neural.copy(),
            foreground=np.ones((8, 8), np.float32),
            local_variance=np.full((8, 8), 0.3, np.float32),
            query_compatibility=neural.copy(),
        )
        oof_examples.append({"id": example_id, "fold": 0, "path": map_name})
    (root / "train.json").write_text(
        json.dumps(
            {"schema": "hatch-matching-challenge/v1", "examples": examples}
        ),
        encoding="utf-8",
    )
    (root / "folds.json").write_text(
        json.dumps(
            {
                "schema": "hatchmatch-group-folds/v1",
                "folds": [
                    {
                        "fold": 0,
                        "train_ids": ["q-train"],
                        "valid_ids": valid_ids,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "oof.json").write_text(
        json.dumps(
            {"schema": "hatchmatch-oof-maps/v1", "examples": oof_examples}
        ),
        encoding="utf-8",
    )
    config_path = root / "search.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema": "hatchmatch-calibration-search/v1",
                "seed": 20260923,
                "recall_floor": 0.95,
                "n_trials": 1,
                "max_pixels_per_document": 64,
                "output_dir": "calibration-out",
                "oof_manifest": "oof.json",
                "fold_manifest": "folds.json",
                "training_manifest": "train.json",
                "data_root": ".",
                "parameters": {
                    "threshold": {"low": 0.2, "high": 0.8},
                    "min_component": {"low": 1, "high": 4},
                    "close_radius": {"low": 0, "high": 1},
                    "foreground_floor": {"low": 0.0, "high": 0.2},
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_load_out_of_fold_examples_excludes_generated_cad(tmp_path: Path) -> None:
    from hatchmatch.calibrate import load_out_of_fold_examples

    config_path = _write_kind_fixture(tmp_path, include_real=True)
    config = yaml.safe_load(config_path.read_text())

    channels, targets, knowns = load_out_of_fold_examples(config, tmp_path)

    assert [channel.example_id for channel in channels] == ["q-real"]
    assert [channel.kind for channel in channels] == ["real"]
    assert len(targets) == len(knowns) == 1

    cad_only = tmp_path / "cad-only"
    cad_config = yaml.safe_load(
        _write_kind_fixture(cad_only, include_real=False).read_text()
    )
    with pytest.raises(ValueError, match="real"):
        load_out_of_fold_examples(cad_config, cad_only)
    assert not (cad_only / "calibration-out" / "best.json").exists()


def test_calibrate_import_does_not_load_training_labels() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import hatchmatch.calibrate, sys; "
                "assert 'hatchmatch.data' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
