"""Provenance, ablation promotion, and the final score gate."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.run_experiments as experiments

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_RUN_FILES = {"config.json", "metrics.json", "provenance.json"}


def _score(
    document_id: str,
    iou: float,
    *,
    kind: str = "real",
    query_id: str | None = None,
) -> dict[str, object]:
    return {
        "document_id": document_id,
        "query_id": query_id or document_id,
        "iou": iou,
        "kind": kind,
    }


def _metrics_paths() -> set[Path]:
    runs = ROOT / "runs"
    if not runs.exists():
        return set()
    return set(runs.rglob("metrics.json"))


def test_run_requires_unedited_metrics_and_provenance(tmp_path: Path) -> None:
    missing = experiments.validate_run(tmp_path)
    assert set(missing) == REQUIRED_RUN_FILES


def test_validate_run_rejects_metrics_that_differ_from_provenance(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    config_bytes = b'{"schema":"hatchmatch-run-config/v1"}\n'
    metrics_bytes = b'{"summary":{"document_macro_iou":0.41}}\n'
    (run / "config.json").write_bytes(config_bytes)
    (run / "metrics.json").write_bytes(metrics_bytes)
    provenance = {
        "schema": "hatchmatch-run-provenance/v1",
        "artifacts": {
            "config.json": {
                "sha256": hashlib.sha256(config_bytes).hexdigest(),
                "bytes": len(config_bytes),
            },
            "metrics.json": {
                "sha256": hashlib.sha256(metrics_bytes).hexdigest(),
                "bytes": len(metrics_bytes),
            },
        },
    }
    (run / "provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )

    assert experiments.validate_run(run) == []

    (run / "metrics.json").write_bytes(metrics_bytes + b" ")
    problems = experiments.validate_run(run)
    assert "metrics.json" in problems
    assert "config.json" not in problems


def test_config_hash_is_sha256_of_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    payload = b"threshold: 0.5\n"
    path.write_bytes(payload)

    assert experiments.config_sha256(path) == hashlib.sha256(payload).hexdigest()


def test_seal_run_does_not_invent_metrics(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: demo\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="metrics.json"):
        experiments.seal_run(run, config_path, command="baseline")

    assert not (run / "metrics.json").exists()
    assert not (run / "config.json").exists()
    assert not (run / "provenance.json").exists()


def test_completed_run_refuses_overwrite_and_keeps_metric_bytes(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    config_path = tmp_path / "config.yaml"
    raw = b"name: demo\n"
    config_path.write_bytes(raw)
    metrics = b'{"summary":{"document_macro_iou":0.41}}\n'
    (run / "metrics.json").write_bytes(metrics)

    experiments.seal_run(run, config_path, command="baseline")

    stored = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert stored["sha256"] == hashlib.sha256(raw).hexdigest()
    assert experiments.validate_run(run) == []
    original = (run / "metrics.json").read_bytes()

    with pytest.raises(FileExistsError, match="overwrite"):
        experiments.seal_run(run, config_path, command="baseline")

    assert (run / "metrics.json").read_bytes() == original


def test_baseline_does_not_replace_a_completed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    config_path = tmp_path / "baseline.yaml"
    config_path.write_bytes(b"placeholder: true\n")
    (run / "metrics.json").write_bytes(
        b'{"summary":{"document_macro_iou":0.2}}\n'
    )
    experiments.seal_run(run, config_path, command="baseline")
    original = (run / "metrics.json").read_bytes()

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("baseline continued past overwrite refusal")

    monkeypatch.setattr(
        experiments, "_load_config", lambda path: {"run_dir": run}
    )
    monkeypatch.setattr(experiments, "_verify_challenge_revision", fail)
    monkeypatch.setattr(experiments, "_predict", fail)
    monkeypatch.setattr(experiments.subprocess, "run", fail)

    with pytest.raises(FileExistsError, match="overwrite"):
        experiments.run_baseline(config_path)

    assert (run / "metrics.json").read_bytes() == original


def test_promotion_uses_real_document_macro_iou() -> None:
    baseline = [
        _score("a", 0.40, query_id="a1"),
        _score("a", 0.60, query_id="a2"),
        _score("b", 0.20),
        _score("cad", 0.99, kind="generated_cad"),
    ]
    candidate = [
        _score("a", 0.50, query_id="a1"),
        _score("a", 0.70, query_id="a2"),
        _score("b", 0.30),
        _score("cad", 0.0, kind="generated_cad"),
    ]

    decision = experiments.promote_ablation(baseline, candidate)

    assert decision["promoted"] is True
    assert decision["official_public_validation"] is False
    assert decision["baseline_document_macro_iou"] == pytest.approx(0.35)
    assert decision["candidate_document_macro_iou"] == pytest.approx(0.45)
    assert decision["baseline_document_iou"] == {
        "a": pytest.approx(0.5),
        "b": pytest.approx(0.2),
    }
    assert decision["collapsed_documents"] == []
    assert decision["sharp_drop_documents"] == []


def test_promotion_rejects_flat_or_worse_macro() -> None:
    baseline = [_score("a", 0.40), _score("b", 0.50)]
    same = [_score("a", 0.40), _score("b", 0.50)]
    worse = [_score("a", 0.45), _score("b", 0.40)]

    assert experiments.promote_ablation(baseline, same)["promoted"] is False
    assert experiments.promote_ablation(baseline, worse)["promoted"] is False


def test_promotion_rejects_collapse_to_zero_and_sharp_drops() -> None:
    baseline = [_score("a", 0.40), _score("b", 0.50)]
    collapsed = [_score("a", 0.95), _score("b", 0.0)]
    sharp = [_score("a", 0.80), _score("b", 0.44)]
    mild = [_score("a", 0.50), _score("b", 0.45)]
    still_zero = [_score("a", 0.0), _score("b", 0.60)]
    baseline_zero = [_score("a", 0.0), _score("b", 0.50)]

    collapse = experiments.promote_ablation(baseline, collapsed)
    assert collapse["promoted"] is False
    assert collapse["collapsed_documents"] == ["b"]
    assert collapse["candidate_document_macro_iou"] == pytest.approx(0.475)

    dropped = experiments.promote_ablation(baseline, sharp)
    assert dropped["promoted"] is False
    assert dropped["sharp_drop_documents"] == ["b"]

    assert experiments.promote_ablation(baseline, mild)["promoted"] is True
    kept = experiments.promote_ablation(baseline_zero, still_zero)
    assert kept["promoted"] is True
    assert kept["collapsed_documents"] == []


def test_promotion_rejects_incomplete_or_non_real_scores() -> None:
    baseline = [_score("a", 0.40), _score("b", 0.50)]

    with pytest.raises(ValueError, match="document"):
        experiments.promote_ablation(baseline, [_score("a", 0.90)])
    with pytest.raises(ValueError, match="real"):
        experiments.promote_ablation(
            [_score("cad", 0.2, kind="generated_cad")],
            [_score("cad", 0.9, kind="generated_cad")],
        )


def test_ablation_factors_cover_declared_changes() -> None:
    factors = experiments.ablation_factors()
    assert [item["name"] for item in factors[:7]] == [
        "texture_channels",
        "neural_model",
        "calibration",
        "morphology",
        "backbone_size",
        "fold_count",
        "seeds",
    ]
    assert [item["name"] for item in factors if item["factor"] == "tta"] == [
        "tta:identity",
        "tta:horizontal",
        "tta:vertical",
        "tta:rot90",
    ]


def test_selection_promotes_only_caller_supplied_improvements() -> None:
    scores = {
        "schema": "hatchmatch-ablation-scores/v1",
        "comparisons": [
            {
                "factor": "texture_channels",
                "baseline": [_score("a", 0.40), _score("b", 0.50)],
                "candidate": [_score("a", 0.50), _score("b", 0.50)],
            },
            {
                "factor": "morphology",
                "baseline": [_score("a", 0.40), _score("b", 0.50)],
                "candidate": [_score("a", 0.90), _score("b", 0.0)],
            },
        ],
    }

    decision = experiments.select_ablations(scores)

    assert decision["promoted"] == ["texture_channels"]
    assert decision["rejected"] == ["morphology"]
    assert decision["official_public_validation"] is False
    with pytest.raises(ValueError, match="factor"):
        experiments.select_ablations(
            {
                "schema": "hatchmatch-ablation-scores/v1",
                "comparisons": [
                    {
                        "factor": "not-a-factor",
                        "baseline": [_score("a", 0.2)],
                        "candidate": [_score("a", 0.3)],
                    }
                ],
            }
        )


def test_runs_gitignore_ignores_artifacts() -> None:
    text = (ROOT / "runs" / ".gitignore").read_text(encoding="utf-8")
    assert "*" in text.splitlines()
    assert "!.gitignore" in text.splitlines()


@pytest.mark.parametrize(
    "command",
    ["baseline", "train-folds", "predict-oof", "calibrate", "ablate", "final"],
)
def test_subcommand_accepts_config(command: str) -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_experiments.py", command, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--config" in completed.stdout


def test_train_folds_and_predict_oof_do_not_fabricate_metrics(
    tmp_path: Path,
) -> None:
    train_config = tmp_path / "train.yaml"
    train_config.write_text(
        "folds: [0]\nseeds: [1]\noutput_dir: out\n"
        "data:\n  manifest: missing-train.json\n"
        "  fold_manifest: missing-folds.json\n",
        encoding="utf-8",
    )
    predict_config = tmp_path / "predict.yaml"
    predict_config.write_text(
        "checkpoints:\n  - path: missing/best.pt\n"
        "output_dir: oof-out\n",
        encoding="utf-8",
    )

    for command, config in (
        ("train-folds", train_config),
        ("predict-oof", predict_config),
    ):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_experiments.py"),
                command,
                "--config",
                str(config),
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode != 0, completed.stderr
        assert "missing" in completed.stderr.lower()
        assert "fabricat" in completed.stderr.lower()
        assert not (tmp_path / "out" / "metrics.json").exists()
        assert not (tmp_path / "oof-out" / "metrics.json").exists()
        assert list(tmp_path.rglob("metrics.json")) == []


def test_calibrate_subcommand_refuses_missing_oof_maps() -> None:
    before = _metrics_paths()
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiments.py",
            "calibrate",
            "--config",
            "configs/search.yaml",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "out-of-fold" in completed.stderr.lower()
    assert not (ROOT / "runs" / "calibration" / "best.json").exists()
    assert not (ROOT / "runs" / "calibration" / "metrics.json").exists()
    assert _metrics_paths() == before


def test_ablate_and_final_fail_closed_without_checkpoints_or_oof_maps() -> None:
    before = _metrics_paths()
    for command in ("ablate", "final"):
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_experiments.py",
                command,
                "--config",
                "configs/final.yaml",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode != 0, completed.stdout
        stderr = completed.stderr.lower()
        assert "missing" in stderr
        assert "checkpoint" in stderr
        assert "out-of-fold" in stderr
        assert "fabricat" in stderr
        assert completed.stderr.count("best.pt") >= 15
        assert "document_macro_iou" not in completed.stdout
        assert "document_macro_iou" not in completed.stderr

    assert _metrics_paths() == before
    assert not (ROOT / "runs" / "ablation" / "metrics.json").exists()
    assert not (ROOT / "runs" / "final" / "metrics.json").exists()


def test_ablate_does_not_launch_native_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("native validation must not run")

    monkeypatch.setattr(experiments.subprocess, "run", fail)

    code = experiments.run_ablate(ROOT / "configs" / "final.yaml")

    assert code == 2
    assert not (ROOT / "runs" / "ablation" / "metrics.json").exists()
    assert not (ROOT / "runs" / "final" / "metrics.json").exists()
