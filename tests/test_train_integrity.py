import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from hatchmatch.checkpoints import load_verified_checkpoint, save_checkpoint
from hatchmatch.data import TrainingExample, TrainingLabels
from hatchmatch.train import (
    ExponentialMovingAverage,
    _ValidationDataset,
    _dataset_provenance,
    _dependency_versions,
    _gather_query_records,
    _gather_rank_states,
    _load_fold,
    _native_query_iou_records,
    _resume_invariants,
    _select_run_seeds,
    _validate_resume_checkpoint,
    _validation_collate,
    optimizer_update,
    restore_rng_state,
    snapshot_rng_state,
)


def _training_example(index: int, document: str) -> TrainingExample:
    labels = TrainingLabels(
        masks={name: None for name in ("positive", "negative", "known", "blank")},
        boxes={name: () for name in ("positive", "negative", "known", "blank")},
    )
    return TrainingExample(
        id=f"q{index}",
        document_id=document,
        kind="real",
        image=Path(f"image-{index}.png"),
        width=8,
        height=8,
        query_box=type("BoxValue", (), {"x0": 0, "y0": 0, "x1": 1, "y1": 1})(),
        context_boxes=(),
        labels=labels,
    )


def _fold_payload(source: Path) -> tuple[list[TrainingExample], dict[str, object]]:
    examples = [
        _training_example(index, f"doc-{index // 2}") for index in range(8)
    ]
    partitions = ((range(4, 8), range(0, 4)), (range(0, 4), range(4, 8)))
    folds = []
    for fold, (train, valid) in enumerate(partitions):
        train_indices = list(train)
        valid_indices = list(valid)
        folds.append(
            {
                "fold": fold,
                "train_indices": train_indices,
                "valid_indices": valid_indices,
                "train_ids": [examples[index].id for index in train_indices],
                "valid_ids": [examples[index].id for index in valid_indices],
                "train_documents": sorted(
                    {examples[index].document_id for index in train_indices}
                ),
                "valid_documents": sorted(
                    {examples[index].document_id for index in valid_indices}
                ),
                "valid_kind_counts": {"generated_cad": 0, "real": 4},
            }
        )
    return examples, {
        "schema": "hatchmatch-group-folds/v1",
        "source_manifest": str(source.resolve()),
        "seed": 71,
        "fold_count": 2,
        "example_count": len(examples),
        "document_count": 4,
        "folds": folds,
    }


def _fold_config(tmp_path: Path, source: Path, fold_path: Path) -> dict[str, object]:
    return {
        "folds": [0, 1],
        "data": {
            "manifest": str(source),
            "fold_manifest": str(fold_path),
            "fold_seed": 71,
        },
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.update(source_manifest="/wrong/train.json"),
            "source",
        ),
        (lambda payload: payload.update(seed=72), "seed"),
        (
            lambda payload: payload["folds"][0]["train_indices"].append(99),
            "bounds",
        ),
        (
            lambda payload: payload["folds"][0].update(
                train_ids=["wrong", *payload["folds"][0]["train_ids"][1:]]
            ),
            "IDs",
        ),
        (
            lambda payload: payload["folds"][0].update(train_documents=["wrong"]),
            "documents",
        ),
        (
            lambda payload: (
                payload["folds"][0].update(
                    train_indices=[1, 4, 5, 6, 7],
                    valid_indices=[0, 2, 3],
                    train_ids=["q1", "q4", "q5", "q6", "q7"],
                    valid_ids=["q0", "q2", "q3"],
                    train_documents=["doc-0", "doc-2", "doc-3"],
                    valid_documents=["doc-0", "doc-1"],
                )
            ),
            "document",
        ),
        (
            lambda payload: payload["folds"][1].update(
                valid_indices=payload["folds"][0]["valid_indices"],
                valid_ids=payload["folds"][0]["valid_ids"],
                valid_documents=payload["folds"][0]["valid_documents"],
                train_indices=payload["folds"][0]["train_indices"],
                train_ids=payload["folds"][0]["train_ids"],
                train_documents=payload["folds"][0]["train_documents"],
            ),
            "coverage",
        ),
    ],
)
def test_loaded_fold_manifest_rejects_malformed_or_leaky_content(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    source = tmp_path / "train.json"
    source.write_text("{}", encoding="utf-8")
    fold_path = tmp_path / "folds.json"
    examples, payload = _fold_payload(source)
    mutation(payload)
    fold_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_fold(_fold_config(tmp_path, source, fold_path), examples, 0)


def test_loaded_fold_manifest_accepts_and_returns_exact_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "train.json"
    source.write_text("{}", encoding="utf-8")
    fold_path = tmp_path / "folds.json"
    examples, payload = _fold_payload(source)
    fold_path.write_text(json.dumps(payload), encoding="utf-8")

    record = _load_fold(_fold_config(tmp_path, source, fold_path), examples, 0)

    assert record == payload["folds"][0]


def test_validation_keeps_native_thin_positive_eligible(tmp_path: Path) -> None:
    drawing = np.full((8, 8), 255, dtype=np.uint8)
    positive = np.zeros((8, 8), dtype=np.uint8)
    positive[4, 4] = 255
    known = np.full((8, 8), 255, dtype=np.uint8)
    Image.fromarray(drawing).save(tmp_path / "drawing.png")
    Image.fromarray(positive).save(tmp_path / "positive.png")
    Image.fromarray(known).save(tmp_path / "known.png")
    from hatchmatch.contracts import Box

    labels = TrainingLabels(
        masks={
            "positive": tmp_path / "positive.png",
            "negative": None,
            "known": tmp_path / "known.png",
            "blank": None,
        },
        boxes={name: () for name in ("positive", "negative", "known", "blank")},
        explicit_known=True,
    )
    example = TrainingExample(
        id="thin",
        document_id="doc-thin",
        kind="real",
        image=tmp_path / "drawing.png",
        width=8,
        height=8,
        query_box=Box(0, 0, 1, 1),
        context_boxes=(),
        labels=labels,
    )
    dataset = _ValidationDataset([example], size=2, query_size=2)

    item = dataset[0]
    batch = _validation_collate([item])
    records = _native_query_iou_records(
        torch.full((1, 1, 2, 2), -10.0),
        batch["native_target"],
        batch["native_known"],
        batch["content_box"],
        batch["document_id"],
        threshold=0.5,
    )

    assert item["target"].shape == (1, 8, 8)
    assert item["target"].sum().item() == 1
    assert records == [("doc-thin", 0.0)]


class _OverflowScaler:
    def __init__(self) -> None:
        self.scale = 8.0

    def unscale_(self, optimizer) -> None:
        pass

    def step(self, optimizer) -> None:
        pass

    def update(self) -> None:
        self.scale = 4.0

    def get_scale(self) -> float:
        return self.scale


def test_amp_overflow_reports_skipped_step_for_scheduler_ema_and_counter() -> None:
    model = nn.Linear(1, 1, bias=False)
    original = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=4)
    ema = ExponentialMovingAverage(model, decay=0.5)
    model(torch.ones(1, 1)).sum().backward()

    result = optimizer_update(
        model,
        optimizer,
        scaler=_OverflowScaler(),
        max_grad_norm=1.0,
    )
    global_step = 0
    if result.stepped:
        scheduler.step()
        ema.update(model)
        global_step += 1

    assert not result.stepped
    assert scheduler.last_epoch == 0
    assert global_step == 0
    torch.testing.assert_close(model.weight, original)
    torch.testing.assert_close(ema.state_dict()["weight"], original)


def _tiny_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    ema: ExponentialMovingAverage,
    value: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    output = model(value)
    output.square().mean().backward()
    result = optimizer_update(model, optimizer, scaler=None, max_grad_norm=1.0)
    assert result.stepped
    scheduler.step()
    ema.update(model)


def _tiny_components() -> tuple[
    nn.Module,
    torch.optim.Optimizer,
    CosineAnnealingLR,
    ExponentialMovingAverage,
]:
    model = nn.Sequential(nn.Linear(2, 3), nn.Dropout(0.4), nn.Linear(3, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=4)
    ema = ExponentialMovingAverage(model, decay=0.8)
    return model, optimizer, scheduler, ema


def test_mid_epoch_resume_matches_uninterrupted_weights_ema_and_scheduler(
    tmp_path: Path,
) -> None:
    values = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)
    full = _tiny_components()
    full_generator = torch.Generator().manual_seed(900)
    full_loader = DataLoader(
        TensorDataset(values),
        batch_size=1,
        shuffle=False,
        generator=full_generator,
    )
    for (value,) in full_loader:
        _tiny_step(*full, value)

    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)
    interrupted = _tiny_components()
    interrupted_generator = torch.Generator().manual_seed(900)
    interrupted_loader = DataLoader(
        TensorDataset(values),
        batch_size=1,
        shuffle=False,
        generator=interrupted_generator,
    )
    for index, (value,) in enumerate(interrupted_loader):
        _tiny_step(*interrupted, value)
        if index == 1:
            break
    path = tmp_path / "resume.pt"
    save_checkpoint(
        path,
        {
            "model": interrupted[0].state_dict(),
            "optimizer": interrupted[1].state_dict(),
            "scheduler": interrupted[2].state_dict(),
            "ema": interrupted[3].state_dict(),
            "rank_states": [
                {
                    "rng_state": snapshot_rng_state(),
                    "data_loader_generator_state": interrupted_generator.get_state(),
                }
            ],
            "global_step": 2,
        },
    )

    torch.manual_seed(999)
    resumed = _tiny_components()
    resumed_generator = torch.Generator()
    checkpoint = load_verified_checkpoint(path)
    resumed[0].load_state_dict(checkpoint["model"])
    resumed[1].load_state_dict(checkpoint["optimizer"])
    resumed[2].load_state_dict(checkpoint["scheduler"])
    resumed[3].load_state_dict(checkpoint["ema"])
    rank_state = checkpoint["rank_states"][0]
    resumed_generator.set_state(rank_state["data_loader_generator_state"])
    restore_rng_state(rank_state["rng_state"])
    resumed_loader = DataLoader(
        TensorDataset(values),
        batch_size=1,
        shuffle=False,
        generator=resumed_generator,
    )
    global_step = checkpoint["global_step"]
    for index, (value,) in enumerate(resumed_loader):
        if index < 2:
            continue
        _tiny_step(*resumed, value)
        global_step += 1

    for expected, actual in zip(
        full[0].state_dict().values(),
        resumed[0].state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for key, expected in full[3].state_dict().items():
        torch.testing.assert_close(
            resumed[3].state_dict()[key],
            expected,
            rtol=0,
            atol=0,
        )
    assert resumed[2].state_dict() == full[2].state_dict()
    assert global_step == 4


def test_resume_validation_rejects_every_training_identity_mismatch() -> None:
    config = {
        "seed": 11,
        "folds": [0],
        "model": {"backbone": "tiny"},
        "data": {"tile_size": 32, "workers": 0},
        "training": {
            "gradient_accumulation": 2,
            "learning_rate": 0.1,
            "max_steps": None,
        },
    }
    fold = {
        "fold": 0,
        "train_ids": ["a"],
        "valid_ids": ["b"],
        "train_documents": ["da"],
        "valid_documents": ["db"],
    }
    provenance = {
        "source_manifest": {"sha256": "1" * 64},
        "checksum_manifest": {"sha256": "2" * 64},
        "fold_manifest": {"sha256": "3" * 64},
    }
    checkpoint = {
        "seed": 11,
        "world_size": 2,
        "gradient_accumulation": 2,
        "fold": copy.deepcopy(fold),
        "provenance": copy.deepcopy(provenance),
        "resume_invariants": _resume_invariants(config),
    }
    mutations = [
        ("seed", lambda value: value.update(seed=12)),
        ("world size", lambda value: value.update(world_size=1)),
        ("accumulation", lambda value: value.update(gradient_accumulation=4)),
        ("fold", lambda value: value["fold"]["valid_ids"].append("c")),
        (
            "provenance",
            lambda value: value["provenance"]["fold_manifest"].update(
                sha256="4" * 64
            ),
        ),
        (
            "config",
            lambda value: value["resume_invariants"]["training"].update(
                learning_rate=0.2
            ),
        ),
    ]
    for message, mutate in mutations:
        changed = copy.deepcopy(checkpoint)
        mutate(changed)
        with pytest.raises(ValueError, match=message):
            _validate_resume_checkpoint(
                changed,
                config=config,
                fold_record=fold,
                provenance=provenance,
                world_size=2,
            )


def test_resume_requires_selecting_one_configured_seed() -> None:
    with pytest.raises(ValueError, match="single seed"):
        _select_run_seeds([1, 2, 3], selected_seed=None, resume=Path("last.pt"))

    assert _select_run_seeds(
        [1, 2, 3],
        selected_seed=2,
        resume=Path("last.pt"),
    ) == [2]


def test_provenance_validates_source_checksum_and_hashes_fold_bytes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "train.json"
    source.write_bytes(b'{"schema":"test"}\n')
    fold = tmp_path / "folds.json"
    fold.write_bytes(b'{"folds":[]}\n')
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    checksum = tmp_path / "checksums.json"
    checksum.write_text(
        json.dumps(
            {
                "train.json": {
                    "bytes": source.stat().st_size,
                    "sha256": source_digest,
                }
            }
        ),
        encoding="utf-8",
    )
    config = {
        "data": {
            "manifest": str(source),
            "checksum_file": str(checksum),
            "fold_manifest": str(fold),
        }
    }

    provenance = _dataset_provenance(config)

    assert provenance["source_manifest"]["sha256"] == source_digest
    assert provenance["checksum_manifest"]["sha256"] == hashlib.sha256(
        checksum.read_bytes()
    ).hexdigest()
    assert provenance["fold_manifest"]["sha256"] == hashlib.sha256(
        fold.read_bytes()
    ).hexdigest()

    source.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        _dataset_provenance(config)


def test_dependency_provenance_covers_training_stack() -> None:
    assert {
        "albumentations",
        "numpy",
        "opencv-python-headless",
        "pillow",
        "pyyaml",
        "scikit-image",
        "scikit-learn",
        "torch",
        "transformers",
    } <= _dependency_versions().keys()


def _two_rank_worker(rank: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        torch.manual_seed(100 + rank)
        model = nn.Linear(2, 1)
        ddp = DistributedDataParallel(model)
        ema = ExponentialMovingAverage(ddp.module, decay=0.5)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.1)
        optimizer.zero_grad(set_to_none=True)
        ddp(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        ema.update(ddp.module)

        gathered_ema: list[dict[str, torch.Tensor] | None] = [None, None]
        dist.all_gather_object(gathered_ema, ema.state_dict())
        for key in gathered_ema[0]:
            torch.testing.assert_close(
                gathered_ema[0][key],
                gathered_ema[1][key],
                rtol=0,
                atol=0,
            )

        generator = torch.Generator().manual_seed(700 + rank)
        rank_states = _gather_rank_states(generator)
        assert len(rank_states) == 2
        assert not torch.equal(
            rank_states[0]["rng_state"]["torch"],
            rank_states[1]["rng_state"]["torch"],
        )
        records = _gather_query_records(
            [("doc-a", 1.0)] if rank == 0 else [("doc-b", 0.0)]
        )
        assert sorted(records) == [("doc-a", 1.0), ("doc-b", 0.0)]
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_ema_rank_state_and_metric_aggregation(
    tmp_path: Path,
) -> None:
    init_file = tmp_path / "gloo-init"
    mp.spawn(
        _two_rank_worker,
        args=(str(init_file),),
        nprocs=2,
        join=True,
    )
