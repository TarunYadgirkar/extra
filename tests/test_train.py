import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import Dataset, DistributedSampler

from hatchmatch.checkpoints import load_verified_checkpoint
from hatchmatch.train import (
    EarlyStopping,
    ExponentialMovingAverage,
    _advance_epoch,
    document_macro_iou,
    optimizer_update,
    restore_rng_state,
    snapshot_rng_state,
)


class _EpochDataset(Dataset[int]):
    def __init__(self) -> None:
        self.epoch = -1

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> int:
        return index

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def test_document_macro_iou_averages_queries_then_documents_on_known_pixels() -> None:
    # doc-a query IoUs are 1 and 0; doc-b has 1. The document macro is
    # mean(mean(1, 0), mean(1)) = 0.75, not query-macro 2/3.
    logits = torch.tensor(
        [
            [[[10.0, -10.0, 10.0]]],
            [[[-10.0, 10.0, -10.0]]],
            [[[10.0, 10.0, -10.0]]],
            [[[10.0, 10.0, 10.0]]],
        ]
    )
    target = torch.tensor(
        [
            [[[1.0, 0.0, 0.0]]],
            [[[1.0, 0.0, 0.0]]],
            [[[1.0, 1.0, 0.0]]],
            [[[0.0, 0.0, 0.0]]],
        ]
    )
    known = torch.tensor(
        [
            [[[1.0, 1.0, 0.0]]],
            [[[1.0, 1.0, 0.0]]],
            [[[1.0, 1.0, 1.0]]],
            [[[1.0, 1.0, 1.0]]],
        ]
    )

    metric = document_macro_iou(
        logits,
        target,
        known,
        ["doc-a", "doc-a", "doc-b", "doc-c"],
    )

    assert metric == pytest.approx(0.75)


def test_document_macro_iou_ignores_unknown_predictions() -> None:
    target = torch.tensor([[[[1.0, 0.0]]]])
    known = torch.tensor([[[[1.0, 0.0]]]])

    low_unknown = document_macro_iou(
        torch.tensor([[[[10.0, -10.0]]]]),
        target,
        known,
        ["doc"],
    )
    high_unknown = document_macro_iou(
        torch.tensor([[[[10.0, 10.0]]]]),
        target,
        known,
        ["doc"],
    )

    assert low_unknown == high_unknown == 1.0


def test_ddp_sampler_and_dataset_receive_same_epoch() -> None:
    dataset = _EpochDataset()
    sampler = DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=True)
    before = list(iter(sampler))

    _advance_epoch(dataset, sampler, 3)
    after = list(iter(sampler))

    assert dataset.epoch == 3
    assert sampler.epoch == 3
    assert before != after


def test_ema_updates_parameters_and_preserves_integer_buffers() -> None:
    model = nn.BatchNorm1d(2)
    ema = ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3.0)
        model.num_batches_tracked.fill_(7)

    ema.update(model)

    torch.testing.assert_close(ema.state_dict()["weight"], torch.full((2,), 2.0))
    assert ema.state_dict()["num_batches_tracked"].item() == 7


def test_gradient_accumulation_and_clipping_match_single_averaged_update() -> None:
    accumulated = nn.Linear(1, 1, bias=False)
    reference = copy.deepcopy(accumulated)
    first = torch.tensor([[1.0]])
    second = torch.tensor([[3.0]])
    target = torch.tensor([[0.0]])
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)

    for sample in (first, second):
        (nn.functional.mse_loss(accumulated(sample), target) / 2).backward()
    result = optimizer_update(
        accumulated,
        accumulated_optimizer,
        scaler=None,
        max_grad_norm=0.25,
    )
    (
        (
            nn.functional.mse_loss(reference(first), target)
            + nn.functional.mse_loss(reference(second), target)
        )
        / 2
    ).backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.25)
    reference_optimizer.step()

    assert result.stepped
    assert result.grad_norm > 0.25
    torch.testing.assert_close(accumulated.weight, reference.weight)


def test_early_stopping_uses_document_macro_iou() -> None:
    stopping = EarlyStopping(patience=2, min_delta=0.01)

    assert stopping.update(0.50)
    assert not stopping.update(0.505)
    assert not stopping.should_stop
    assert not stopping.update(0.49)
    assert stopping.should_stop
    assert stopping.best == 0.50


def test_rng_state_round_trip_supports_deterministic_resume() -> None:
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    state = snapshot_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(2))

    restore_rng_state(state)
    resumed = (random.random(), np.random.random(), torch.rand(2))

    assert resumed[:2] == expected[:2]
    torch.testing.assert_close(resumed[2], expected[2], rtol=0, atol=0)


def test_resume_checkpoint_contains_complete_training_and_provenance_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    payload = {
        "state_dict": {"weight": torch.tensor([1.0])},
        "ema_state_dict": {"weight": torch.tensor([0.9])},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "scheduler_state_dict": {"last_epoch": 1},
        "scaler_state_dict": {},
        "rng_state": snapshot_rng_state(),
        "training_state": {"epoch": 1, "batch_in_epoch": 2, "global_step": 3},
        "architecture": {"backbone": "nvidia/mit-b2"},
        "dependency_versions": {"torch": torch.__version__},
        "fold": {
            "index": 0,
            "train_ids": ["q1"],
            "valid_ids": ["q2"],
        },
        "dataset_checksum": hashlib.sha256(b"dataset").hexdigest(),
        "git_sha": "a" * 40,
    }
    from hatchmatch.checkpoints import save_checkpoint

    digest = save_checkpoint(path, payload)
    loaded = load_verified_checkpoint(path, digest)

    assert set(payload) <= set(loaded)
    assert loaded["training_state"]["global_step"] == 3


@pytest.mark.parametrize("name", ["train-b2.yaml", "train-b4.yaml"])
def test_training_configs_define_five_folds_and_three_seeds(name: str) -> None:
    config = yaml.safe_load((Path("configs") / name).read_text())

    assert config["folds"] == [0, 1, 2, 3, 4]
    assert len(config["seeds"]) == 3
    assert len(set(config["seeds"])) == 3


def test_b2_config_pins_model_artifact_revision() -> None:
    config = yaml.safe_load(Path("configs/train-b2.yaml").read_text())

    assert config["model"]["backbone"] == "nvidia/mit-b2"
    assert (
        config["model"]["revision"]
        == "3bb39e8739149c3777d0325349b2a6c32c6413db"
    )
