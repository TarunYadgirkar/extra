"""Deterministic distributed fold training for query-conditioned segmentation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from hatchmatch.checkpoints import load_verified_checkpoint, save_checkpoint
from hatchmatch.data import (
    FOLD_SCHEMA,
    HatchTileDataset,
    TrainingExample,
    load_training_examples,
    make_group_folds,
)
from hatchmatch.features import compact_texture_channels
from hatchmatch.losses import masked_segmentation_loss
from hatchmatch.model import QuerySegFormer


class ExponentialMovingAverage:
    """Track a detached exponential moving average of model state."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self._state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        self._backup: dict[str, Tensor] | None = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        if current.keys() != self._state.keys():
            raise ValueError("EMA and model states have different keys")
        for key, value in current.items():
            shadow = self._state[key]
            if torch.is_floating_point(shadow):
                shadow.lerp_(value.detach(), 1.0 - self.decay)
            else:
                shadow.copy_(value.detach())

    def state_dict(self) -> dict[str, Tensor]:
        return {key: value.detach().clone() for key, value in self._state.items()}

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        if state.keys() != self._state.keys():
            raise ValueError("loaded EMA state has different keys")
        for key, value in state.items():
            self._state[key].copy_(value)

    @torch.no_grad()
    def apply(self, model: nn.Module) -> None:
        if self._backup is not None:
            raise RuntimeError("EMA weights are already applied")
        self._backup = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        model.load_state_dict(self._state, strict=True)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if self._backup is None:
            raise RuntimeError("EMA weights are not applied")
        model.load_state_dict(self._backup, strict=True)
        self._backup = None


@dataclass
class EarlyStopping:
    """Maximize validation document-macro IoU with finite patience."""

    patience: int
    min_delta: float = 0.0
    best: float = -math.inf
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("early stopping patience must be positive")
        if self.min_delta < 0:
            raise ValueError("early stopping min_delta must be non-negative")

    def update(self, metric: float) -> bool:
        improved = metric > self.best + self.min_delta
        if improved:
            self.best = float(metric)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    def state_dict(self) -> dict[str, float | int]:
        return {"best": self.best, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.best = float(state["best"])
        self.bad_epochs = int(state["bad_epochs"])


def snapshot_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU, and all available CUDA RNG streams."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state returned by :func:`snapshot_rng_state`."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def _gather_rank_states(
    data_loader_generator: torch.Generator,
) -> list[dict[str, Any]]:
    """Collect each rank's model RNG and isolated DataLoader RNG state."""

    local_state = {
        "rng_state": snapshot_rng_state(),
        "data_loader_generator_state": data_loader_generator.get_state(),
    }
    if not dist.is_initialized():
        return [local_state]
    gathered: list[dict[str, Any] | None] = [
        None for _ in range(dist.get_world_size())
    ]
    dist.all_gather_object(gathered, local_state)
    if any(state is None for state in gathered):
        raise RuntimeError("failed to gather every rank's resume state")
    return [state for state in gathered if state is not None]


def _seed_everything(seed: int, rank: int = 0) -> None:
    process_seed = int(seed) + int(rank)
    random.seed(process_seed)
    np.random.seed(process_seed % 2**32)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class OptimizerStepResult:
    """Outcome of one attempted optimizer update."""

    grad_norm: float
    stepped: bool


def optimizer_update(
    model: nn.Module,
    optimizer: Optimizer,
    *,
    scaler: torch.amp.GradScaler | None,
    max_grad_norm: float,
) -> OptimizerStepResult:
    """Unscale, clip, update, and clear one accumulated gradient."""

    if scaler is not None:
        scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if scaler is None:
        optimizer.step()
        stepped = True
    else:
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        stepped = scaler.get_scale() >= scale_before
    optimizer.zero_grad(set_to_none=True)
    return OptimizerStepResult(grad_norm=float(norm), stepped=stepped)


def _advance_epoch(
    dataset: Dataset[Any],
    sampler: Sampler[Any],
    epoch: int,
) -> None:
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)  # type: ignore[attr-defined]
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)  # type: ignore[attr-defined]


def _query_iou_records(
    logits: Tensor,
    target: Tensor,
    known: Tensor,
    document_ids: Sequence[str],
    threshold: float,
) -> list[tuple[str, float]]:
    if logits.shape != target.shape or logits.shape != known.shape:
        raise ValueError("logits, target, and known must have identical shapes")
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("metric tensors must have shape [B, 1, H, W]")
    if len(document_ids) != logits.shape[0]:
        raise ValueError("one document ID is required per query")
    prediction = torch.sigmoid(logits.detach()) >= threshold
    target_mask = target.detach() == 1
    known_mask = known.detach() == 1
    records: list[tuple[str, float]] = []
    for index, document_id in enumerate(document_ids):
        eligible_target = target_mask[index] & known_mask[index]
        if not bool(eligible_target.any()):
            continue
        eligible_prediction = prediction[index] & known_mask[index]
        intersection = int((eligible_prediction & eligible_target).sum().item())
        union = int((eligible_prediction | eligible_target).sum().item())
        records.append((str(document_id), intersection / union))
    return records


def _macro_from_records(records: Sequence[tuple[str, float]]) -> float | None:
    documents: defaultdict[str, list[float]] = defaultdict(list)
    for document_id, iou in records:
        documents[document_id].append(float(iou))
    if not documents:
        return None
    document_scores = [
        math.fsum(values) / len(values)
        for _, values in sorted(documents.items())
    ]
    return math.fsum(document_scores) / len(document_scores)


def document_macro_iou(
    logits: Tensor,
    target: Tensor,
    known: Tensor,
    document_ids: Sequence[str],
    *,
    threshold: float = 0.5,
) -> float | None:
    """Average eligible known-pixel query IoUs, first within each document."""

    return _macro_from_records(
        _query_iou_records(logits, target, known, document_ids, threshold)
    )


class _DistributedEvaluationSampler(Sampler[int]):
    """Shard validation indices without DistributedSampler padding duplicates."""

    def __init__(self, size: int, rank: int, world_size: int) -> None:
        self.size = size
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self) -> int:
        return len(range(self.rank, self.size, self.world_size))


def _letterbox(
    array: np.ndarray,
    size: int,
    *,
    interpolation: int,
    fill: int | bool,
) -> np.ndarray:
    height, width = array.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, min(size, round(width * scale)))
    resized_height = max(1, min(size, round(height * scale)))
    resized = cv2.resize(
        array.astype(np.uint8),
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    output = np.full((size, size), fill, dtype=resized.dtype)
    x0 = (size - resized_width) // 2
    y0 = (size - resized_height) // 2
    output[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return output


def _letterbox_content_box(
    height: int,
    width: int,
    size: int,
) -> tuple[int, int, int, int]:
    scale = min(size / width, size / height)
    resized_width = max(1, min(size, round(width * scale)))
    resized_height = max(1, min(size, round(height * scale)))
    x0 = (size - resized_width) // 2
    y0 = (size - resized_height) // 2
    return y0, y0 + resized_height, x0, x0 + resized_width


class _ValidationDataset(Dataset[dict[str, Tensor | str]]):
    """Return each held-out query exactly once with its known domain."""

    def __init__(
        self,
        examples: Sequence[TrainingExample],
        *,
        size: int,
        query_size: int,
    ) -> None:
        self.examples = tuple(examples)
        self.size = size
        self.query_size = query_size
        self._source = HatchTileDataset(
            examples,
            tile_size=size,
            query_size=query_size,
            samples_per_epoch=1,
            seed=0,
            augment=False,
            cache_size=0,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        arrays = self._source._load_arrays(index)
        image = _letterbox(
            arrays.image,
            self.size,
            interpolation=cv2.INTER_AREA,
            fill=255,
        )
        query = _letterbox(
            arrays.query,
            self.query_size,
            interpolation=cv2.INTER_AREA,
            fill=255,
        )
        return {
            "image": torch.from_numpy(
                np.repeat(image[None], 3, axis=0).astype(np.float32) / 255.0
            ),
            "query": torch.from_numpy(
                np.repeat(query[None], 3, axis=0).astype(np.float32) / 255.0
            ),
            "texture": torch.from_numpy(
                np.moveaxis(compact_texture_channels(image), -1, 0).copy()
            ),
            "target": torch.from_numpy(arrays.target[None].astype(np.float32)),
            "known": torch.from_numpy(arrays.known[None].astype(np.float32)),
            "content_box": _letterbox_content_box(
                arrays.image.shape[0],
                arrays.image.shape[1],
                self.size,
            ),
            "document_id": self.examples[index].document_id,
        }


def _validation_collate(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        key: torch.stack([item[key] for item in items])
        for key in ("image", "query", "texture")
    } | {
        "native_target": [item["target"] for item in items],
        "native_known": [item["known"] for item in items],
        "content_box": [item["content_box"] for item in items],
        "document_id": [item["document_id"] for item in items],
    }


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Tensor]:
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ("image", "query", "texture", "target", "known")
    }


def _native_query_iou_records(
    logits: Tensor,
    native_target: Sequence[Tensor],
    native_known: Sequence[Tensor],
    content_boxes: Sequence[Sequence[int]],
    document_ids: Sequence[str],
    threshold: float,
) -> list[tuple[str, float]]:
    """Project predictions to native size, then score untouched known domains."""

    if not (
        logits.shape[0]
        == len(native_target)
        == len(native_known)
        == len(content_boxes)
        == len(document_ids)
    ):
        raise ValueError("native validation batch metadata must have equal lengths")
    records: list[tuple[str, float]] = []
    for index, (target, known, box, document_id) in enumerate(
        zip(
            native_target,
            native_known,
            content_boxes,
            document_ids,
            strict=True,
        )
    ):
        y0, y1, x0, x1 = (int(value) for value in box)
        content_logits = logits[index : index + 1, :, y0:y1, x0:x1]
        native_logits = torch.nn.functional.interpolate(
            content_logits,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        records.extend(
            _query_iou_records(
                native_logits,
                target[None].to(logits.device),
                known[None].to(logits.device),
                [document_id],
                threshold,
            )
        )
    return records


def _gather_query_records(
    local_records: Sequence[tuple[str, float]],
) -> list[tuple[str, float]]:
    if not dist.is_initialized():
        return list(local_records)
    gathered: list[list[tuple[str, float]] | None] = [
        None for _ in range(dist.get_world_size())
    ]
    dist.all_gather_object(gathered, list(local_records))
    return [
        record
        for shard in gathered
        if shard is not None
        for record in shard
    ]


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    *,
    amp: bool,
    threshold: float,
) -> float | None:
    model.eval()
    local_records: list[tuple[str, float]] = []
    for batch in loader:
        tensors = {
            key: batch[key].to(device, non_blocking=True)
            for key in ("image", "query", "texture")
        }
        with torch.amp.autocast(device.type, enabled=amp):
            logits = model(
                tensors["image"],
                tensors["query"],
                tensors["texture"],
            )
        local_records.extend(
            _native_query_iou_records(
                logits,
                batch["native_target"],
                batch["native_known"],
                batch["content_box"],
                batch["document_id"],
                threshold,
            )
        )
    return _macro_from_records(_gather_query_records(local_records))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_provenance(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    data_config = config["data"]
    source_path = Path(data_config["manifest"]).resolve()
    checksum_path = Path(data_config["checksum_file"]).resolve()
    fold_path = Path(data_config["fold_manifest"]).resolve()
    source_bytes = source_path.read_bytes()
    checksum_bytes = checksum_path.read_bytes()
    fold_bytes = fold_path.read_bytes()
    try:
        checksum_manifest = json.loads(checksum_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("dataset checksum manifest is not valid JSON") from exc
    try:
        source_key = str(source_path.relative_to(checksum_path.parent))
    except ValueError as exc:
        raise ValueError("source manifest is outside checksum manifest root") from exc
    expected = checksum_manifest.get(source_key)
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    if (
        not isinstance(expected, dict)
        or expected.get("bytes") != len(source_bytes)
        or expected.get("sha256") != source_digest
    ):
        raise ValueError("source manifest does not match checksum manifest")

    def descriptor(path: Path, content: bytes) -> dict[str, Any]:
        return {
            "path": str(path),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    return {
        "source_manifest": descriptor(source_path, source_bytes),
        "checksum_manifest": descriptor(checksum_path, checksum_bytes),
        "fold_manifest": descriptor(fold_path, fold_bytes),
    }


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _dependency_versions() -> dict[str, str]:
    names = (
        "albumentations",
        "numpy",
        "opencv-python-headless",
        "pillow",
        "pyyaml",
        "scikit-image",
        "scikit-learn",
        "torch",
        "transformers",
    )
    return {name: importlib.metadata.version(name) for name in names}


def _fold_indices(value: object, *, name: str, count: int) -> list[int]:
    if not isinstance(value, list) or not all(type(index) is int for index in value):
        raise ValueError(f"fold {name} must be a list of integer indices")
    if len(set(value)) != len(value):
        raise ValueError(f"fold {name} contains duplicate indices")
    if any(index < 0 or index >= count for index in value):
        raise ValueError(f"fold {name} contains out-of-bounds indices")
    return value


def _validate_fold_manifest(
    payload: object,
    examples: Sequence[TrainingExample],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("schema") != FOLD_SCHEMA:
        raise ValueError("unsupported fold manifest schema")
    source = payload.get("source_manifest")
    if not isinstance(source, str) or Path(source).resolve() != Path(
        config["data"]["manifest"]
    ).resolve():
        raise ValueError("fold manifest source provenance does not match")
    if payload.get("seed") != int(config["data"]["fold_seed"]):
        raise ValueError("fold manifest seed does not match")
    configured_folds = list(config["folds"])
    if payload.get("fold_count") != len(configured_folds):
        raise ValueError("fold manifest fold count does not match")
    if payload.get("example_count") != len(examples):
        raise ValueError("fold manifest example count does not match")
    document_count = len({example.document_id for example in examples})
    if payload.get("document_count") != document_count:
        raise ValueError("fold manifest document count does not match")
    records = payload.get("folds")
    if not isinstance(records, list):
        raise ValueError("fold manifest folds must be a list")
    if [record.get("fold") for record in records if isinstance(record, dict)] != configured_folds:
        raise ValueError("fold manifest fold identities do not match configuration")

    all_indices = set(range(len(examples)))
    validated: list[dict[str, Any]] = []
    validation_coverage: list[int] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("fold records must be mappings")
        fold_index = record["fold"]
        train = _fold_indices(
            record.get("train_indices"),
            name=f"{fold_index} train_indices",
            count=len(examples),
        )
        valid = _fold_indices(
            record.get("valid_indices"),
            name=f"{fold_index} valid_indices",
            count=len(examples),
        )
        if set(train) & set(valid):
            raise ValueError(f"fold {fold_index} train/valid indices overlap")
        if set(train) | set(valid) != all_indices:
            raise ValueError(f"fold {fold_index} indices do not provide complete coverage")
        expected_train_ids = [examples[index].id for index in train]
        expected_valid_ids = [examples[index].id for index in valid]
        if record.get("train_ids") != expected_train_ids:
            raise ValueError(f"fold {fold_index} train IDs do not match indices")
        if record.get("valid_ids") != expected_valid_ids:
            raise ValueError(f"fold {fold_index} valid IDs do not match indices")
        expected_train_documents = sorted(
            {examples[index].document_id for index in train}
        )
        expected_valid_documents = sorted(
            {examples[index].document_id for index in valid}
        )
        if set(expected_train_documents) & set(expected_valid_documents):
            raise ValueError(f"fold {fold_index} document groups overlap")
        if record.get("train_documents") != expected_train_documents:
            raise ValueError(f"fold {fold_index} train documents do not match source")
        if record.get("valid_documents") != expected_valid_documents:
            raise ValueError(f"fold {fold_index} valid documents do not match source")
        validation_coverage.extend(valid)
        validated.append(dict(record))
    if sorted(validation_coverage) != list(range(len(examples))):
        raise ValueError("fold manifest validation coverage is not exactly once")
    return validated


def _load_fold(
    config: Mapping[str, Any],
    examples: Sequence[TrainingExample],
    fold_index: int,
) -> dict[str, Any]:
    fold_path = Path(config["data"]["fold_manifest"])
    if fold_index not in config["folds"]:
        raise ValueError(f"fold {fold_index} is not configured")
    try:
        fold_bytes = fold_path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(f"fold manifest is missing: {fold_path}") from exc
    try:
        payload = json.loads(fold_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("fold manifest is not valid UTF-8 JSON") from exc
    records = _validate_fold_manifest(payload, examples, config)
    return records[config["folds"].index(fold_index)]


def _cpu_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.detach().cpu() if isinstance(value, Tensor) else copy.deepcopy(value)
        for key, value in state.items()
    }


def _resume_invariants(config: Mapping[str, Any]) -> dict[str, Any]:
    training = {
        key: copy.deepcopy(value)
        for key, value in config["training"].items()
        if key != "max_steps"
    }
    return {
        "folds": copy.deepcopy(config["folds"]),
        "model": copy.deepcopy(config["model"]),
        "data": copy.deepcopy(config["data"]),
        "training": training,
    }


def _fold_identity(fold_record: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "fold",
        "train_indices",
        "valid_indices",
        "train_ids",
        "valid_ids",
        "train_documents",
        "valid_documents",
    )
    return {
        key: copy.deepcopy(fold_record[key])
        for key in keys
        if key in fold_record
    }


def _validate_resume_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    fold_record: Mapping[str, Any],
    provenance: Mapping[str, Any],
    world_size: int,
) -> None:
    if checkpoint.get("seed") != int(config["seed"]):
        raise ValueError("resume checkpoint seed does not match config")
    if checkpoint.get("world_size") != world_size:
        raise ValueError("resume checkpoint world size does not match")
    accumulation = int(config["training"]["gradient_accumulation"])
    if checkpoint.get("gradient_accumulation") != accumulation:
        raise ValueError("resume checkpoint accumulation does not match")
    if checkpoint.get("fold") != _fold_identity(fold_record):
        raise ValueError("resume checkpoint fold identity does not match")
    if checkpoint.get("provenance") != dict(provenance):
        raise ValueError("resume checkpoint provenance does not match")
    if checkpoint.get("resume_invariants") != _resume_invariants(config):
        raise ValueError("resume checkpoint config invariants do not match")
    rank_states = checkpoint.get("rank_states")
    if not isinstance(rank_states, list) or len(rank_states) != world_size:
        raise ValueError("resume checkpoint rank states do not match world size")


def _checkpoint_payload(
    *,
    model: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: torch.amp.GradScaler,
    stopping: EarlyStopping,
    config: Mapping[str, Any],
    fold_record: Mapping[str, Any],
    provenance: Mapping[str, Any],
    rank_states: Sequence[Mapping[str, Any]],
    world_size: int,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
    metric: float | None,
) -> dict[str, Any]:
    raw_state = _cpu_state(model.state_dict())
    ema_state = _cpu_state(ema.state_dict())
    return {
        "state_dict": ema_state,
        "model_state_dict": raw_state,
        "ema_state_dict": ema_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rank_states": copy.deepcopy(list(rank_states)),
        "training_state": {
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "global_step": global_step,
            "early_stopping": stopping.state_dict(),
            "validation_document_macro_iou": metric,
        },
        "architecture": dict(config["model"]),
        "dependency_versions": _dependency_versions(),
        "fold": _fold_identity(fold_record),
        "provenance": copy.deepcopy(dict(provenance)),
        "dataset_checksum": provenance["checksum_manifest"]["sha256"],
        "world_size": world_size,
        "gradient_accumulation": int(
            config["training"]["gradient_accumulation"]
        ),
        "resume_invariants": _resume_invariants(config),
        "git_sha": _git_sha(),
        "seed": int(config["seed"]),
        "config": copy.deepcopy(dict(config)),
    }


def _restore_training(
    path: Path,
    *,
    model: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: torch.amp.GradScaler,
    stopping: EarlyStopping,
    config: Mapping[str, Any],
    fold_record: Mapping[str, Any],
    provenance: Mapping[str, Any],
    world_size: int,
    data_loader_generator: torch.Generator,
    device: torch.device,
) -> tuple[int, int, int]:
    checkpoint = load_verified_checkpoint(path, map_location=device)
    _validate_resume_checkpoint(
        checkpoint,
        config=config,
        fold_record=fold_record,
        provenance=provenance,
        world_size=world_size,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    ema.load_state_dict(checkpoint["ema_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    training_state = checkpoint["training_state"]
    stopping.load_state_dict(training_state["early_stopping"])
    rank = dist.get_rank() if dist.is_initialized() else 0
    rank_state = checkpoint["rank_states"][rank]
    data_loader_generator.set_state(rank_state["data_loader_generator_state"])
    restore_rng_state(rank_state["rng_state"])
    return (
        int(training_state["epoch"]),
        int(training_state["batch_in_epoch"]),
        int(training_state["global_step"]),
    )


def _initialize_distributed() -> tuple[int, int, int, bool]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo"
        )
        initialized_here = True
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    return rank, world_size, local_rank, initialized_here


def _write_metadata(path: Path, payload: Mapping[str, Any], digest: str) -> None:
    metadata = {
        "checkpoint": path.name,
        "sha256": digest,
        "fold": payload["fold"],
        "seed": payload["seed"],
        "architecture": payload["architecture"],
        "dependency_versions": payload["dependency_versions"],
        "dataset_checksum": payload["dataset_checksum"],
        "provenance": payload["provenance"],
        "world_size": payload["world_size"],
        "gradient_accumulation": payload["gradient_accumulation"],
        "git_sha": payload["git_sha"],
        "training_state": payload["training_state"],
    }
    Path(f"{path}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def train_fold(config: Mapping[str, Any], fold: int) -> Path:
    """Train one configured seed on one grouped fold and return its checkpoint."""

    rank, world_size, local_rank, initialized_here = _initialize_distributed()
    try:
        if fold not in config["folds"]:
            raise ValueError(f"fold {fold} is not configured")
        seed = int(config.get("seed", config["seeds"][0]))
        _seed_everything(seed, rank)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        amp = bool(config["training"]["amp"]) and device.type == "cuda"

        manifest_path = Path(config["data"]["manifest"])
        examples = load_training_examples(manifest_path, config["data"]["root"])
        fold_record = _load_fold(config, examples, fold)
        provenance = _dataset_provenance(config)
        train_examples = [examples[index] for index in fold_record["train_indices"]]
        valid_examples = [examples[index] for index in fold_record["valid_indices"]]
        data_loader_generator = torch.Generator().manual_seed(
            seed + 100_000 + rank
        )
        validation_generator = torch.Generator().manual_seed(
            seed + 200_000 + rank
        )
        train_dataset = HatchTileDataset(
            train_examples,
            tile_size=int(config["data"]["tile_size"]),
            query_size=int(config["data"]["query_size"]),
            samples_per_epoch=int(config["training"]["samples_per_epoch"]),
            seed=seed,
            augment=True,
            cache_size=int(config["data"].get("cache_size", 1)),
        )
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(config["training"]["batch_size"]),
            sampler=train_sampler,
            num_workers=int(config["data"]["workers"]),
            pin_memory=device.type == "cuda",
            persistent_workers=False,
            generator=data_loader_generator,
        )
        valid_dataset = _ValidationDataset(
            valid_examples,
            size=int(config["data"]["validation_size"]),
            query_size=int(config["data"]["query_size"]),
        )
        valid_sampler = _DistributedEvaluationSampler(
            len(valid_dataset),
            rank,
            world_size,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=int(config["training"]["validation_batch_size"]),
            sampler=valid_sampler,
            num_workers=int(config["data"]["workers"]),
            pin_memory=device.type == "cuda",
            persistent_workers=False,
            collate_fn=_validation_collate,
            generator=validation_generator,
        )

        model = QuerySegFormer(**config["model"]).to(device)
        if world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
            )
        optimizer = AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        accumulation = int(config["training"]["gradient_accumulation"])
        updates_per_epoch = math.ceil(len(train_loader) / accumulation)
        configured_steps = int(config["training"]["epochs"]) * updates_per_epoch
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, configured_steps))
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        ema = ExponentialMovingAverage(
            _unwrap(model),
            decay=float(config["training"]["ema_decay"]),
        )
        stopping = EarlyStopping(
            patience=int(config["training"]["early_stopping_patience"]),
            min_delta=float(config["training"]["early_stopping_min_delta"]),
        )

        start_epoch = 0
        start_batch = 0
        global_step = 0
        resume = config.get("resume")
        if resume:
            start_epoch, start_batch, global_step = _restore_training(
                Path(resume),
                model=_unwrap(model),
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                stopping=stopping,
                config=config,
                fold_record=fold_record,
                provenance=provenance,
                world_size=world_size,
                data_loader_generator=data_loader_generator,
                device=device,
            )
        output_directory = (
            Path(config["output_dir"]) / f"fold-{fold}" / f"seed-{seed}"
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        last_path = output_directory / "last.pt"
        best_path = output_directory / "best.pt"
        max_steps_value = config["training"].get("max_steps")
        max_steps = int(max_steps_value) if max_steps_value is not None else None
        stop_for_steps = False

        for epoch in range(start_epoch, int(config["training"]["epochs"])):
            _advance_epoch(train_dataset, train_sampler, epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch_count = len(train_loader)
            for batch_index, batch in enumerate(train_loader):
                if epoch == start_epoch and batch_index < start_batch:
                    continue
                group_start = batch_index - (batch_index % accumulation)
                group_size = min(accumulation, batch_count - group_start)
                should_update = (
                    (batch_index + 1) % accumulation == 0
                    or batch_index + 1 == batch_count
                )
                synchronization = (
                    model.no_sync()
                    if isinstance(model, DistributedDataParallel) and not should_update
                    else nullcontext()
                )
                tensors = _to_device(batch, device)
                with synchronization:
                    with torch.amp.autocast(device.type, enabled=amp):
                        logits = model(
                            tensors["image"],
                            tensors["query"],
                            tensors["texture"],
                        )
                        loss = masked_segmentation_loss(
                            logits,
                            tensors["target"],
                            tensors["known"],
                        ) / group_size
                    scaler.scale(loss).backward()
                if not should_update:
                    continue
                step_result = optimizer_update(
                    model,
                    optimizer,
                    scaler=scaler,
                    max_grad_norm=float(config["training"]["gradient_clip_norm"]),
                )
                if not step_result.stepped:
                    continue
                scheduler.step()
                ema.update(_unwrap(model))
                global_step += 1
                if max_steps is not None and global_step >= max_steps:
                    next_epoch = epoch
                    next_batch = batch_index + 1
                    if next_batch >= batch_count:
                        next_epoch += 1
                        next_batch = 0
                    payload = _checkpoint_payload(
                        model=_unwrap(model),
                        ema=ema,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        stopping=stopping,
                        config=config,
                        fold_record=fold_record,
                        provenance=provenance,
                        rank_states=_gather_rank_states(data_loader_generator),
                        world_size=world_size,
                        epoch=next_epoch,
                        batch_in_epoch=next_batch,
                        global_step=global_step,
                        metric=None,
                    )
                    if rank == 0:
                        digest = save_checkpoint(last_path, payload)
                        _write_metadata(last_path, payload, digest)
                    stop_for_steps = True
                    break
            start_batch = 0
            if stop_for_steps:
                break

            unwrapped = _unwrap(model)
            ema.apply(unwrapped)
            metric = _validate(
                model,
                valid_loader,
                device,
                amp=amp,
                threshold=float(config["training"]["threshold"]),
            )
            ema.restore(unwrapped)
            if metric is None:
                raise RuntimeError("validation fold has no eligible positive queries")
            improved = stopping.update(metric)
            payload = _checkpoint_payload(
                model=unwrapped,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                stopping=stopping,
                config=config,
                fold_record=fold_record,
                provenance=provenance,
                rank_states=_gather_rank_states(data_loader_generator),
                world_size=world_size,
                epoch=epoch + 1,
                batch_in_epoch=0,
                global_step=global_step,
                metric=metric,
            )
            if rank == 0:
                digest = save_checkpoint(last_path, payload)
                _write_metadata(last_path, payload, digest)
                if improved:
                    best_digest = save_checkpoint(best_path, payload)
                    _write_metadata(best_path, payload, best_digest)
            if stopping.should_stop:
                break

        if dist.is_initialized():
            dist.barrier()
        return best_path if best_path.exists() else last_path
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("training config must contain a mapping")
    base = path.parent
    for key in ("manifest", "root", "fold_manifest", "checksum_file"):
        config["data"][key] = str((base / config["data"][key]).resolve())
    config["output_dir"] = str((base / config["output_dir"]).resolve())
    return config


def _select_run_seeds(
    configured_seeds: Sequence[int],
    *,
    selected_seed: int | None,
    resume: Path | None,
) -> list[int]:
    seeds = [int(seed) for seed in configured_seeds]
    if selected_seed is not None:
        if selected_seed not in seeds:
            raise ValueError(f"seed {selected_seed} is not configured")
        return [selected_seed]
    if resume is not None and len(seeds) != 1:
        raise ValueError("resume requires selecting a single seed with --seed")
    return seeds


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int)
    arguments = parser.parse_args(argv)
    config = _load_config(arguments.config)
    if arguments.max_steps is not None:
        if arguments.max_steps < 1:
            parser.error("--max-steps must be positive")
        config["training"]["max_steps"] = arguments.max_steps
    if arguments.resume is not None:
        config["resume"] = str(arguments.resume.resolve())

    try:
        seeds = _select_run_seeds(
            config["seeds"],
            selected_seed=arguments.seed,
            resume=arguments.resume,
        )
    except ValueError as exc:
        parser.error(str(exc))
    paths = []
    for seed in seeds:
        seeded_config = copy.deepcopy(config)
        seeded_config["seed"] = int(seed)
        paths.append(train_fold(seeded_config, arguments.fold))
    if int(os.environ.get("RANK", "0")) == 0:
        for path in paths:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
