"""Known-pixel-only objectives for dense hatch segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _loss_inputs(
    logits: Tensor,
    target: Tensor,
    known: Tensor,
) -> tuple[Tensor, Tensor]:
    if logits.shape != target.shape or logits.shape != known.shape:
        raise ValueError("logits, target, and known must have identical shapes")
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("loss inputs must have shape [B, 1, H, W]")
    target_values = target.to(device=logits.device, dtype=logits.dtype)
    mask = (known.to(device=logits.device) == 1).to(dtype=logits.dtype)
    return target_values, mask


def _masked_mean(values: Tensor, mask: Tensor, logits: Tensor) -> Tensor:
    count = mask.sum()
    if count.item() == 0:
        return logits.sum() * 0.0
    return (values * mask).sum() / count


def masked_focal_loss(
    logits: Tensor,
    target: Tensor,
    known: Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """Return binary focal loss reduced over exactly known pixels."""

    target_values, mask = _loss_inputs(logits, target, known)
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits,
        target_values,
        reduction="none",
    )
    probability = torch.sigmoid(logits)
    probability_correct = (
        probability * target_values + (1.0 - probability) * (1.0 - target_values)
    )
    alpha_weight = (
        alpha * target_values + (1.0 - alpha) * (1.0 - target_values)
    )
    focal = alpha_weight * (1.0 - probability_correct).pow(gamma) * cross_entropy
    return _masked_mean(focal, mask, logits)


def masked_soft_dice_loss(
    logits: Tensor,
    target: Tensor,
    known: Tensor,
    *,
    epsilon: float = 1e-6,
) -> Tensor:
    """Return soft Dice loss with unknown probabilities excluded."""

    target_values, mask = _loss_inputs(logits, target, known)
    if mask.sum().item() == 0:
        return logits.sum() * 0.0
    probability = torch.sigmoid(logits)
    intersection = (probability * target_values * mask).sum()
    denominator = ((probability + target_values) * mask).sum()
    return 1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)


def masked_boundary_loss(
    logits: Tensor,
    target: Tensor,
    known: Tensor,
) -> Tensor:
    """Compare adjacent probability changes whose endpoints are both known."""

    target_values, mask = _loss_inputs(logits, target, known)
    probability = torch.sigmoid(logits)

    horizontal_mask = mask[..., :, 1:] * mask[..., :, :-1]
    horizontal_error = (
        (probability[..., :, 1:] - probability[..., :, :-1]).abs()
        - (target_values[..., :, 1:] - target_values[..., :, :-1]).abs()
    ).abs()
    vertical_mask = mask[..., 1:, :] * mask[..., :-1, :]
    vertical_error = (
        (probability[..., 1:, :] - probability[..., :-1, :]).abs()
        - (target_values[..., 1:, :] - target_values[..., :-1, :]).abs()
    ).abs()

    count = horizontal_mask.sum() + vertical_mask.sum()
    if count.item() == 0:
        return logits.sum() * 0.0
    total = (
        (horizontal_error * horizontal_mask).sum()
        + (vertical_error * vertical_mask).sum()
    )
    return total / count


def masked_segmentation_loss(
    logits: Tensor,
    target: Tensor,
    known: Tensor,
) -> Tensor:
    """Combine focal, soft Dice, and boundary objectives on known pixels."""

    return (
        0.5 * masked_focal_loss(logits, target, known)
        + 0.4 * masked_soft_dice_loss(logits, target, known)
        + 0.1 * masked_boundary_loss(logits, target, known)
    )
