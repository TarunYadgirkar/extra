import pytest
import torch

from hatchmatch.losses import (
    masked_boundary_loss,
    masked_focal_loss,
    masked_segmentation_loss,
    masked_soft_dice_loss,
)


LOSS_COMPONENTS = (
    masked_focal_loss,
    masked_soft_dice_loss,
    masked_boundary_loss,
    masked_segmentation_loss,
)


@pytest.mark.parametrize("loss_function", LOSS_COMPONENTS)
def test_every_loss_component_ignores_unknown_logits_and_targets(loss_function):
    logits = torch.tensor([[[[-1.0, 20.0], [2.0, -20.0]]]])
    target = torch.tensor([[[[0.0, 0.0], [1.0, 0.0]]]])
    known = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    changed_logits = logits.clone()
    changed_logits[..., 0, 1] = -100.0
    changed_logits[..., 1, 1] = 100.0
    changed_target = target.clone()
    changed_target[..., 0, 1] = 1.0
    changed_target[..., 1, 1] = 1.0

    original = loss_function(logits, target, known)
    changed = loss_function(changed_logits, changed_target, known)

    torch.testing.assert_close(original, changed, rtol=0, atol=0)


def test_masked_focal_loss_matches_binary_focal_definition():
    logits = torch.zeros(1, 1, 1, 2)
    target = torch.tensor([[[[0.0, 1.0]]]])
    known = torch.ones_like(target)
    expected = torch.log(torch.tensor(2.0)) / 8.0

    actual = masked_focal_loss(logits, target, known)

    torch.testing.assert_close(actual, expected)


def test_masked_soft_dice_loss_uses_only_known_probabilities():
    logits = torch.zeros(1, 1, 1, 3)
    target = torch.tensor([[[[1.0, 0.0, 1.0]]]])
    known = torch.tensor([[[[1.0, 1.0, 0.0]]]])
    expected = torch.tensor(1.0 - (1.0 + 1e-6) / (2.0 + 1e-6))

    actual = masked_soft_dice_loss(logits, target, known)

    torch.testing.assert_close(actual, expected)


def test_masked_boundary_loss_rewards_matching_known_edges():
    target = torch.tensor([[[[0.0, 0.0, 1.0, 1.0]]]])
    known = torch.ones_like(target)
    matching_logits = (target * 2.0 - 1.0) * 20.0
    flat_logits = torch.zeros_like(target)

    matching = masked_boundary_loss(matching_logits, target, known)
    flat = masked_boundary_loss(flat_logits, target, known)

    assert matching < flat


def test_masked_boundary_loss_matches_known_pair_numerical_oracle():
    probabilities = torch.tensor([[[[0.1, 0.4, 0.9, 0.2, 0.6]]]])
    logits = torch.logit(probabilities)
    target = torch.tensor([[[[0.0, 1.0, 0.0, 1.0, 1.0]]]])
    known = torch.tensor([[[[1.0, 1.0, 0.0, 1.0, 1.0]]]])
    # Only pairs (0, 1) and (3, 4) have two known endpoints. Their errors are
    # abs(0.3 - 1.0) and abs(0.4 - 0.0), normalized by two valid pairs.
    expected = torch.tensor((0.7 + 0.4) / 2.0)

    actual = masked_boundary_loss(logits, target, known)

    torch.testing.assert_close(actual, expected)


def test_segmentation_loss_is_weighted_sum_of_all_components():
    logits = torch.tensor([[[[-1.0, 0.5], [2.0, -0.5]]]])
    target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    known = torch.ones_like(target)
    expected = (
        0.5 * masked_focal_loss(logits, target, known)
        + 0.4 * masked_soft_dice_loss(logits, target, known)
        + 0.1 * masked_boundary_loss(logits, target, known)
    )

    actual = masked_segmentation_loss(logits, target, known)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("loss_function", LOSS_COMPONENTS)
def test_zero_known_pixels_return_differentiable_zero(loss_function):
    logits = torch.randn(2, 1, 3, 5, requires_grad=True)
    target = torch.randint(0, 2, logits.shape).float()
    known = torch.zeros_like(target)

    loss = loss_function(logits, target, known)
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
