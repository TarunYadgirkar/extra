import torch
import pytest

from hatchmatch.model import QuerySegFormer


def test_model_preserves_arbitrary_drawing_shape_and_uses_query():
    torch.manual_seed(0)
    model = QuerySegFormer(backbone="nvidia/mit-b2", pretrained=False).eval()
    image = torch.rand(1, 3, 65, 79).expand(2, -1, -1, -1)
    texture = torch.rand(1, 4, 65, 79).expand(2, -1, -1, -1)
    query = torch.cat(
        (torch.rand(1, 3, 47, 53), torch.zeros(1, 3, 47, 53)),
        dim=0,
    )

    with torch.no_grad():
        logits = model(image, query, texture)

    assert logits.shape == (2, 1, 65, 79)
    assert not torch.equal(logits[0], logits[1])


def test_drawing_and_query_encoders_are_the_same_module():
    model = QuerySegFormer(backbone="nvidia/mit-b0", pretrained=False)

    assert model.drawing_encoder is model.query_encoder


def test_pretrained_encoder_requires_an_immutable_revision():
    with pytest.raises(ValueError, match="revision"):
        QuerySegFormer(backbone="nvidia/mit-b0", pretrained=True)


def test_model_fuses_texture_channels():
    torch.manual_seed(1)
    model = QuerySegFormer(backbone="nvidia/mit-b0", pretrained=False).eval()
    image = torch.rand(1, 3, 37, 43).expand(2, -1, -1, -1)
    query = torch.rand(1, 3, 35, 39).expand(2, -1, -1, -1)
    texture = torch.cat(
        (
            torch.zeros(1, 4, 37, 43),
            torch.ones(1, 4, 37, 43),
        ),
        dim=0,
    )

    with torch.no_grad():
        logits = model(image, query, texture)

    assert not torch.equal(logits[0], logits[1])


def test_gradients_reach_shared_encoder_query_conditioning_and_texture_path():
    torch.manual_seed(2)
    model = QuerySegFormer(backbone="nvidia/mit-b0", pretrained=False)
    image = torch.rand(1, 3, 35, 41, requires_grad=True)
    query = torch.rand(1, 3, 33, 37, requires_grad=True)
    texture = torch.rand(1, 4, 35, 41, requires_grad=True)

    model(image, query, texture).mean().backward()

    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert query.grad.abs().sum() > 0
    assert texture.grad is not None and texture.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.encoder.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.film.parameters()
    )
