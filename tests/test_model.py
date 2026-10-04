import pytest
import torch
from transformers import SegformerConfig, SegformerModel

from hatchmatch.model import QuerySegFormer


MIT_VARIANTS = (
    ("nvidia/mit-b0", (32, 64, 160, 256), (2, 2, 2, 2), 256),
    ("nvidia/mit-b1", (64, 128, 320, 512), (2, 2, 2, 2), 768),
    ("nvidia/mit-b2", (64, 128, 320, 512), (3, 4, 6, 3), 768),
    ("nvidia/mit-b3", (64, 128, 320, 512), (3, 4, 18, 3), 768),
    ("nvidia/mit-b4", (64, 128, 320, 512), (3, 8, 27, 3), 768),
    ("nvidia/mit-b5", (64, 128, 320, 512), (3, 6, 40, 3), 768),
)
B2_REVISION = "3bb39e8739149c3777d0325349b2a6c32c6413db"


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


@pytest.mark.parametrize(
    ("backbone", "hidden_sizes", "depths", "decoder_hidden_size"),
    MIT_VARIANTS,
)
def test_all_canonical_mit_variants_construct_offline_with_expected_dimensions(
    backbone,
    hidden_sizes,
    depths,
    decoder_hidden_size,
):
    model = QuerySegFormer(
        backbone=backbone,
        pretrained=False,
        decoder_channels=16,
    ).eval()

    with torch.no_grad():
        features = model._features(torch.rand(1, 3, 32, 40))

    assert tuple(model.encoder.config.hidden_sizes) == hidden_sizes
    assert tuple(model.encoder.config.depths) == depths
    assert model.encoder.config.decoder_hidden_size == decoder_hidden_size
    assert tuple(feature.shape[1] for feature in features) == hidden_sizes
    assert tuple(layer.in_channels for layer in model.lateral) == hidden_sizes
    assert all(layer.out_channels == 16 for layer in model.lateral)
    assert model.drawing_encoder is model.query_encoder is model.encoder


def test_unpinned_pretrained_encoder_requires_an_immutable_revision():
    with pytest.raises(ValueError, match="revision"):
        QuerySegFormer(backbone="nvidia/mit-b0", pretrained=True)


@pytest.mark.parametrize(
    ("arguments", "expected_backbone", "expected_revision"),
    [
        ({}, "nvidia/mit-b2", B2_REVISION),
        (
            {
                "backbone": "nvidia/mit-b0",
                "pretrained": True,
                "revision": "a" * 40,
            },
            "nvidia/mit-b0",
            "a" * 40,
        ),
    ],
)
def test_pretrained_revision_is_forwarded_unchanged(
    monkeypatch,
    arguments,
    expected_backbone,
    expected_revision,
):
    calls = []

    def fake_from_pretrained(cls, backbone, **kwargs):
        calls.append((backbone, kwargs))
        return cls(SegformerConfig())

    monkeypatch.setattr(
        SegformerModel,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    QuerySegFormer(**arguments)

    assert calls == [(expected_backbone, {"revision": expected_revision})]


def test_normalized_similarity_reaches_each_film_level(monkeypatch):
    model = QuerySegFormer(
        backbone="nvidia/mit-b0",
        pretrained=False,
        decoder_channels=8,
    ).eval()
    drawing_features = []
    query_features = []
    for channels in model.encoder.config.hidden_sizes:
        drawing = torch.zeros(1, channels, 1, 3)
        drawing[:, 0, :, 0] = 2.0
        drawing[:, 1, :, 1] = 3.0
        drawing[:, 0, :, 2] = -4.0
        prototype_source = torch.zeros(1, channels, 2, 2)
        prototype_source[:, 0] = 5.0
        drawing_features.append(drawing)
        query_features.append(prototype_source)
    calls = iter((tuple(drawing_features), tuple(query_features)))
    monkeypatch.setattr(model, "_features", lambda _pixels: next(calls))
    similarities = []
    handles = [
        film.register_forward_pre_hook(
            lambda _module, inputs: similarities.append(inputs[0].detach().clone())
        )
        for film in model.film
    ]

    with torch.no_grad():
        model(
            torch.zeros(1, 3, 5, 7),
            torch.zeros(1, 3, 5, 7),
            torch.zeros(1, 4, 5, 7),
        )
    for handle in handles:
        handle.remove()

    expected = torch.tensor([[[[1.0, 0.0, -1.0]]]])
    assert len(similarities) == 4
    for similarity in similarities:
        torch.testing.assert_close(similarity, expected)


def test_film_fpn_and_texture_fusion_are_wired_before_two_refinement_blocks():
    model = QuerySegFormer(
        backbone="nvidia/mit-b0",
        pretrained=False,
        decoder_channels=16,
    ).eval()
    smooth_order = []
    smooth_handles = [
        module.register_forward_hook(
            lambda _module, _inputs, _output, index=index: smooth_order.append(index)
        )
        for index, module in enumerate(model.smooth)
    ]
    refinement_inputs = []
    refinement_handle = model.refine.register_forward_pre_hook(
        lambda _module, inputs: refinement_inputs.append(inputs[0].detach().clone())
    )
    texture = torch.rand(1, 4, 35, 41)

    with torch.no_grad():
        model(
            torch.rand(1, 3, 35, 41),
            torch.rand(1, 3, 33, 37),
            texture,
        )
    refinement_handle.remove()
    for handle in smooth_handles:
        handle.remove()

    assert len(model.film) == len(model.smooth) == len(model.lateral) == 4
    assert all(film[-1].out_channels == 32 for film in model.film)
    assert smooth_order == [3, 2, 1, 0]
    assert len(model.refine) == 2
    assert model.refine[0][0].in_channels == 16 + 4
    torch.testing.assert_close(refinement_inputs[0][:, -4:], texture)


@pytest.mark.parametrize("decoder_channels", [0, 1])
def test_decoder_rejects_widths_that_create_zero_channel_blocks(decoder_channels):
    with pytest.raises(ValueError, match="at least 2"):
        QuerySegFormer(
            backbone="nvidia/mit-b0",
            pretrained=False,
            decoder_channels=decoder_channels,
        )


def test_decoder_accepts_widths_without_an_artificial_divisibility_rule():
    model = QuerySegFormer(
        backbone="nvidia/mit-b0",
        pretrained=False,
        decoder_channels=10,
    )

    assert model.refine[1][0].out_channels == 5


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
