"""Shared-encoder, query-conditioned dense hatch segmentation model."""

from __future__ import annotations

import re

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import SegformerConfig, SegformerModel


_MIT_VARIANTS = {
    "nvidia/mit-b0": ((32, 64, 160, 256), (2, 2, 2, 2)),
    "nvidia/mit-b1": ((64, 128, 320, 512), (2, 2, 2, 2)),
    "nvidia/mit-b2": ((64, 128, 320, 512), (3, 4, 6, 3)),
    "nvidia/mit-b3": ((64, 128, 320, 512), (3, 4, 18, 3)),
    "nvidia/mit-b4": ((64, 128, 320, 512), (3, 8, 27, 3)),
    "nvidia/mit-b5": ((64, 128, 320, 512), (3, 6, 40, 3)),
}


def _offline_config(backbone: str) -> SegformerConfig:
    variant = _MIT_VARIANTS.get(backbone)
    if variant is None:
        try:
            return SegformerConfig.from_pretrained(
                backbone,
                local_files_only=True,
            )
        except OSError as exc:
            raise ValueError(
                f"no offline configuration is available for backbone {backbone!r}"
            ) from exc
    hidden_sizes, depths = variant
    return SegformerConfig(
        hidden_sizes=list(hidden_sizes),
        depths=list(depths),
        decoder_hidden_size=256,
        output_hidden_states=True,
    )


class _ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        groups = min(8, output_channels)
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
        )


class QuerySegFormer(nn.Module):
    """Segment a drawing using a query crop and four texture channels."""

    def __init__(
        self,
        backbone: str = "nvidia/mit-b2",
        *,
        pretrained: bool = True,
        revision: str | None = None,
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()
        if decoder_channels <= 0 or decoder_channels % min(8, decoder_channels):
            raise ValueError("decoder_channels must be positive and divisible by 8")
        if pretrained:
            if revision is None or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
                raise ValueError(
                    "pretrained weights require a pinned 40-character revision"
                )
            self.encoder = SegformerModel.from_pretrained(
                backbone,
                revision=revision,
            )
        else:
            self.encoder = SegformerModel(_offline_config(backbone))

        hidden_sizes = tuple(self.encoder.config.hidden_sizes)
        self.lateral = nn.ModuleList(
            nn.Conv2d(channels, decoder_channels, kernel_size=1)
            for channels in hidden_sizes
        )
        self.film = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(1, decoder_channels, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(decoder_channels, 2 * decoder_channels, kernel_size=1),
            )
            for _ in hidden_sizes
        )
        self.smooth = nn.ModuleList(
            _ConvBlock(decoder_channels, decoder_channels)
            for _ in hidden_sizes
        )
        self.refine = nn.Sequential(
            _ConvBlock(decoder_channels + 4, decoder_channels),
            _ConvBlock(decoder_channels, decoder_channels // 2),
        )
        self.classifier = nn.Conv2d(decoder_channels // 2, 1, kernel_size=1)

    @property
    def drawing_encoder(self) -> SegformerModel:
        """Return the encoder used for drawing features."""

        return self.encoder

    @property
    def query_encoder(self) -> SegformerModel:
        """Return the same encoder used for query features."""

        return self.encoder

    def _features(self, pixels: Tensor) -> tuple[Tensor, ...]:
        output = self.encoder(
            pixel_values=pixels,
            output_hidden_states=True,
            return_dict=True,
        )
        return tuple(output.hidden_states)

    def forward(self, image: Tensor, query: Tensor, texture: Tensor) -> Tensor:
        """Return native-resolution logits with shape ``[B, 1, H, W]``."""

        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B, 3, H, W]")
        if query.ndim != 4 or query.shape[1] != 3:
            raise ValueError("query must have shape [B, 3, QH, QW]")
        if texture.ndim != 4 or texture.shape[1] != 4:
            raise ValueError("texture must have shape [B, 4, H, W]")
        if image.shape[0] != query.shape[0] or image.shape[0] != texture.shape[0]:
            raise ValueError("image, query, and texture batches must match")

        drawing_features = self._features(image)
        query_features = self._features(query)
        conditioned: list[Tensor] = []
        for index, (drawing, query_level) in enumerate(
            zip(drawing_features, query_features, strict=True)
        ):
            prototype = F.adaptive_avg_pool2d(query_level, output_size=1)
            normalized_drawing = F.normalize(drawing, dim=1, eps=1e-6)
            normalized_prototype = F.normalize(prototype, dim=1, eps=1e-6)
            similarity = (normalized_drawing * normalized_prototype).sum(
                dim=1,
                keepdim=True,
            )
            gamma, beta = self.film[index](similarity).chunk(2, dim=1)
            lateral = self.lateral[index](drawing)
            conditioned.append(lateral * (1.0 + torch.tanh(gamma)) + beta)

        top_down: Tensor | None = None
        for index in range(len(conditioned) - 1, -1, -1):
            level = conditioned[index]
            if top_down is not None:
                level = level + F.interpolate(
                    top_down,
                    size=level.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            top_down = self.smooth[index](level)
        if top_down is None:
            raise RuntimeError("encoder did not return feature maps")

        output_size = image.shape[-2:]
        dense = F.interpolate(
            top_down,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        resized_texture = F.interpolate(
            texture,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.classifier(self.refine(torch.cat((dense, resized_texture), dim=1)))
