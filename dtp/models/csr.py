import torch
import torch.nn as nn

from .blocks import (
    AdaptiveFusion,
    AdaptiveLayerNorm,
    AdaptiveMixingModule,
    FrequencyAwareModule,
    ResBlock,
)


class CrossFrequencySemanticRecomposition(nn.Module):
    """Paper name: Cross-frequency Semantic Recomposition (CSR)."""

    def __init__(
        self,
        upscale: int,
        in_channels: int = 3,
        features: int = 24,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if upscale not in (2, 4):
            raise ValueError(f"Unsupported upscale factor: {upscale}")

        self.dropout = nn.Dropout(dropout_rate)
        self.conv_first = nn.Conv2d(in_channels, features, 3, 1, 1)
        self.frequency_aware = FrequencyAwareModule(features)
        self.adaptive_fusion = AdaptiveFusion(features)
        self.fusion = nn.Sequential(
            nn.Conv2d(features, features, 3, 1, 1),
            AdaptiveLayerNorm(features),
            nn.LeakyReLU(0.2, inplace=True),
            self.dropout,
            nn.Conv2d(features, features, 3, 1, 1),
            AdaptiveLayerNorm(features),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.feature_mixing = AdaptiveMixingModule(features)
        self.multi_scale_fusion = nn.ModuleList(
            [
                nn.Conv2d(features, features, 3, 1, 1),
                nn.Conv2d(features, features, 5, 1, 2),
                nn.Conv2d(features, features, 7, 1, 3),
            ]
        )
        self.feature_gate = nn.Sequential(
            nn.Conv2d(features * 3, features, 1),
            nn.Sigmoid(),
        )
        self.body = nn.ModuleList([ResBlock(features) for _ in range(3)])
        self.reconstruction = nn.Sequential(
            nn.Conv2d(features * 2, features * 2, 3, 1, 1),
            AdaptiveLayerNorm(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            self.dropout,
            ResBlock(features * 2),
            nn.Conv2d(features * 2, features * 2, 3, 1, 1),
            AdaptiveLayerNorm(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if upscale == 2:
            self.upconv = nn.Sequential(
                nn.Conv2d(features * 2, features * 4, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.PixelShuffle(2),
                ResBlock(features),
                nn.Conv2d(features, in_channels, 3, 1, 1),
            )
        else:
            self.upconv = nn.Sequential(
                nn.Conv2d(features * 2, features * 4, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.PixelShuffle(2),
                ResBlock(features),
                nn.Conv2d(features, features * 4, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.PixelShuffle(2),
                ResBlock(features),
                nn.Conv2d(features, in_channels, 3, 1, 1),
            )

    def forward(
        self,
        original_llr: torch.Tensor,
        texture_dlr: torch.Tensor,
        luminance_hlr: torch.Tensor,
    ) -> torch.Tensor:
        feat_input = self.conv_first(original_llr)
        feat_texture = self.conv_first(texture_dlr)
        feat_luminance = self.conv_first(luminance_hlr)

        texture_aware, luminance_aware = self.frequency_aware(feat_texture, feat_luminance)
        fused = self.adaptive_fusion(feat_input, texture_aware, luminance_aware)
        features = self.fusion(fused)

        multi_scale = torch.cat([conv(features) for conv in self.multi_scale_fusion], dim=1)
        features = self.feature_mixing(feat_input, features * self.feature_gate(multi_scale))

        refined_features = features
        for block in self.body:
            refined_features = block(refined_features)

        reconstruction = self.reconstruction(torch.cat([feat_input, refined_features], dim=1))
        restored_hsr = self.upconv(reconstruction)
        return torch.tanh(restored_hsr) * 0.5 + 0.5


CSRModule = CrossFrequencySemanticRecomposition
LLSRNet = CrossFrequencySemanticRecomposition
