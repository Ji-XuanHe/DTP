import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.activation(self.conv1(x))
        residual = self.conv2(residual)
        return x + residual


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return self.sigmoid(pooled)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.conv(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(attention)


class FrequencyAwareModule(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(
        self,
        texture_features: torch.Tensor,
        luminance_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        texture_features = self.channel_attention(texture_features) * texture_features
        luminance_features = self.channel_attention(luminance_features) * luminance_features
        texture_features = self.spatial_attention(texture_features) * texture_features
        luminance_features = self.spatial_attention(luminance_features) * luminance_features
        return texture_features, luminance_features


class AdaptiveFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 3, 1),
            nn.Softmax(dim=1),
        )

    def forward(
        self,
        original: torch.Tensor,
        texture_features: torch.Tensor,
        luminance_features: torch.Tensor,
    ) -> torch.Tensor:
        weights = self.attention(torch.cat([original, texture_features, luminance_features], dim=1))
        return (
            weights[:, [0], :, :] * original
            + weights[:, [1], :, :] * texture_features
            + weights[:, [2], :, :] * luminance_features
        )


class AdaptiveMixingModule(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        reduced = max(channels // 4, 1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(reduced, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, base: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        mixing_weight = self.attention(enhanced)
        return mixing_weight * enhanced + (1.0 - mixing_weight) * base


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        normalized = self.norm(x.permute(0, 2, 3, 1))
        return normalized.permute(0, 3, 1, 2).reshape(batch, channels, height, width)
