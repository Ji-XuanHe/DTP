import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleWaveletNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, levels: int = 3) -> None:
        super().__init__()
        self.levels = levels
        self.in_channels = in_channels
        self.wavelet_channels = in_channels * 4
        self.out_channels_per_level = out_channels // levels

        self.wavelet_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.wavelet_channels, self.out_channels_per_level, 3, 1, 1),
                    nn.InstanceNorm2d(self.out_channels_per_level),
                    nn.PReLU(),
                )
                for _ in range(levels)
            ]
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(self.out_channels_per_level * levels, out_channels, 1, 1, 0),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU(),
        )

    def wavelet_transform(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        coeffs = []
        for batch_index in range(x.shape[0]):
            image = x[batch_index, 0].detach().cpu().numpy()
            ll, (lh, hl, hh) = pywt.dwt2(image, "haar")
            bands = []
            for band in (ll, lh, hl, hh):
                band_tensor = torch.from_numpy(band).float().unsqueeze(0).unsqueeze(0)
                if band_tensor.shape[2:] != x.shape[2:]:
                    band_tensor = F.interpolate(
                        band_tensor,
                        size=x.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                bands.append(band_tensor)
            coeffs.append(torch.cat(bands, dim=1))
        return torch.cat(coeffs, dim=0).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale_features = []
        current_x = x
        for level, wavelet_conv in enumerate(self.wavelet_convs):
            wavelet_features = []
            for channel_index in range(self.in_channels):
                coeffs = self.wavelet_transform(current_x[:, channel_index : channel_index + 1])
                wavelet_features.append(coeffs)
            wavelet_tensor = torch.cat(wavelet_features, dim=1)
            features = wavelet_conv(wavelet_tensor)
            if level > 0:
                features = F.interpolate(
                    features,
                    size=x.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            multi_scale_features.append(features)
            if level < self.levels - 1:
                current_x = F.avg_pool2d(current_x, 2)
        return self.fusion(torch.cat(multi_scale_features, dim=1))


class FrequencyAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 4, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))
        return x * attention


class DecomposeNet(nn.Module):
    def __init__(self, features: int = 64) -> None:
        super().__init__()
        self.wavelet_net = MultiScaleWaveletNet(3, features)
        self.frequency_attention = FrequencyAttention(features // 4)
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(features, features, 3, 1, 1),
            nn.BatchNorm2d(features),
            nn.PReLU(),
            nn.Conv2d(features, features, 3, 1, 1),
            nn.BatchNorm2d(features),
            nn.PReLU(),
        )
        self.separator = nn.Sequential(
            nn.Conv2d(features, 6, 3, 1, 1),
            nn.BatchNorm2d(6),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        wavelet_features = self.wavelet_net(x)
        attended_features = self.frequency_attention(wavelet_features)
        enhanced_features = self.feature_extraction(attended_features)
        separated = self.separator(enhanced_features)
        high_frequency = torch.stack([separated[:, index, :, :] for index in range(3)], dim=1)
        low_frequency = torch.stack([separated[:, index, :, :] for index in range(3, 6)], dim=1)
        return high_frequency, low_frequency
