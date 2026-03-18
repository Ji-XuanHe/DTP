import torch
import torch.nn as nn
import torch.nn.functional as F


class BioInspiredLuminanceEnhancer(nn.Module):
    """Paper name: the luminance branch in SDR with bio-inspired activation."""

    def __init__(self) -> None:
        super().__init__()
        n = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=torch.float32)
        sigma = torch.tensor([[0.5] * 8], dtype=torch.float32)
        bias = torch.tensor([0.0001] * 8, dtype=torch.float32)

        self.n = nn.Parameter(n)
        self.sigma = nn.Parameter(sigma)
        self.register_buffer("bias", bias)

        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 1),
            nn.Sigmoid(),
        )

        features = 32
        self.conv_1 = nn.Sequential(
            nn.Conv2d(24, features, 3, 1, 1),
            nn.InstanceNorm2d(features),
            nn.PReLU(),
        )
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(features, features * 2, 3, 2, 1),
                    nn.InstanceNorm2d(features * 2),
                    nn.PReLU(),
                ),
                nn.Sequential(
                    nn.Conv2d(features * 2, features * 4, 3, 2, 1),
                    nn.InstanceNorm2d(features * 4),
                    nn.PReLU(),
                ),
            ]
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(features * 4, features * 4, 7, 1, 3),
            nn.InstanceNorm2d(features * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(features * 4, 1, 7, 1, 3),
            nn.Sigmoid(),
        )
        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(features * 8, features * 2, 4, 2, 1),
                    nn.InstanceNorm2d(features * 2),
                    nn.PReLU(),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(features * 4, features, 4, 2, 1),
                    nn.InstanceNorm2d(features),
                    nn.PReLU(),
                ),
            ]
        )
        self.output = nn.Sequential(
            nn.Conv2d(features, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _normalize_nr_response(response: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = response.shape
        normalized = F.batch_norm(
            response.reshape(batch * channels, 1, height, width),
            running_mean=None,
            running_var=None,
            weight=None,
            bias=None,
            training=True,
            momentum=0.1,
            eps=1e-5,
        )
        return normalized.reshape(batch, channels, height, width)

    def forward(self, luminance_llr: torch.Tensor) -> torch.Tensor:
        luminance_llr = torch.clamp(luminance_llr, min=1e-6)
        weights = self.weight_net(luminance_llr)
        nr_outputs = []
        for index in range(8):
            exponent = self.n[0, index]
            sigma = self.sigma[0, index]
            response = torch.pow(luminance_llr, exponent) / (
                torch.pow(luminance_llr, exponent) + torch.pow(sigma, exponent) + self.bias[index]
            )
            response = self._normalize_nr_response(response)
            nr_outputs.append(response * weights[:, index : index + 1])

        features = self.conv_1(torch.cat(nr_outputs, dim=1))
        skip_connections = []
        input_size = features.shape[2:]
        for encoder in self.encoder:
            features = encoder(features)
            skip_connections.append(features)

        features = features * self.spatial_attention(features)
        for index, decoder in enumerate(self.decoder):
            target_size = skip_connections[-(index + 1)].shape[2:]
            features = F.interpolate(features, size=target_size, mode="bilinear", align_corners=False)
            features = torch.cat([features, skip_connections[-(index + 1)]], dim=1)
            features = decoder(features)

        luminance_hlr = self.output(features)
        luminance_hlr = F.interpolate(luminance_hlr, size=input_size, mode="bilinear", align_corners=False)

        input_mean = torch.mean(luminance_llr, dim=1, keepdim=True)
        output_mean = torch.mean(luminance_hlr, dim=1, keepdim=True)
        return luminance_hlr * (input_mean / (output_mean + 1e-6))


class HierarchicalTextureDenoiser(nn.Module):
    """Paper name: the texture branch in SDR with residual hierarchical denoising."""

    def __init__(self, features: int = 64) -> None:
        super().__init__()
        self.activation = nn.PReLU()
        self.conv1 = nn.Conv2d(3, features, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.conv_mid = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.conv_out = nn.Conv2d(features, 3, 1, 1, 0, bias=True)
        self.norm_features = nn.BatchNorm2d(features, affine=True)

    def _residual_stack(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.conv_mid(x))
        refined = self.activation(self.conv_mid(hidden))
        return self.activation(self.conv_out(x + refined))

    def forward(self, texture_llr: torch.Tensor) -> torch.Tensor:
        stage1 = self.norm_features(self.activation(self.conv1(texture_llr)))
        out1 = self._residual_stack(stage1)

        stage2 = self.norm_features(self.activation(self.conv2(stage1)))
        out2 = self._residual_stack(stage2)

        stage3 = self.norm_features(self.activation(self.conv2(stage2)))
        out3 = self._residual_stack(stage3)

        stage4 = self.norm_features(self.activation(self.conv2(stage3)))
        out4 = self._residual_stack(stage4)

        stage5 = self.norm_features(self.activation(self.conv2(stage4)))
        out5 = self._residual_stack(stage5)

        return out1 + out2 + out3 + out4 + out5


class SemanticsSpecificDualPathRepresentation(nn.Module):
    """Paper name: Semantics-specific Dual-path Representation (SDR)."""

    def __init__(self) -> None:
        super().__init__()
        self.luminance_enhancer = BioInspiredLuminanceEnhancer()
        self.texture_denoiser = HierarchicalTextureDenoiser()

    def forward(self, luminance_llr: torch.Tensor, texture_llr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        luminance_hlr = self.luminance_enhancer(luminance_llr)
        texture_dlr = self.texture_denoiser(texture_llr)
        return luminance_hlr, texture_dlr


EnhanceNet = BioInspiredLuminanceEnhancer
DenoiseNet = HierarchicalTextureDenoiser
SDRModule = SemanticsSpecificDualPathRepresentation
