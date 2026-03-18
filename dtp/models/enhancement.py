import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhanceNet(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=1e-6)
        weights = self.weight_net(x)
        nr_outputs = []
        for index in range(8):
            exponent = self.n[0, index]
            sigma = self.sigma[0, index]
            response = torch.pow(x, exponent) / (
                torch.pow(x, exponent) + torch.pow(sigma, exponent) + self.bias[index]
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

        output = self.output(features)
        output = F.interpolate(output, size=input_size, mode="bilinear", align_corners=False)

        x_mean = torch.mean(x, dim=1, keepdim=True)
        output_mean = torch.mean(output, dim=1, keepdim=True)
        return output * (x_mean / (output_mean + 1e-6))
