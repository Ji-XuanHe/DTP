import torch
import torch.nn as nn


class DenoiseNet(nn.Module):
    def __init__(self, features: int = 64) -> None:
        super().__init__()
        self.activation = nn.PReLU()
        self.conv1 = nn.Conv2d(3, features, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.conv_mid = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.conv_out = nn.Conv2d(features, 3, 1, 1, 0, bias=True)
        self.norm_features = nn.BatchNorm2d(features, affine=True)

    def _residual_stack(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.activation(self.conv_mid(x))
        x2 = self.activation(self.conv_mid(x1))
        return self.activation(self.conv_out(x + x2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stage1 = self.norm_features(self.activation(self.conv1(x)))
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
