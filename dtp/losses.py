import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class TotalVariationLoss(nn.Module):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        height = x.size(2)
        width = x.size(3)
        count_h = (height - 1) * width
        count_w = height * (width - 1)
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, : height - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, : width - 1], 2).sum()
        return self.weight * (h_tv / count_h + w_tv / count_w) / batch_size


class LowFrequencyTVLoss(TotalVariationLoss):
    def __init__(self) -> None:
        super().__init__(weight=5.0)


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize: bool = True) -> None:
        super().__init__()
        blocks = []
        features = self._build_vgg16_features()
        slices = ((0, 4), (4, 9), (9, 16), (16, 23))
        for start, end in slices:
            block = features[start:end].eval()
            for parameter in block.parameters():
                parameter.requires_grad = False
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @staticmethod
    def _build_vgg16_features() -> nn.Sequential:
        try:
            weights = models.VGG16_Weights.IMAGENET1K_V1
            return models.vgg16(weights=weights).features
        except (AttributeError, TypeError):
            return models.vgg16(pretrained=True).features

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        feature_layers: tuple[int, ...] = (0, 1, 2, 3),
        style_layers: tuple[int, ...] = (),
    ) -> torch.Tensor:
        if prediction.shape[1] != 3:
            prediction = prediction.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        prediction = (prediction - self.mean) / self.std
        target = (target - self.mean) / self.std

        if self.resize:
            prediction = F.interpolate(prediction, size=(224, 224), mode="bilinear", align_corners=False)
            target = F.interpolate(target, size=(224, 224), mode="bilinear", align_corners=False)

        loss = prediction.new_tensor(1e-8)
        x = prediction
        y = target
        for index, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if index in feature_layers:
                loss = loss + F.l1_loss(x, y)
            if index in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss = loss + F.l1_loss(gram_x, gram_y)
        return loss


class IlluminationAwareLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, low_input: torch.Tensor) -> torch.Tensor:
        prediction_illumination = torch.mean(prediction, dim=1, keepdim=True)
        target_illumination = torch.mean(target, dim=1, keepdim=True)
        input_illumination = torch.mean(low_input, dim=1, keepdim=True)

        target_ratio = target_illumination / (input_illumination + 1e-6)
        prediction_ratio = prediction_illumination / (input_illumination + 1e-6)
        return self.l1(prediction_ratio, target_ratio)


class FrequencyConsistencyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(
        self,
        prediction: torch.Tensor,
        texture_llr: torch.Tensor,
        luminance_llr: torch.Tensor,
    ) -> torch.Tensor:
        prediction_freq = torch.fft.fft2(prediction)
        texture_spectrum = torch.fft.fft2(texture_llr)
        luminance_spectrum = torch.fft.fft2(luminance_llr)
        return self.l1(
            torch.abs(prediction_freq),
            torch.abs(texture_spectrum) + torch.abs(luminance_spectrum),
        )


class EnhanceLoss(nn.Module):
    def __init__(
        self,
        l1_weight: float = 1.0,
        l2_weight: float = 0.5,
        perception_weight: float = 0.1,
        grad_weight: float = 0.1,
        illum_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.vgg = VGGPerceptualLoss() if perception_weight > 0 else None
        self.illumination = IlluminationAwareLoss()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.perception_weight = perception_weight
        self.grad_weight = grad_weight
        self.illum_weight = illum_weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, low_input: torch.Tensor) -> torch.Tensor:
        l1_loss = self.l1(prediction, target)
        l2_loss = self.l2(prediction, target)
        perception_loss = prediction.new_tensor(0.0)
        if self.vgg is not None:
            perception_loss = self.vgg(prediction, target)

        pred_grad_x = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
        pred_grad_y = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        grad_loss = self.l1(pred_grad_x, target_grad_x) + self.l1(pred_grad_y, target_grad_y)
        illumination_loss = self.illumination(prediction, target, low_input)

        return (
            self.l1_weight * l1_loss
            + self.l2_weight * l2_loss
            + self.perception_weight * perception_loss
            + self.grad_weight * grad_loss
            + self.illum_weight * illumination_loss
        )


class LLSRLoss(nn.Module):
    def __init__(
        self,
        l1_weight: float = 1.0,
        mse_weight: float = 1.0,
        vgg_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.vgg_loss = VGGPerceptualLoss() if vgg_loss_weight > 0 else None
        self.vgg_loss_weight = vgg_loss_weight
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = torch.clamp(prediction, 0, 1)
        target = torch.clamp(target, 0, 1)

        mse_loss = F.mse_loss(prediction, target)
        l1_loss = self.l1_loss(prediction, target)
        vgg_loss = prediction.new_tensor(0.0)
        if self.vgg_loss is not None:
            vgg_loss = self.vgg_loss(prediction, target)

        return (
            self.mse_weight * mse_loss
            + self.l1_weight * l1_loss
            + self.vgg_loss_weight * vgg_loss
        )
