from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .decomposition import DecomposeNet
from .denoising import DenoiseNet
from .enhancement import EnhanceNet
from .sr import LLSRNet


class DTPModel(nn.Module):
    def __init__(self, scale: int = 2) -> None:
        super().__init__()
        self.scale = scale
        self.decomposer = DecomposeNet()
        self.enhancer = EnhanceNet()
        self.denoiser = DenoiseNet()
        self.super_resolver = LLSRNet(upscale=scale)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        high_freq, low_freq = self.decomposer(x)
        enhanced_low = self.enhancer(low_freq)
        denoised_high = self.denoiser(high_freq)
        sr = self.super_resolver(x, denoised_high, enhanced_low)
        return {
            "high_freq": high_freq,
            "low_freq": low_freq,
            "enhanced_low": enhanced_low,
            "denoised_high": denoised_high,
            "sr": sr,
        }

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "La_net": self.enhancer.state_dict(),
            "DES_net": self.denoiser.state_dict(),
            "decom_net": self.decomposer.state_dict(),
            "sr_net": self.super_resolver.state_dict(),
        }

    def load_checkpoint(self, checkpoint: str | Path | dict[str, Any], strict: bool = True) -> dict[str, Any]:
        if isinstance(checkpoint, (str, Path)):
            checkpoint = torch.load(checkpoint, map_location="cpu")

        self.enhancer.load_state_dict(checkpoint["La_net"], strict=strict)
        self.denoiser.load_state_dict(checkpoint["DES_net"], strict=strict)
        self.decomposer.load_state_dict(checkpoint["decom_net"], strict=strict)
        self.super_resolver.load_state_dict(checkpoint["sr_net"], strict=strict)
        return checkpoint


def build_dtp_model(scale: int = 2) -> DTPModel:
    return DTPModel(scale=scale)
