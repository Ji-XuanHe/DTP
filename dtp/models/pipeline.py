from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .csr import CrossFrequencySemanticRecomposition
from .fsd import FrequencyStructuralDecoupling
from .sdr import SemanticsSpecificDualPathRepresentation


class DecouplingThenPerceive(nn.Module):
    """Paper name: Decoupling then Perceive (DTP)."""

    def __init__(self, scale: int = 2) -> None:
        super().__init__()
        self.scale = scale
        self.fsd = FrequencyStructuralDecoupling()
        self.sdr = SemanticsSpecificDualPathRepresentation()
        self.csr = CrossFrequencySemanticRecomposition(upscale=scale)

    @property
    def luminance_enhancer(self) -> nn.Module:
        return self.sdr.luminance_enhancer

    @property
    def texture_denoiser(self) -> nn.Module:
        return self.sdr.texture_denoiser

    def forward(self, original_llr: torch.Tensor) -> dict[str, torch.Tensor]:
        luminance_llr, texture_llr = self.fsd(original_llr)
        luminance_hlr, texture_dlr = self.sdr(luminance_llr, texture_llr)
        restored_hsr = self.csr(original_llr, texture_dlr, luminance_hlr)
        return {
            "luminance_llr": luminance_llr,
            "texture_llr": texture_llr,
            "luminance_hlr": luminance_hlr,
            "texture_dlr": texture_dlr,
            "restored_hsr": restored_hsr,
            # Legacy aliases retained for compatibility with the initial public release.
            "low_freq": luminance_llr,
            "high_freq": texture_llr,
            "enhanced_low": luminance_hlr,
            "denoised_high": texture_dlr,
            "sr": restored_hsr,
        }

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "fsd": self.fsd.state_dict(),
            "sdr": self.sdr.state_dict(),
            "csr": self.csr.state_dict(),
        }

    def load_checkpoint(self, checkpoint: str | Path | dict[str, Any], strict: bool = True) -> dict[str, Any]:
        if isinstance(checkpoint, (str, Path)):
            checkpoint = torch.load(checkpoint, map_location="cpu")

        if {"fsd", "sdr", "csr"}.issubset(checkpoint.keys()):
            self.fsd.load_state_dict(checkpoint["fsd"], strict=strict)
            self.sdr.load_state_dict(checkpoint["sdr"], strict=strict)
            self.csr.load_state_dict(checkpoint["csr"], strict=strict)
            return checkpoint

        # Legacy checkpoint format from the original internal codebase.
        self.luminance_enhancer.load_state_dict(checkpoint["La_net"], strict=strict)
        self.texture_denoiser.load_state_dict(checkpoint["DES_net"], strict=strict)
        self.fsd.load_state_dict(checkpoint["decom_net"], strict=strict)
        self.csr.load_state_dict(checkpoint["sr_net"], strict=strict)
        return checkpoint


DTPModel = DecouplingThenPerceive


def build_dtp_model(scale: int = 2) -> DecouplingThenPerceive:
    return DecouplingThenPerceive(scale=scale)


def build_dtp_framework(scale: int = 2) -> DecouplingThenPerceive:
    return DecouplingThenPerceive(scale=scale)
