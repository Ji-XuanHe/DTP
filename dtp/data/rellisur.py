import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dtp.utils.image import list_images, read_image


class RellisurDataset(Dataset):
    def __init__(
        self,
        lowlight_dir: str | Path,
        ground_truth_dir: str | Path,
        low_ground_truth_dir: str | Path,
        training: bool = False,
        cutblur_prob: float = 0.0,
    ) -> None:
        self.lowlight_dir = Path(lowlight_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.low_ground_truth_dir = Path(low_ground_truth_dir)
        self.training = training
        self.cutblur_prob = cutblur_prob

        for path in (self.lowlight_dir, self.ground_truth_dir, self.low_ground_truth_dir):
            if not path.exists():
                raise FileNotFoundError(f"Missing dataset directory: {path}")

        self.lowlight_files = sorted(
            list_images(self.lowlight_dir),
            key=lambda path: self._sort_key(path.name),
        )

    @staticmethod
    def _sort_key(name: str) -> tuple[int, str]:
        prefix = name[:5]
        if prefix.isdigit():
            return int(prefix), name
        return 10**9, name

    @staticmethod
    def _ground_truth_name(lowlight_name: str) -> str:
        suffix = Path(lowlight_name).suffix
        return f"{lowlight_name[:5]}{suffix}"

    @staticmethod
    def _cutblur(lowlight: torch.Tensor, low_ground_truth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        upsampled_lowlight = F.interpolate(
            lowlight.unsqueeze(0),
            size=low_ground_truth.shape[1:],
            mode="nearest",
        ).squeeze(0)
        mask = (torch.rand(low_ground_truth.shape[1:], device=low_ground_truth.device) > 0.5).float()
        mask = mask.unsqueeze(0)
        mixed_gt = mask * low_ground_truth + (1.0 - mask) * upsampled_lowlight
        mixed_lowlight = mask * upsampled_lowlight + (1.0 - mask) * low_ground_truth
        return mixed_lowlight, mixed_gt

    @staticmethod
    def _augment(
        lowlight: torch.Tensor,
        ground_truth: torch.Tensor,
        low_ground_truth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            lowlight = torch.flip(lowlight, dims=[2])
            ground_truth = torch.flip(ground_truth, dims=[2])
            low_ground_truth = torch.flip(low_ground_truth, dims=[2])
        if random.random() < 0.5:
            lowlight = torch.flip(lowlight, dims=[1])
            ground_truth = torch.flip(ground_truth, dims=[1])
            low_ground_truth = torch.flip(low_ground_truth, dims=[1])

        rotations = random.randint(0, 3)
        if rotations:
            lowlight = torch.rot90(lowlight, rotations, dims=[1, 2])
            ground_truth = torch.rot90(ground_truth, rotations, dims=[1, 2])
            low_ground_truth = torch.rot90(low_ground_truth, rotations, dims=[1, 2])

        return lowlight, ground_truth, low_ground_truth

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lowlight_path = self.lowlight_files[index]
        gt_name = self._ground_truth_name(lowlight_path.name)
        ground_truth_path = self.ground_truth_dir / gt_name
        low_ground_truth_path = self.low_ground_truth_dir / gt_name

        lowlight = read_image(lowlight_path)
        ground_truth = read_image(ground_truth_path)
        low_ground_truth = read_image(low_ground_truth_path)

        if self.training:
            lowlight, ground_truth, low_ground_truth = self._augment(lowlight, ground_truth, low_ground_truth)
            if self.cutblur_prob > 0 and random.random() < self.cutblur_prob:
                lowlight, low_ground_truth = self._cutblur(lowlight, low_ground_truth)

        return lowlight, ground_truth, low_ground_truth

    def __len__(self) -> int:
        return len(self.lowlight_files)
