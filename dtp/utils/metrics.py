import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to_numpy_images(batch: torch.Tensor) -> np.ndarray:
    if batch.dim() != 4:
        raise ValueError("Expected a 4D tensor shaped as (N, C, H, W)")
    batch = batch.detach().cpu().clamp(0, 1)
    return batch.permute(0, 2, 3, 1).numpy().astype(np.float32)


def batch_psnr(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    prediction_np = _to_numpy_images(prediction)
    target_np = _to_numpy_images(target)
    scores = [
        peak_signal_noise_ratio(target_np[index], prediction_np[index], data_range=data_range)
        for index in range(prediction_np.shape[0])
    ]
    return float(np.mean(scores))


def batch_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    win_size: int = 3,
) -> float:
    prediction_np = _to_numpy_images(prediction)
    target_np = _to_numpy_images(target)
    scores = []
    for index in range(prediction_np.shape[0]):
        try:
            score = structural_similarity(
                target_np[index],
                prediction_np[index],
                data_range=data_range,
                win_size=win_size,
                channel_axis=-1,
            )
        except TypeError:
            score = structural_similarity(
                target_np[index],
                prediction_np[index],
                data_range=data_range,
                win_size=win_size,
                multichannel=True,
            )
        scores.append(score)
    return float(np.mean(scores))
