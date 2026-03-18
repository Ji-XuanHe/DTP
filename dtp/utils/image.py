from pathlib import Path

import cv2
import numpy as np
import torch

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted([item for item in path.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS])


def read_image(path: str | Path) -> torch.Tensor:
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()


def write_image(image: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    if image.dim() == 4:
        image = image.squeeze(0)

    image = image.detach().cpu().clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255.0).round().astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image)
