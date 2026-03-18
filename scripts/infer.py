import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm

from dtp.models import build_dtp_model
from dtp.utils import ensure_dir, list_images, read_image, write_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DTP inference on an image or a folder.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-branches", action="store_true")
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_output_path(input_root: Path, input_path: Path, output_root: Path) -> Path:
    if input_root.is_file():
        if output_root.suffix:
            return output_root
        return output_root / f"{input_path.stem}_dtp{input_path.suffix}"
    relative_name = input_path.relative_to(input_root).name
    return output_root / relative_name


def save_branches(outputs: dict[str, torch.Tensor], output_path: Path) -> None:
    branch_dir = output_path.parent / f"{output_path.stem}_branches"
    ensure_dir(branch_dir)
    write_image(outputs["high_freq"], branch_dir / "high_freq.png")
    write_image(outputs["low_freq"], branch_dir / "low_freq.png")
    write_image(outputs["enhanced_low"], branch_dir / "enhanced_low.png")
    write_image(outputs["denoised_high"], branch_dir / "denoised_high.png")


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    model = build_dtp_model(scale=args.scale).to(device)
    model.load_checkpoint(args.checkpoint)
    model.eval()

    input_root = Path(args.input)
    output_root = Path(args.output)
    if input_root.is_dir():
        ensure_dir(output_root)
    elif not output_root.suffix:
        ensure_dir(output_root)

    input_files = list_images(input_root)
    with torch.no_grad():
        for image_path in tqdm(input_files, desc="Infer"):
            image = read_image(image_path).unsqueeze(0).to(device)
            outputs = model(image)
            output_path = resolve_output_path(input_root, image_path, output_root)
            write_image(outputs["sr"], output_path)
            if args.save_branches:
                save_branches(outputs, output_path)


if __name__ == "__main__":
    main()
