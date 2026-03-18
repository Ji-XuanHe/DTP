import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dtp.data import RellisurDataset
from dtp.losses import (
    EnhanceLoss,
    FrequencyConsistencyLoss,
    LLSRLoss,
    LowFrequencyTVLoss,
    TotalVariationLoss,
)
from dtp.models import build_dtp_model
from dtp.utils import batch_psnr, batch_ssim, ensure_dir, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DTP with FSD, SDR, and CSR modules.")
    parser.add_argument("--train-lowlight-dir", required=True, type=str)
    parser.add_argument("--train-gt-dir", required=True, type=str)
    parser.add_argument("--train-low-gt-dir", required=True, type=str)
    parser.add_argument("--val-lowlight-dir", type=str)
    parser.add_argument("--val-gt-dir", type=str)
    parser.add_argument("--val-low-gt-dir", type=str)
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="checkpoints/dtp")
    parser.add_argument("--resume", type=str)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutblur-prob", type=float, default=0.5)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--sdr-luminance-lr", "--enhance-lr", dest="sdr_luminance_lr", type=float, default=2e-4)
    parser.add_argument("--fsd-lr", "--decompose-lr", dest="fsd_lr", type=float, default=2e-4)
    parser.add_argument("--sdr-texture-lr", "--denoise-lr", dest="sdr_texture_lr", type=float, default=1e-4)
    parser.add_argument("--csr-lr", "--sr-lr", dest="csr_lr", type=float, default=1e-4)
    parser.add_argument("--scheduler-start-epoch", type=int, default=50)
    parser.add_argument("--sdr-luminance-perceptual-weight", "--enhance-perceptual-weight", dest="sdr_luminance_perceptual_weight", type=float, default=0.1)
    parser.add_argument("--csr-vgg-weight", "--sr-vgg-weight", dest="csr_vgg_weight", type=float, default=0.1)
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_optimizers(model: torch.nn.Module, args: argparse.Namespace) -> dict[str, torch.optim.Optimizer]:
    return {
        "sdr_luminance": torch.optim.AdamW(
            model.luminance_enhancer.parameters(),
            lr=args.sdr_luminance_lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        ),
        "sdr_texture": torch.optim.Adam(
            model.texture_denoiser.parameters(),
            lr=args.sdr_texture_lr,
            weight_decay=args.weight_decay,
        ),
        "fsd": torch.optim.Adam(
            model.fsd.parameters(),
            lr=args.fsd_lr,
            weight_decay=args.weight_decay,
        ),
        "csr": torch.optim.Adam(
            model.csr.parameters(),
            lr=args.csr_lr,
            weight_decay=args.weight_decay,
        ),
    }


def create_schedulers(
    optimizers: dict[str, torch.optim.Optimizer],
    epochs: int,
) -> dict[str, torch.optim.lr_scheduler.LRScheduler]:
    return {
        name: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
        for name, optimizer in optimizers.items()
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "config": vars(args),
        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "schedulers": {name: scheduler.state_dict() for name, scheduler in schedulers.items()},
    }
    checkpoint.update(model.checkpoint_state())
    torch.save(checkpoint, output_path)


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    psnr_scores = []
    ssim_scores = []
    for lowlight, ground_truth, _ in tqdm(loader, desc="Validate", leave=False):
        lowlight = lowlight.to(device)
        ground_truth = ground_truth.to(device)
        prediction = model(lowlight)["restored_hsr"]
        psnr_scores.append(batch_psnr(prediction, ground_truth))
        ssim_scores.append(batch_ssim(prediction, ground_truth))
    return float(np.mean(psnr_scores)), float(np.mean(ssim_scores))


def remap_training_state(name: str) -> str:
    legacy_mapping = {
        "enhancer": "sdr_luminance",
        "denoiser": "sdr_texture",
        "decomposer": "fsd",
        "sr": "csr",
    }
    return legacy_mapping.get(name, name)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    logger = setup_logger("train", output_dir / "logs")
    device = select_device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    logger.info("Using device: %s", device)

    train_dataset = RellisurDataset(
        args.train_lowlight_dir,
        args.train_gt_dir,
        args.train_low_gt_dir,
        training=True,
        cutblur_prob=args.cutblur_prob,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    val_loader = None
    if args.val_lowlight_dir and args.val_gt_dir and args.val_low_gt_dir:
        val_dataset = RellisurDataset(
            args.val_lowlight_dir,
            args.val_gt_dir,
            args.val_low_gt_dir,
            training=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    model = build_dtp_model(scale=args.scale).to(device)
    optimizers = create_optimizers(model, args)
    schedulers = create_schedulers(optimizers, args.epochs)

    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    tv_loss = TotalVariationLoss()
    low_tv_loss = LowFrequencyTVLoss()
    frequency_loss = FrequencyConsistencyLoss().to(device)
    sdr_luminance_loss = EnhanceLoss(perception_weight=args.sdr_luminance_perceptual_weight).to(device)
    csr_loss = LLSRLoss(vgg_loss_weight=args.csr_vgg_weight).to(device)

    start_epoch = 0
    best_psnr = float("-inf")

    if args.resume:
        logger.info("Loading checkpoint from %s", args.resume)
        checkpoint = model.load_checkpoint(args.resume)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        for name, optimizer_state in checkpoint.get("optimizers", {}).items():
            mapped_name = remap_training_state(name)
            if mapped_name in optimizers:
                optimizers[mapped_name].load_state_dict(optimizer_state)
        for name, scheduler_state in checkpoint.get("schedulers", {}).items():
            mapped_name = remap_training_state(name)
            if mapped_name in schedulers:
                schedulers[mapped_name].load_state_dict(scheduler_state)

    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = []
        epoch_psnr = []
        epoch_ssim = []

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step, (lowlight, ground_truth, low_ground_truth) in enumerate(progress, start=1):
            lowlight = lowlight.to(device)
            ground_truth = ground_truth.to(device)
            low_ground_truth = low_ground_truth.to(device)

            luminance_llr, texture_llr = model.fsd(lowlight)
            gt_luminance, gt_texture = model.fsd(low_ground_truth)

            luminance_hlr, texture_dlr = model.sdr(luminance_llr, texture_llr.detach())

            reconstruction_loss = mse_loss(gt_texture + gt_luminance, low_ground_truth) + mse_loss(
                texture_llr + luminance_llr,
                lowlight,
            )
            fsd_loss = (
                100.0 * reconstruction_loss
                + 2.0 * (mse_loss(luminance_llr, lowlight) + mse_loss(gt_luminance, low_ground_truth))
                + low_tv_loss(luminance_llr)
                + tv_loss(gt_luminance)
                + 0.1 * frequency_loss(texture_llr + luminance_llr, texture_llr, luminance_llr)
            )
            sdr_luminance_branch_loss = sdr_luminance_loss(luminance_hlr, gt_luminance.detach(), luminance_llr)
            sdr_texture_branch_loss = l1_loss(texture_dlr, gt_texture.detach())

            prediction = model.csr(lowlight, texture_dlr, luminance_hlr)
            csr_branch_loss = csr_loss(prediction, ground_truth)

            total_loss = fsd_loss + 10.0 * sdr_luminance_branch_loss + sdr_texture_branch_loss + 5.0 * csr_branch_loss
            (total_loss / args.accumulation_steps).backward()

            if step % args.accumulation_steps == 0 or step == len(train_loader):
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                for optimizer in optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            epoch_losses.append(total_loss.item())
            epoch_psnr.append(batch_psnr(prediction.detach(), ground_truth))
            epoch_ssim.append(batch_ssim(prediction.detach(), ground_truth))

            if step % args.print_every == 0 or step == len(train_loader):
                progress.set_postfix(
                    loss=f"{np.mean(epoch_losses):.4f}",
                    psnr=f"{np.mean(epoch_psnr):.4f}",
                    ssim=f"{np.mean(epoch_ssim):.4f}",
                )

        schedulers["fsd"].step()
        if epoch >= args.scheduler_start_epoch:
            schedulers["sdr_luminance"].step()
            schedulers["sdr_texture"].step()
            schedulers["csr"].step()

        logger.info(
            "Epoch %d | loss %.6f | train PSNR %.4f | train SSIM %.4f",
            epoch,
            float(np.mean(epoch_losses)),
            float(np.mean(epoch_psnr)),
            float(np.mean(epoch_ssim)),
        )

        if val_loader is not None:
            val_psnr, val_ssim = validate(model, val_loader, device)
            logger.info("Epoch %d | val PSNR %.4f | val SSIM %.4f", epoch, val_psnr, val_ssim)
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(model, optimizers, schedulers, epoch, output_dir / "best.pth", args)

        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizers, schedulers, epoch, output_dir / f"epoch_{epoch:03d}.pth", args)


if __name__ == "__main__":
    main()
