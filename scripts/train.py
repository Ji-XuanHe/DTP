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
    parser = argparse.ArgumentParser(description="Train DTP for low-light super-resolution.")
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
    parser.add_argument("--enhance-lr", type=float, default=2e-4)
    parser.add_argument("--decompose-lr", type=float, default=2e-4)
    parser.add_argument("--denoise-lr", type=float, default=1e-4)
    parser.add_argument("--sr-lr", type=float, default=1e-4)
    parser.add_argument("--scheduler-start-epoch", type=int, default=50)
    parser.add_argument("--enhance-perceptual-weight", type=float, default=0.1)
    parser.add_argument("--sr-vgg-weight", type=float, default=0.1)
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
        "enhancer": torch.optim.AdamW(
            model.enhancer.parameters(),
            lr=args.enhance_lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        ),
        "denoiser": torch.optim.Adam(
            model.denoiser.parameters(),
            lr=args.denoise_lr,
            weight_decay=args.weight_decay,
        ),
        "decomposer": torch.optim.Adam(
            model.decomposer.parameters(),
            lr=args.decompose_lr,
            weight_decay=args.weight_decay,
        ),
        "sr": torch.optim.Adam(
            model.super_resolver.parameters(),
            lr=args.sr_lr,
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
        prediction = model(lowlight)["sr"]
        psnr_scores.append(batch_psnr(prediction, ground_truth))
        ssim_scores.append(batch_ssim(prediction, ground_truth))
    return float(np.mean(psnr_scores)), float(np.mean(ssim_scores))


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
    enhance_loss = EnhanceLoss(perception_weight=args.enhance_perceptual_weight).to(device)
    sr_loss = LLSRLoss(vgg_loss_weight=args.sr_vgg_weight).to(device)

    start_epoch = 0
    best_psnr = float("-inf")

    if args.resume:
        logger.info("Loading checkpoint from %s", args.resume)
        checkpoint = model.load_checkpoint(args.resume)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        for name, optimizer_state in checkpoint.get("optimizers", {}).items():
            if name in optimizers:
                optimizers[name].load_state_dict(optimizer_state)
        for name, scheduler_state in checkpoint.get("schedulers", {}).items():
            if name in schedulers:
                schedulers[name].load_state_dict(scheduler_state)

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

            lowlight_high, lowlight_low = model.decomposer(lowlight)
            gt_high, gt_low = model.decomposer(low_ground_truth)

            enhanced_low = model.enhancer(lowlight_low)
            denoised_high = model.denoiser(lowlight_high.detach())

            reconstruction_loss = mse_loss(gt_high + gt_low, low_ground_truth) + mse_loss(
                lowlight_high + lowlight_low,
                lowlight,
            )
            decomposition_loss = (
                100.0 * reconstruction_loss
                + 2.0 * (mse_loss(lowlight_low, lowlight) + mse_loss(gt_low, low_ground_truth))
                + low_tv_loss(lowlight_low)
                + tv_loss(gt_low)
                + 0.1 * frequency_loss(lowlight_high + lowlight_low, lowlight_high, lowlight_low)
            )
            branch_enhance_loss = enhance_loss(enhanced_low, gt_low.detach(), lowlight_low)
            branch_denoise_loss = l1_loss(denoised_high, gt_high.detach())

            prediction = model.super_resolver(lowlight, denoised_high, enhanced_low)
            branch_sr_loss = sr_loss(prediction, ground_truth)

            total_loss = decomposition_loss + 10.0 * branch_enhance_loss + branch_denoise_loss + 5.0 * branch_sr_loss
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

        schedulers["decomposer"].step()
        if epoch >= args.scheduler_start_epoch:
            schedulers["enhancer"].step()
            schedulers["denoiser"].step()
            schedulers["sr"].step()

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
