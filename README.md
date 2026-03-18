# DTP: Low-Light Image Super-Resolution

This repository contains the cleaned open-source release of the core DTP model for low-light image super-resolution. The released code keeps the main training and inference pipeline used in our ICME 2026 paper while intentionally excluding pretrained weights, datasets, baselines, web demos, ablation folders, and qualitative result dumps.

## What Is Included

- `dtp/`: core package with the decomposition, enhancement, denoising, and super-resolution modules
- `scripts/train.py`: joint training entry point for the four-stage pipeline
- `scripts/infer.py`: lightweight inference script for single images or folders
- `requirements.txt`: minimal Python dependencies

## What Is Not Included

- pretrained checkpoints
- RELLISUR dataset files
- test outputs, report folders, and visualization dumps
- baselines, ablation studies, and UI code

## Method Overview

DTP follows a four-stage pipeline:

1. `DecomposeNet`: decomposes the low-light input into high-frequency detail and low-frequency illumination components.
2. `EnhanceNet`: enhances the low-frequency branch with an adaptive Naka-Rushton transform and a U-Net style decoder.
3. `DenoiseNet`: denoises the high-frequency branch.
4. `LLSRNet`: fuses the original input, denoised detail, and enhanced illumination for super-resolution reconstruction.

## Repository Layout

```text
.
|-- dtp
|   |-- data
|   |-- models
|   |-- utils
|   `-- losses.py
|-- scripts
|   |-- infer.py
|   `-- train.py
|-- requirements.txt
`-- README.md
```

## Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.10+, PyTorch 2.x, and CUDA-capable GPUs. CPU inference is supported, but training is expected to run on GPU.

## Dataset Layout

The training script expects the RELLISUR dataset in the following structure:

```text
RELLISUR/RELLISUR-Dataset/
|-- Train
|   |-- LLLR
|   `-- NLHR
|       |-- X1
|       |-- X2
|       `-- X4
|-- Val
|   |-- LLLR
|   `-- NLHR
|       |-- X1
|       |-- X2
|       `-- X4
`-- Test
    |-- LLLR
    `-- NLHR
        |-- X1
        |-- X2
        `-- X4
```

The released dataset loader assumes:

- low-light filenames begin with a five-digit image id
- the ground-truth filename is `<first-five-digits><suffix>`
- `X1` is the normal-light low-resolution target
- `X2` or `X4` is the super-resolution target

## Training

```bash
python scripts/train.py \
  --train-lowlight-dir RELLISUR/RELLISUR-Dataset/Train/LLLR \
  --train-gt-dir RELLISUR/RELLISUR-Dataset/Train/NLHR/X2 \
  --train-low-gt-dir RELLISUR/RELLISUR-Dataset/Train/NLHR/X1 \
  --val-lowlight-dir RELLISUR/RELLISUR-Dataset/Val/LLLR \
  --val-gt-dir RELLISUR/RELLISUR-Dataset/Val/NLHR/X2 \
  --val-low-gt-dir RELLISUR/RELLISUR-Dataset/Val/NLHR/X1 \
  --scale 2 \
  --epochs 200 \
  --batch-size 2 \
  --output-dir checkpoints/dtp_x2
```

Key points:

- the checkpoint format remains compatible with the original four-key release format:
  - `La_net`
  - `DES_net`
  - `decom_net`
  - `sr_net`
- validation is optional
- the training loss follows the original weighted multi-branch objective

## Inference

Single image:

```bash
python scripts/infer.py \
  --checkpoint path/to/checkpoint.pth \
  --input path/to/image.png \
  --output outputs/
```

Folder inference:

```bash
python scripts/infer.py \
  --checkpoint path/to/checkpoint.pth \
  --input path/to/input_folder \
  --output outputs/ \
  --save-branches
```

Optional intermediate outputs:

- `high_freq`
- `low_freq`
- `enhanced_low`
- `denoised_high`
- final SR result

## Notes For Release

- This repository is a code-only release.
- Pretrained weights and benchmark results are intentionally omitted.
- If you want to publish checkpoints later, the current `DTPModel.load_checkpoint()` method is already compatible with the legacy checkpoint key names.

## Citation

The BibTeX entry will be added after the final publication metadata is available.
