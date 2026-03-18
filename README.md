# DTP: Low-Light Image Super-Resolution

Official PyTorch implementation of **DTP**, a four-stage low-light image super-resolution framework.

> Accepted by **IEEE International Conference on Multimedia and Expo (ICME 2026)**.

This repository is the cleaned public release of our core model code. It keeps the main training and inference pipeline used in the paper while intentionally excluding pretrained weights, datasets, baselines, web demos, ablation folders, and qualitative result dumps.

## Highlights

- Four-stage pipeline: **decomposition -> enhancement -> denoising -> super-resolution**
- Modular PyTorch implementation for training and inference
- Compatible with the original four-branch checkpoint format
- Clean open-source release without private experimental artifacts

## Method Overview

Given a low-light low-resolution input, DTP first separates illumination-dominant and detail-dominant information, then processes the two branches with dedicated enhancement and denoising modules, and finally reconstructs the high-resolution result through frequency-aware fusion.

The released pipeline includes:

1. `DecomposeNet`
   Separates the input into high-frequency detail and low-frequency illumination components with multi-scale Haar wavelet decomposition and frequency attention.
2. `EnhanceNet`
   Enhances the low-frequency branch with an adaptive Naka-Rushton transform and a U-Net style architecture.
3. `DenoiseNet`
   Suppresses noise in the high-frequency branch.
4. `LLSRNet`
   Fuses the original input, denoised detail, and enhanced illumination for final super-resolution reconstruction.

## Repository Structure

```text
.
|-- dtp
|   |-- data
|   |   `-- rellisur.py
|   |-- models
|   |   |-- decomposition.py
|   |   |-- enhancement.py
|   |   |-- denoising.py
|   |   |-- sr.py
|   |   `-- pipeline.py
|   |-- utils
|   `-- losses.py
|-- scripts
|   |-- train.py
|   `-- infer.py
|-- requirements.txt
`-- README.md
```

## Release Scope

This repository only contains the code required to understand, train, and run the model.

Included:

- model definitions
- loss functions
- dataset loader
- training script
- inference script

Not included:

- pretrained checkpoints
- RELLISUR dataset files
- benchmark outputs and visual results
- baselines and ablation code
- web UI and internal tooling

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Recommended environment:

- Python 3.10+
- PyTorch 2.x
- CUDA-capable GPU for training

CPU inference is supported, but training is expected to run on GPU.

## Dataset Preparation

The training script is written for the **RELLISUR** dataset layout:

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

Dataset naming assumptions in the released loader:

- each low-light filename starts with a five-digit image id
- the ground-truth filename is reconstructed as `<first-five-digits><suffix>`
- `X1` is the normal-light low-resolution supervision
- `X2` or `X4` is the super-resolution target

## Training

Example for `x2` super-resolution:

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

Notes:

- validation is optional
- the released training script preserves the original joint optimization strategy
- the decomposition, enhancement, denoising, and SR branches are optimized separately
- checkpoints are saved in a format compatible with the original codebase

Checkpoint keys:

- `La_net`
- `DES_net`
- `decom_net`
- `sr_net`

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

When `--save-branches` is enabled, the script also exports:

- `high_freq`
- `low_freq`
- `enhanced_low`
- `denoised_high`
- final SR output

## Open-Source Notes

- This is a **code-only** release.
- Pretrained weights are intentionally not included in the current public version.
- Test-time visual results and benchmark dumps are also excluded.
- The `DTPModel.load_checkpoint()` interface already supports the legacy checkpoint structure if weights are released later.

## Citation

If you find this repository useful, please cite our ICME 2026 paper. The final BibTeX entry will be added after the official publication metadata is available.

```bibtex
@inproceedings{dtp_icme2026,
  title     = {To appear},
  author    = {To appear},
  booktitle = {IEEE International Conference on Multimedia and Expo (ICME)},
  year      = {2026}
}
```

## Contact

For questions regarding the paper or the release, please open an issue in this repository.
