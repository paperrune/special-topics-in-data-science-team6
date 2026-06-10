# Task-Specific OOD Detection Experiments

This repository contains the runnable code used for the Team 6 experiments in
Special Topics in Data Science. The project studies task-specific
out-of-distribution detection where the ID dataset is a downstream task and the
OOD evaluation set is ImageNet-1K validation.

## Repository Structure

```text
baseline-validation/
  main.py                 # ResNet-18 baseline experiment entry point
  script.sh               # Batch script for all baseline runs
  methods/                # MSP, Energy, Mahalanobis, ViM, VOS, NPOS

cls-rec/
  src/
    CIFAR-10/
    CIFAR-100/
    CUB_200_2011/
    Oxford_IIIT_Pets/
    Stanford_Cars/
```

`baseline-validation` runs the standard OOD baselines. `cls-rec` contains the
latent-boundary experiments with classification/reconstruction targets and
scratch/pretrained variants.

## Requirements

The code was run with Python 3 and PyTorch. Main packages:

```text
torch
torchvision
numpy
scikit-learn
```

CUDA is expected for the training scripts.

## Data

For baseline experiments, ImageNet validation is used as the fixed OOD set.
The default ImageNet path is:

```text
/home/yang/data/imagenet1k/val
```

The baseline code expects the following ID datasets:

```text
baseline-validation/data              # CIFAR-10 and CIFAR-100
/home/yang/data/cub200
/home/yang/data/stanfordcars
/home/yang/data/oxfordpets
```

For `cls-rec`, each training script uses paths written inside the script. In
particular, CIFAR datasets are downloaded under the local `../data` path, and
ImageNet validation is expected at:

```text
cls-rec/src/ImageNet/data/validation
```

## Running Baselines

Run one baseline experiment:

```bash
cd baseline-validation
python3 main.py \
  --dataset cifar10 \
  --method msp \
  --epochs 10 \
  --pretrained \
  --device cuda:0 \
  --output-dir saved/cifar10/msp/pretrained
```

Available datasets:

```text
cifar10, cifar100, cub200, oxfordpets, stanfordcars
```

Available methods:

```text
msp, energy, mahalanobis, vim, vos, npos
```

To run the prepared baseline sweep:

```bash
cd baseline-validation
bash script.sh
```

Each run writes checkpoints and metrics to:

```text
saved/{dataset}/{method}/{pretrained or no-pretrained}/
```

If the output directory already exists, the run is treated as completed and is
skipped.

## Running Latent-Boundary Experiments

Each proposed-method variant is run from its own folder with `torchrun`.
Example:

```bash
cd cls-rec/src/CIFAR-10/OOD-classification
torchrun --nproc_per_node=1 training.py
```

Pretrained and reconstruction variants are separate directories, for example:

```text
OOD-classification
OOD-classification+pretrained
OOD-reconstruction
OOD-reconstruction+pretrained
```

During training, the scripts periodically print AUROC, AUPR-In, and FPR@95 and
save checkpoints in numbered iteration folders such as `10000/`,
`20000/`, etc.

## Notes

- Baseline models use ResNet-18.
- Baseline pretrained runs use torchvision ImageNet-1K weights.
- Epoch 0 checkpoints are saved before training for baseline experiments.
- The OOD evaluation set is ImageNet-1K validation for all experiments.
