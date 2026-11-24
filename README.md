# How Robust Are Deep Learning Models? Corruptions, Adversarial Attacks, Robust Training and Abstention on CIFAR-10

M.Tech AI project. Benchmarks CIFAR-10 classifiers (simple CNN, ResNet-18, EfficientNet-B0) against image corruptions and
adversarial perturbations, compares robust-training strategies (including an adaptive corruption-aware scheduler),
and studies uncertainty-aware abstention.

> **Status: work in progress (Day 1 of 10).** This README grows with the code; see `docs/COMMIT_PLAN.md`.

## Directory structure
```
robust-cifar10/
├── rb/                  # library code (data, models, corruptions, training, evaluation)
├── scripts/             # prepare_data.py, export_commit_log.py (more added daily)
├── tests/               # smoke / unit tests
├── data/                # NOT committed; created by scripts/prepare_data.py
├── checkpoints/         # trained model weights (or a link to them, see below)
├── results/             # metrics CSVs, logs, figures
├── report/              # CVPR-format LaTeX report
├── docs/                # COMMIT_PLAN.md, COMMIT_LOG.md
├── requirements.txt
└── environment.yml
```

## Setup
```powershell
# option A: conda
conda env create -f environment.yml && conda activate robust-cifar10
# option B: pip (install a CUDA build of torch first if you have a GPU)
python -m venv .venv ; .\.venv\Scripts\activate ; pip install -r requirements.txt
```

## Data (not stored in this repo)
```powershell
python scripts/prepare_data.py     # downloads CIFAR-10 (HF: uoft-cs/cifar10) -> data/cifar_{train,test}.npz (~170 MB)
python tests/test_data.py          # sanity checks
```
Split: 45,000 train / 5,000 validation (fixed seed 0) from the official training set; the official 10,000-image test set is used only for evaluation.

## Progress log
| Day | Added |
|---|---|
| 1 | Skeleton, environment files, data pipeline |
