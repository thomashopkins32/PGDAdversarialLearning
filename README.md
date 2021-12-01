# PGD Adversarial Learning
Projected Gradient Descent for Adversarial Learning

Based on the paper https://arxiv.org/pdf/1706.06083.pdf

## Python Environmenet Setup
```bash
git clone https://github.com/thomashopkins32/PGDAdversarialLearning.git
cd PGDAdversarialLearning
conda create -n pgd-env python=3.7
conda activate pgd-env
pip install -r requirements.txt
```

## Data Sets
CIFAR-10 and CIFAR-100 are available from `torchvision`.
Alternatively, set `"dataset"` to either `"cifar-10"` or `"cifar-100"` in `config.json`.

## Configuration
Change `config.json` then run `train.py` to train your model.

## Notebooks
Run some of the other experiments using `jupyter lab`. Some of these depend on `config.json`.
