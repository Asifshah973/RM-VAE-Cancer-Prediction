# RM-VAE + DNN for Cancer Classification

Reproducible reference implementation of the **Riemannian Manifold Variational Autoencoder (RM-VAE) + Deep Neural Network (DNN)** framework described in the accompanying manuscript.

## What this repository implements

- Leakage-free stratified 10-fold cross-validation.
- Training-only z-score standardization.
- Breast-cancer training-only Gaussian augmentation when N < 150, sigma = 0.01.
- Two-stage training: RM-VAE representation learning, followed by a frozen-encoder DNN classifier.
- Standard Gaussian VAE prior; decoder-induced pullback Riemannian metric.
- Random-projection Jacobian estimator with r = 128.
- Geometric smoothness regularization using finite differences with delta = 1e-3.
- Posterior-mean latent embeddings at inference.
- Nested cross-validation support for beta, lambda and latent dimension k.
- Classical baselines, XGBoost, GA/PSO feature-selection, and optional VGG-16/ResNet-50 image baselines.
- Accuracy, precision, recall, F1, MSE, RMSE, ROC-AUC, confidence intervals and Wilcoxon/Holm testing.
- Fixed manuscript seed 1234, with optional multi-seed robustness experiments.

## Important reproducibility note

The manuscript specifies the mathematical architecture and training protocol, but it does **not** give numerical hidden-layer widths for the encoder, decoder, or DNN. Therefore this repository makes those widths explicit in YAML configuration files rather than silently inventing them as if they were manuscript-reported values. If the authors have the original implementation with exact widths, replace the configurable widths in `configs/*.yaml` before claiming exact code-level reproduction.

## Expected datasets

The manuscript describes three gene-expression datasets:

| Dataset | Samples | Genes | Latent k |
|---|---:|---:|---:|
| Breast | 97 | 24,481 | 64 |
| Lung | 181 | 12,533 | 64 |
| Ovarian | 253 | 15,154 | 32 |

The repository does not redistribute third-party datasets. Put each dataset in `data/` after checking its licence/terms.

### Accepted CSV format

A simple format is recommended:

```text
sample_id,label,gene_1,gene_2,...,gene_d
S001,0,2.1,3.4,...
S002,1,1.9,3.7,...
```

The loader also supports CSV files where `label` is the first/last column and a sample-id column is present.

## Installation

### Option A: pip

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate rmvae-cancer
```

TensorFlow will use a GPU automatically when a compatible CUDA installation is available.

## Step 1 — Put your data in the repository

Example:

```text
data/
├── breast.csv
├── lung.csv
└── ovarian.csv
```

Do not commit private, restricted, patient-identifiable, or licence-protected data.

## Step 2 — Generate fixed folds

```bash
python scripts/generate_folds.py --dataset breast --data data/breast.csv --config configs/breast.yaml
python scripts/generate_folds.py --dataset lung --data data/lung.csv --config configs/lung.yaml
python scripts/generate_folds.py --dataset ovarian --data data/ovarian.csv --config configs/ovarian.yaml
```

The fold files are saved under `folds/<dataset>/` and can be committed if the data licence permits.

## Step 3 — Run the proposed RM-VAE + DNN pipeline

```bash
python scripts/run_rmvae.py --dataset breast --data data/breast.csv --config configs/breast.yaml
python scripts/run_rmvae.py --dataset lung --data data/lung.csv --config configs/lung.yaml
python scripts/run_rmvae.py --dataset ovarian --data data/ovarian.csv --config configs/ovarian.yaml
```

Outputs are written to `results/<dataset>/`.

## Step 4 — Run baselines

```bash
python scripts/run_baselines.py --dataset breast --data data/breast.csv --config configs/breast.yaml
```

Available tabular baselines include SVC, decision tree, logistic regression, random forest, XGBoost (when installed), and K-NN. GA/PSO feature-selection experiments are provided separately because their computational cost is substantially higher.

## Step 5 — GA/PSO feature-selection baselines

These are computationally expensive on 10,000+ genes. Run only when required:

```bash
python scripts/run_feature_selection.py --dataset breast --data data/breast.csv --config configs/breast.yaml
```

## Step 6 — VGG-16 / ResNet-50 image baselines

These models require the original scanner-generated microarray spot-array images as an NPY tensor `[N,H,W,C]` plus a binary-label NPY file. They are initialized randomly, not with ImageNet weights:

```bash
python scripts/run_image_baselines.py --images data/breast_images.npy --labels data/breast_image_labels.npy --model vgg16
python scripts/run_image_baselines.py --images data/breast_images.npy --labels data/breast_image_labels.npy --model resnet50
```

## Step 7 — Nested CV

```bash
python scripts/run_nested_cv.py --dataset breast --data data/breast.csv --config configs/breast.yaml
```

The implementation keeps the outer test fold untouched while beta, lambda and k are selected inside the outer training data.

## Step 6 — Statistical comparison

After fold-level results are available:

```bash
python scripts/statistical_tests.py --input results/breast/fold_results.csv --output results/breast/statistics.csv
```

The Wilcoxon analysis uses **10 paired outer-fold observations per comparison**, matching the manuscript's clarified statistical description.

## Repository structure

```text
RM-VAE-Cancer-Prediction/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
├── data/README.md
├── configs/
│   ├── default.yaml
│   ├── breast.yaml
│   ├── lung.yaml
│   └── ovarian.yaml
├── src/
│   ├── models/
│   ├── geometry/
│   ├── preprocessing/
│   ├── training/
│   ├── baselines/
│   ├── evaluation/
│   └── utils/
├── scripts/
├── folds/
├── results/
├── checkpoints/
├── docs/
└── notebooks/
```

## Manuscript-aligned settings

- Seed: 1234
- CV: stratified 10-fold
- RM-VAE: 100 epochs, Adam, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, batch=32
- DNN: 100 epochs, same Adam settings, early stopping patience=10
- beta=0.5, lambda=0.1
- k: Breast=64, Lung=64, Ovarian=32
- Pullback metric: G(z)=J(z)^T J(z)+epsilon I, epsilon=1e-3
- Random projection dimension: r=128
- Finite-difference delta=1e-3
- Breast augmentation: sigma=0.01, training only, if N<150
- VGG-16 / ResNet-50: random initialization, not ImageNet pretrained, when image data are available

## Citation

Add the final published citation/DOI here after publication.

## Disclaimer

This repository is research software. It is not a clinical diagnostic system and must not be used for patient-care decisions without appropriate independent validation, regulatory review, and clinical governance.
