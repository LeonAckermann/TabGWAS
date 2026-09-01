# TabGWAS

Code for the ICBINB @ NeurIPS 2026 workshop submission
"When Non-linearity Doesn't Pay Off on GWAS Data."
Author information omitted for double-blind review.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Also requires [PLINK v2](https://www.cog-genomics.org/plink/2.0/) on `PATH` for the data-curation step.

## Reproducing the paper

### 1. Data

Build the SNP × phenotype matrix from raw GWAS summary statistics (one-time step; skip if the matrix already exists):

```bash
python main.py --config experiments/data/construct_gwas_mri.yaml
```

### 2. Experiments

Each config runs nested cross-validation + hyperparameter optimization for one model, illness, and p-value threshold sweep, and writes a result JSON to `results/<experiment_name>/`.

**Main results** (Table `tab:friedman_results`, Figure `fig:friedman_test`) — one config per model in [`experiments/main/`](experiments/main):

```bash
python main.py --config experiments/main/ridge_hpo_regression.yaml
# ... repeat for every file in experiments/main/
```

**Residual ablation** (Table `tab:residual_results`, appendix) — fits each non-linear model (DNN, TabPFN, XGBoost) on the out-of-fold residual of a fixed-α Ridge regression, using [`experiments/residual/`](experiments/residual):

```bash
python main.py --config experiments/residual/ridge_hpo_regression.yaml       # residual baseline
python main.py --config experiments/residual/residual_dnn_hpo_regression_01.yaml
# ... repeat for every file in experiments/residual/
```

### 3. Analysis

Turns the result JSONs from step 2 into the paper's tables and figure:

```bash
python analysis/table_friedman_results.py
python analysis/table_residual_results.py
python analysis/fig_friedman_test.py
```


## Pipeline documentation

The full pipeline (data curation, preprocessing, model optimization, nested CV/HPO, W&B logging, SHAP feature attribution) is documented in [`documentation/PIPELINE.md`](documentation/PIPELINE.md).
