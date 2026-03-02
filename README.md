# Audit Ranking

Test whether a ranking **R** is conditionally independent of protected attributes **Z** given legitimate features **X**:

**H₀ : R ⊥ Z | X**

The test uses a **Conditional Randomisation Test (CRT)** with conditional GAN generators.

## Quick start

```bash
# 1. Clone & enter
git clone https://github.com/Ambress92/Audit_ranking.git
cd Audit_ranking

# 2. Create a virtual environment & install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the pipeline

Use `run_condor.py` to run the KCondor CRT pipeline from the command line.

### Built-in datasets

Four datasets are included in `data/`: **adults**, **propublica**, **law**, **edu**.

```bash
# Default parameters (K=5 folds, B=200 bootstrap, sample_size=1000)
python run_condor.py --dataset adults

# Custom parameters
python run_condor.py --dataset law --K 3 --B 100 --sample-size 500

# Save to a specific file
python run_condor.py --dataset propublica --output results/propublica.pkl
```

### Custom CSV

```bash
python run_condor.py --csv my_data.csv \
    --y-col target \
    --z-groups "gender" "race" "gender,race"
```

### Output

Each run produces:
- a `.pkl` file with the raw results dict
- a `.json` summary with dataset info, parameters, and p-values

Both are saved in `results/` by default.

### Parameters

| Flag | Default | Description |
|---|---|---|
| `--dataset` | — | Built-in dataset (`adults`, `propublica`, `law`, `edu`) |
| `--csv` | — | Path to a custom CSV (alternative to `--dataset`) |
| `--y-col` | — | Target column name (required with `--csv`) |
| `--z-groups` | — | Protected feature groups, comma-separated (required with `--csv`) |
| `--K` | 5 | Number of K-Fold splits for cGAN training |
| `--B` | 200 | CRT bootstrap iterations |
| `--sample-size` | 1000 | Balanced sample size per class |
| `--output` | auto | Output pickle path |
| `--seed` | 42 | Random seed |

## Notebooks

The repository also includes Jupyter notebooks for interactive exploration:

| Notebook | Purpose |
|---|---|
| `experiment_real_data` | Run the full CRT on real datasets (all methods) |
| `test_synthetic` | Synthetic-data experiment (γ × β grid) |
| `test_data_betas_v2` | Semi-synthetic β-sweep on real data |
| `test_synthetic_plot_time_opt` | Runtime benchmark |
| `visual` | Generate publication figures from saved results |

## Project structure

```
├── run_condor.py          # CLI entry point
├── src/
│   ├── methods.py         # Scoring functions (KCondor, nKCI, nHSIC, …)
│   ├── crt_cgan.py        # Conditional GAN + CRT calibration
│   ├── synthetic.py       # Synthetic data generators
│   └── utils.py           # Data loading & preprocessing
├── data/                  # Datasets (CSV / JSON)
├── results/               # Output pickles & JSON summaries
└── visualizations/        # Saved figures
```
