# Fairness in Ranking — Conditional Independence Testing

This project tests **conditional independence** between a ranking $R$ and protected attributes $Z$ given legitimate features $X$:

$$H_0 \colon R \perp Z \mid X$$

using a **Conditional Randomisation Test (CRT)** with **conditional GAN** generators for the null distribution.

---

## Repository structure

```
iii/
├── src/                            # Core library
│   ├── methods.py                  # Scoring functions (KCondor, nKCI, nHSIC, …)
│   ├── crt_cgan.py                 # ConditionalGAN + CRT calibration
│   ├── synthetic.py                # Synthetic data generators
│   └── utils.py                    # Data loading, preprocessing, helpers
│
├── data/                           # Raw datasets
│   ├── adult.csv                   # UCI Adult (income)
│   ├── propublica_data_for_fairml.csv  # COMPAS recidivism
│   ├── clean_LawSchool.csv         # Law School (LSAT)
│   └── student_performance.json    # Student Performance (G3)
│
├── experiment_real_data.ipynb      # ★ Real-data experiment (all datasets)
├── test_synthetic.ipynb            # ★ Synthetic-data experiment (γ × β grid)
├── test_data_betas_v2.ipynb        # ★ Semi-synthetic β-sweep on real data
├── test_synthetic_plot_time_opt.ipynb  # ★ Runtime benchmark
├── visual.ipynb                    # ★ Visualisation & figure generation
│
├── results/                        # Saved pickle outputs (auto-created)
├── visualizations/                 # Saved PDF figures (auto-created)
└── OLD/                            # Archived notebooks & previous results
```

## Notebooks

| Notebook | Purpose |
|---|---|
| **experiment_real_data** | Run the CRT on a real dataset. Set `dataset_name` to one of `adults`, `propublica`, `law`, `edu` and execute all cells. Produces a p-value heatmap (method × protected feature). |
| **test_synthetic** | Sweep over `γ` (X→Z dependence) and `β` (Z→R influence) on fully synthetic data. Produces a grid of p-values per method. |
| **test_data_betas_v2** | Semi-synthetic: uses real data features but mixes true and protected-attribute-based rankings via `β`. Shows how p-values transition as bias increases. |
| **test_synthetic_plot_time_opt** | Measures wall-clock time of each method for increasing sample sizes. Compares optimisation levels of KCondor. |
| **visual** | Loads saved pickle results and generates publication-quality figures (heatmaps, contour plots, polar plots). |

## Datasets

| Key | Source | Target ($Y$) | Protected attributes ($Z$) |
|---|---|---|---|
| `adults` | UCI Adult | `income` (binary) | `gender`, `race` |
| `propublica` | ProPublica COMPAS | `Two_yr_Recidivism` | `African_American`, `Female` |
| `law` | Law School | `LSAT` (score) | `sex`, `race_nonwhite` |
| `edu` | Student Performance | `G3` (final grade) | `sex`, `address` |

## Methods

| Display name | Function | Description |
|---|---|---|
| **KCondor** | `Kcondor_v2` | Kernel conditional distance correlation via residualised kernel matrices |
| **nKCI** | `nkci_score` | Kernel Conditional Independence test statistic (causal-learn) |
| **nHSIC** | `nhsic` | Hilbert-Schmidt Independence Criterion p-value |
| **Partial Corr.** | `partial_corr_pg_score` | Partial correlation via OLS incremental R² |
| CMI | `cmi_score` | Conditional Mutual Information (NPEET) |

## Quick start

```bash
# activate the virtual environment
# activate your virtual environment, e.g.:
# source /path/to/your/venv/bin/activate
pip install -r requirements.txt   # if a requirements file is provided

# run a real-data experiment
# → open experiment_real_data.ipynb, set dataset_name, Run All
```

## Key parameters

| Parameter | Default | Meaning |
|---|---|---|
| `K` | 5 | K-Fold splits for cGAN training |
| `B` | 200 | Number of CRT bootstrap samples |
| `s_size` | 1000 | Balanced sample size per class |
| `γ` (gamma) | varies | Strength of X → Z dependence (synthetic) |
| `β` (beta) | varies | Influence of Z on the ranking (synthetic / semi-synthetic) |
