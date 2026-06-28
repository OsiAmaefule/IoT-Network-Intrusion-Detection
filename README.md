# IoT IDS XAI Project

Explainable AI Extension of "Ensemble-Based Lightweight Machine Learning Optimization for IoT Network Intrusion Detection" (Samantaray et al., ISACC 2025).

## Project Structure

```
IoT_IDS_XAI_Project/
├── data/                  # Place UNR-IDD dataset files here
├── src/                   # Source code modules
│   ├── preprocessing.py   # Dataset loading, normalization, train/test split
│   ├── csa_optimizer.py   # Crow Search Algorithm wrapper (via mealpy)
│   ├── classifiers.py     # Classifier definitions and hyperparameter spaces
│   └── xai_analysis.py    # SHAP and LIME analysis functions
├── notebooks/             # Jupyter notebooks (run in order)
│   ├── 01_eda.ipynb       # Exploratory data analysis
│   ├── 02_replication.ipynb  # Train 5 CSA-optimized classifiers
│   └── 03_xai_analysis.ipynb # SHAP + LIME explainability (core contribution)
├── results/               # Saved models (.pkl), SHAP values (.npy), CSVs
├── figures/               # Publication-quality plots (300 DPI)
├── paper/                 # IEEE manuscript drafts and LaTeX/Word files
├── requirements.txt       # Python dependencies
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Download UNR-IDD from: https://sites.google.com/view/tapadhirdas/unr-idd-dataset

Place the CSV file(s) in the `data/` directory.

## Workflow

1. Run `01_eda.ipynb` — explore the dataset, verify class distribution
2. Run `02_replication.ipynb` — train all 5 CSA-optimized models, compare to paper
3. Run `03_xai_analysis.ipynb` — SHAP and LIME analysis (the extension contribution)
4. Use outputs from `figures/` and `results/` to write the paper in `paper/`
