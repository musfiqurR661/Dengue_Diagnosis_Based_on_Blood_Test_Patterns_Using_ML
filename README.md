# Dengue Diagnosis Based on Blood Test Patterns Using ML

Predict dengue infection from routine blood-test patterns using machine learning. This repository contains exploratory notebooks, preprocessing, model training, evaluation, and guidance to reproduce and extend experiments.

- Primary language: Jupyter Notebook (analysis and experiments)
- Supporting language: Python (scripts and utilities)

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Notebooks and How to Run](#notebooks-and-how-to-run)
- [Reproduce Training & Evaluation](#reproduce-training--evaluation)
- [Results & Evaluation Metrics](#results--evaluation-metrics)
- [Tips to Improve Performance](#tips-to-improve-performance)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

Dengue is a mosquito-borne viral infection that can lead to severe health complications. This project explores whether patterns in routine blood tests can be used to classify patients as dengue-positive or dengue-negative using machine learning. The focus is on data cleaning, feature engineering, model selection, evaluation, and sharing reproducible notebooks.

## Repository Structure

- notebooks/ or .ipynb files at root — primary exploratory and modeling notebooks
- data/ (if present) — raw and processed datasets (not included in repo by default)
- src/ (if present) — helper Python modules for preprocessing, modeling, utilities
- README.md — this file

(Adjust the structure above if your repo places notebooks or scripts in a different path.)

## Dataset

- Description: Routine blood test records with labels indicating dengue diagnosis (positive/negative).
- Typical features: platelet count, hematocrit, leukocyte count, hemoglobin, mean platelet volume, age, etc.
- Format: CSV / DataFrame inside Jupyter notebooks
- Source: Please add dataset origin here (e.g., hospital dataset, public UCI repository, Kaggle). If the dataset cannot be shared due to privacy, include synthetic or example data and provide instructions for users to supply their own.

Important: If this project uses sensitive medical data, ensure you follow all applicable regulations and de-identification best practices before sharing.

## Methodology

High-level steps applied in the notebooks:

1. Data loading and exploration
2. Missing value handling (imputation or removal)
3. Outlier detection and treatment
4. Feature engineering and selection
5. Scaling and normalization where needed
6. Handling class imbalance (resampling, class weights)
7. Model training and hyperparameter tuning (e.g., logistic regression, random forest, XGBoost, and others)
8. Evaluation using cross-validation and a held-out test set
9. Explainability (feature importance, SHAP, or similar) — optional

## Notebooks and How to Run

Open the notebooks in the repository with Jupyter or JupyterLab. Notebooks are designed to be read end-to-end:

Prerequisites
- Python 3.8+
- Recommended to use a virtual environment or conda environment

Install dependencies (example):
```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
# OR, if no requirements file:
pip install jupyter numpy pandas scikit-learn matplotlib seaborn xgboost shap
```

Start Jupyter:
```bash
jupyter notebook
# or
jupyter lab
```

Open the main notebooks (filenames may vary):
- 01_data_exploration.ipynb — exploratory data analysis and visualization
- 02_preprocessing_and_feature_engineering.ipynb — cleaning and preparing features
- 03_model_training_and_evaluation.ipynb — training, cross-validation, and evaluation
- 04_model_interpretation.ipynb — explainability and feature importance (if present)

If you prefer command-line execution you can convert notebooks to scripts or use nbconvert to run them headless:
```bash
jupyter nbconvert --to notebook --execute 03_model_training_and_evaluation.ipynb
```

## Reproduce Training & Evaluation

1. Place the dataset CSV in a `data/` folder or adjust the path in the notebooks.
2. Run the notebooks in order (exploration → preprocessing → training → evaluation).
3. Checkpoints and model pickles (if created) will be saved to an `artifacts/` or `models/` folder (not included by default).

If you want to run training scripts directly (if present):
```bash
python src/train.py --data-path data/dataset.csv --output-dir models/ --model random_forest
```
(Implement CLI options in src/train.py if you prefer script-based training.)

## Results & Evaluation Metrics

Use the following metrics to evaluate models in this medical classification task:
- Accuracy
- Precision, Recall (Sensitivity) — recall is particularly important to minimize false negatives
- F1-score
- ROC-AUC
- Confusion matrix

Note: Because of class imbalance and the medical context, emphasize recall/sensitivity and precision depending on intended use (screening vs confirmation).

If you publish specific experiment numbers or model checkpoints, include a reproducible run command and random seeds used.

## Tips to Improve Performance

- Try additional models (LightGBM, CatBoost, ensemble stacking)
- More robust feature engineering: domain-specific derived features
- Better missing-value imputation (KNN, MICE)
- Use nested cross-validation for hyperparameter tuning
- Calibrate probabilities (Platt scaling or isotonic regression) if decision thresholds are important
- Evaluate with stratified cross-validation to preserve class ratios
- Use explainability tools (SHAP, LIME) to check model behavior for clinical plausibility

## Contributing

Contributions are welcome! If you want to improve notebooks, add scripts, or provide cleaned datasets (with appropriate permissions), please:

1. Fork the repository
2. Create a feature branch: git checkout -b feat/your-feature
3. Make changes and add tests or reproducible examples
4. Open a Pull Request describing your changes

Please do not commit private / sensitive data. Use synthetic or example files if needed.

## License

This project is provided under the MIT License. See LICENSE file for details. Add or update the license file in the repository if it's not already present.

## Contact

Repository owner: musfiqurR661  
GitHub: https://github.com/musfiqurR661

If you want, I can:
- Add a requirements.txt based on the notebooks
- Convert the most important notebooks into runnable scripts
- Create a small example dataset and a simple end-to-end script that trains and evaluates a model

Choose one of the options above and I’ll generate the files next.
