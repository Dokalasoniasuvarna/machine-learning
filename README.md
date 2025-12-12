# L1 and L2 Regularization in Logistic Regression

**MSc Machine Learning Tutorial (Individual Assignment)**
**Author:** Sonia Suvarna Dokala
**Student ID:** 24085938
**Programme:** MSc Data Science
**University:** University of Hertfordshire

***

## 📚 Project Overview

This repository contains an MSc tutorial project on **L1 and L2 regularization in logistic regression**, focusing on how these techniques control model complexity and improve generalization on classification tasks. The work combines theory, geometric intuition, and experiments to show how different penalties affect model weights, decision boundaries, and performance.

***

## 🎓 Learning Objectives

By exploring this project, you will:

- Understand why overfitting occurs in logistic regression and how regularization helps prevent it.
- Learn the mathematical formulations of **L1 (Lasso)** and **L2 (Ridge)** regularization for logistic regression.
- Gain geometric intuition for $\ell_1$ vs. $\ell_2$ constraints in weight space.
- See when to choose L1, L2, or Elastic Net in practice.
- Practice tuning regularization strength using cross‑validation and interpreting regularized coefficients.

***

## 📁 Repository Structure

```text
logistic-regularization-tutorial/
│
├── notebooks/
│   └── logistic_l1_l2_elasticnet.ipynb    # Main Jupyter notebook (code & experiments)
├── L1-and-L2-Regularization-in-Logistic-Regression.pdf   # Full written report
├── requirements.txt                        # Python dependencies
├── README.md                               # Project documentation (this file)
├── LICENSE                                 # Project license
└── .gitignore                              # Standard Python gitignore
```

- **`L1-and-L2-Regularization-in-Logistic-Regression.pdf`** – Detailed report with theory, methodology, results, and discussion.
- **`notebooks/logistic_l1_l2_elasticnet.ipynb`** – Implementation of logistic regression with L1, L2, and Elastic Net, plus visualizations.
- **`requirements.txt`** – List of required Python packages.

***

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/logistic-regularization-tutorial.git
cd logistic-regularization-tutorial

# (Optional) create and activate a virtual environment, then:
pip install -r requirements.txt
```


### 2. Run the Notebook

```bash
jupyter notebook notebooks/logistic_l1_l2_elasticnet.ipynb
```

Then:

1. Open `logistic_l1_l2_elasticnet.ipynb`.
2. Run all cells (Cell → Run All).
3. The notebook will:
    - Train logistic regression models with **no regularization**, **L1**, **L2**, and **Elastic Net**.
    - Visualize decision boundaries and coefficient magnitudes for different regularization strengths.
    - Report training/validation metrics to illustrate the bias–variance trade‑off.

***

## 🎯 Key Concepts and Findings

- **L2 Regularization (Ridge)**
    - Penalizes the sum of squared weights.
    - Shrinks all coefficients smoothly toward zero but typically keeps them non‑zero (dense solution).
    - Provides stability when features are correlated and is suitable when most features may be informative.
- **L1 Regularization (Lasso)**
    - Penalizes the sum of absolute weight values.
    - Sets many coefficients exactly to zero, performing automatic feature selection and improving interpretability.
- **Elastic Net**
    - Combines L1 and L2 to balance sparsity and stability, useful for correlated predictors when feature selection is still desired.
- **Regularization Strength**
    - Controlled by $\lambda$ (or $C = 1/\lambda$ in scikit‑learn).
    - Too small: risk of overfitting; too large: risk of underfitting.
    - The report recommends grid search with cross‑validation to find a suitable value.

***

## 🧪 Implementation Notes

- **Feature Scaling**
Features should be standardized (e.g. mean 0, variance 1) before fitting regularized models, so that penalties are applied fairly across all coefficients.
- **Model Evaluation**
The project emphasises using separate training, validation, and test data, and tuning regularization only on training/validation sets to avoid optimistic evaluation.

***

## 👤 Author

**Name:** Sonia Suvarna Dokala
**Student ID:** 24085938
**Github : ** https://github.com/Dokalasoniasuvarna
**Programme:** MSc Data Science, University of Hertfordshire

***

## 📜 License

This project is released under the license specified in the `LICENSE` file (e.g. MIT). Please refer to that file for the detailed terms.

***

## 📌 Academic Note

This repository accompanies the MSc tutorial report *“L1 and L2 Regularization in Logistic Regression: A Comprehensive Guide”* and is intended for educational and research purposes. The content of this README is an original summary of the attached report and respects intellectual property.
