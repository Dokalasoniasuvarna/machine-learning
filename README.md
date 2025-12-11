L1 and L2 Regularization in Logistic Regression
MSc Machine Learning Tutorial (Individual Assignment)
Author: Sonia Suvarana Dokala
Student ID: 24085938
Programme: MSc Data Science
University: University of Hertfordshire

📚 Project Overview
This project provides a comprehensive tutorial on L1 and L2 regularization in logistic regression, focusing on how these techniques control model complexity and improve generalization performance. The report integrates theory, geometric intuition, and empirical results to show how different penalties influence decision boundaries, weight distributions, and classification accuracy.

The work is submitted as part of a university machine learning module and demonstrates understanding of overfitting, the bias–variance trade‑off, and practical regularization strategies in linear models.

🧠 Theoretical Background
Regularization is introduced as a method to reduce overfitting by adding a penalty term to the loss function so that the model balances data fit with simplicity. The regularized logistic regression objective is expressed as a sum of the data loss and a penalty scaled by a hyperparameter λ, which controls regularization strength.

L2 (Ridge) Regularization
Adds a quadratic penalty proportional to the sum of squared weights.

Shrinks all coefficients smoothly toward zero but rarely sets them exactly to zero, resulting in dense models.

Geometrically, constrains the solution within an ℓ 
2
  ball, which encourages stable solutions and distributes weight across correlated features.

L1 (Lasso) Regularization
Adds a penalty proportional to the sum of absolute weight values.

Drives many coefficients exactly to zero, performing automatic feature selection and producing sparse, interpretable models.

Geometrically, constrains the solution within an ℓ 
1
  ball, which favours axis-aligned solutions and promotes sparsity.

Elastic Net
The report also discusses Elastic Net as a combination of L1 and L2 penalties that trades off between sparsity and stability. This is useful when features are correlated but interpretability through feature selection is still important.

🧪 Practical Implementation
The report uses logistic regression as the base model to empirically study the impact of different regularization schemes. Typical experiments include:

Comparing models with no regularization, pure L1, pure L2, and Elastic Net.

Examining how weight magnitudes and sparsity patterns change as λ increases.

Visualizing decision boundaries to see how regularization smooths and stabilizes classification regions.

Key implementation and tuning points:

Feature scaling: Standardizing features is emphasised as essential so that regularization penalizes coefficients fairly across all dimensions.

Hyperparameter tuning: Grid search with cross‑validation over a range of λ values (or C=1/λ in scikit‑learn) is recommended to find an appropriate regularization strength.

Model interpretation: Coefficients are interpreted in terms of sign (direction of effect) and magnitude (strength of effect), with L1 additionally used for feature selection.

🗂 Suggested Repository Structure
A clean, university‑ready repository layout might be:

text


logistic-regularization-tutorial/
│
├── L1-and-L2-Regularization-in-Logistic-Regression.pdf
├── notebooks/
│   └── logistic_l1_l2_elasticnet.ipynb
├── requirements.txt
├── README.md
└── LICENSE


🚀 How to Run (for markers)
Set up environment

bash
git clone https://github.com/<your-username>/logistic-regularization-tutorial.git
cd logistic-regularization-tutorial
pip install -r requirements.txt
Open the notebook

Launch Jupyter Notebook or JupyterLab.

Open notebooks/logistic_l1_l2_elasticnet.ipynb.

Reproduce experiments

Run all cells to train logistic regression models with L1, L2, and Elastic Net penalties.

Inspect decision boundary plots, coefficient paths, and performance metrics across different values of λ.

🎯 Learning Outcomes (Academic)
This work demonstrates that the student can:

Explain and motivate the use of L1 and L2 regularization in logistic regression.

Relate regularization strength to the bias–variance trade‑off and overfitting behaviour.

Implement regularized logistic regression models, perform appropriate feature scaling, and tune hyperparameters using cross‑validation.

Interpret regularized coefficients in terms of feature importance and model robustness, and justify when to prefer L1, L2, or Elastic Net in practice.

📜 Academic Integrity and Copyright
This README provides a concise, original summary of the attached report for university submission and does not reproduce copyrighted text. The full theoretical derivations, figures, and detailed experimental analysis are contained in “L1 and L2 Regularization in Logistic Regression: A Comprehensive Guide” authored by Sonia Suvarana Dokala
