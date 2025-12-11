L1 and L2 Regularization in Logistic Regression
MSc Machine Learning Tutorial
Author: Sonia Suvarna Dokala
Student Id : 24085938
University of Hertfordshire

📚 Overview
This project presents a comprehensive tutorial on L1 and L2 regularization in logistic regression, focusing on how these techniques control model complexity and improve generalization. The report combines mathematical foundations, geometric intuition, and practical guidance for applying regularization in real-world classification tasks.

You will learn:

Why overfitting occurs and how regularization modifies the loss function to prevent it.

The mathematical and geometric differences between L1 (Lasso) and L2 (Ridge) penalties.

How regularization strength affects decision boundaries, weight distributions, and model performance.

📁 Repository Structure
text
logistic-regularization-tutorial/
│
├── L1-and-L2-Regularization-in-Logistic-Regression.pdf
├── notebooks/
│   └── logistic_l1_l2_elasticnet.ipynb
├── requirements.txt
├── README.md
└── LICENSE
L1-and-L2-Regularization-in-Logistic-Regression.pdf – Main report explaining theory, experiments, and conclusions.

notebooks/logistic_l1_l2_elasticnet.ipynb – Jupyter notebook implementing logistic regression with L1, L2, and Elastic Net, including visualization of weights and decision boundaries.

requirements.txt – Python dependencies (e.g. scikit‑learn, numpy, matplotlib).

LICENSE – Project license.

🧠 Core Concepts Covered
Regularization and Overfitting
The report describes overfitting as memorization of noise when models are too flexible relative to available data. Regularization adds a penalty term to the loss function, so the objective becomes Loss=Data Fit+λ⋅Penalty, where λ controls the strength of complexity control.

L2 (Ridge) Regularization
Adds a quadratic penalty proportional to the sum of squared weights.

Shrinks all coefficients smoothly toward zero, but typically does not make them exactly zero.

Geometrically, constrains the solution inside an ℓ 
2
  ball (circle/sphere) in weight space.

Best suited when all features may be informative and stability is important.

L1 (Lasso) Regularization
Adds a penalty proportional to the sum of absolute values of weights.

Drives many coefficients exactly to zero, performing automatic feature selection and producing sparse models.

Geometrically, constrains the solution inside an ℓ 
1
  ball (diamond shape), encouraging axis-aligned solutions.

Useful when interpretability and feature selection are priorities.

Elastic Net
The report also discusses Elastic Net, which combines L1 and L2 penalties to balance sparsity and stability. This is recommended when features are correlated but some level of feature selection is still desired.

🚀 Quick Start
Installation
bash
# Clone the repository
git clone https://github.com/<your-username>/logistic-regularization-tutorial.git
cd logistic-regularization-tutorial

# Install dependencies
pip install -r requirements.txt
Running the Notebook
Launch Jupyter Notebook or JupyterLab.

Open notebooks/logistic_l1_l2_elasticnet.ipynb.

Run all cells to:

Train logistic regression models with different penalties (L1, L2, Elastic Net).

Visualize decision boundaries and weight magnitudes under varying λ.

Compare training/validation performance and observe the bias–variance trade‑off.

🎯 Learning Outcomes
By completing this tutorial, you will be able to:

Explain the role of regularization in controlling model complexity and generalization in logistic regression.

Choose between L1, L2, and Elastic Net based on data characteristics and goals (stability vs. sparsity vs. interpretability).

Implement regularized logistic regression in Python, perform feature scaling, and tune λ using cross‑validation.

Interpret model coefficients and understand how regularization shapes decision boundaries and weight distributions.

📜 Copyright
This README provides a concise, original summary of the attached report and respects intellectual property and copyright. For full details, derivations, and figures, refer to the PDF report included in the repository
- Recap the value of hands-on experience with L1 and L2 regularization.
- Emphasize the importance of understanding regularization for building models that generalize well.
- Note that proper use of regularization prevents overfitting and improves performance on unseen data.
