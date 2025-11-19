🧠 Credit Risk Prediction with XGBoost & SHAP
Interpretable Machine Learning for Loan Default Analysis
📌 Project Overview

This project demonstrates an end-to-end credit risk prediction pipeline using machine learning and interpretability tools.
Students build, tune, and interpret an XGBoost classification model to predict whether a loan applicant will default.
A strong emphasis is placed on explainability using SHAP values, enabling transparent and actionable insights.

📂 Dataset

Name: Credit Risk.csv

Rows: 2000

Columns:

clientid — Identifier (removed during preprocessing)

income

age

loan

default — Target (0 = non-default, 1 = default)

🔧 Data Preprocessing

Missing age values imputed using median.

Dropped clientid (not a predictive feature).

Numerical features standardized using StandardScaler.

Train–test split: 80/20.

🤖 Modeling
Algorithm

XGBoost Classifier

Baseline model trained and evaluated using ROC-AUC.

Hyperparameter Tuning

Performed using GridSearchCV with parameters:

n_estimators

learning_rate

max_depth

subsample

📈 Final Model Performance

AUC Score: 0.9971 (significantly above the 0.80 target)

🔍 Model Interpretability (SHAP)

Model explainability is a core part of this project.

Global Interpretability

SHAP Summary Plot (Bar)

SHAP Beeswarm Plot

Permutation Feature Importance

Top features influencing default risk:

Loan amount

Age

Income

Local Interpretability

SHAP force plots generated for:

Correctly predicted default case

Correctly predicted non-default case

Borderline prediction (~0.5 probability)

These visualizations help explain individual decisions — essential in regulated financial applications.

📊 Key Insights

loan has the strongest influence on default risk.

age and income also contribute significantly.

Exceptionally high model performance suggests:

Patterns in the dataset are very strong

Dataset may be too simplified for real-world lending systems

Further data with more variables would improve generalizability

🚀 Technologies Used

Python

Pandas, NumPy

Scikit-learn

XGBoost

SHAP

Matplotlib

📁 Code Structure
│── shap_project.py      # Main script containing preprocessing, modeling, SHAP, and plots
│── Credit Risk.csv      # Dataset
│── README.md            # Project documentation

📘 How to Run

Clone this repository

git clone https://github.com/<your-username>/<your-repo>.git


Install dependencies

pip install -r requirements.txt


Run the script

python shap_project.py

📝 Executive Summary (For Clients)

The XGBoost model achieved an AUC of 0.9971, far exceeding expectations.
SHAP-based interpretability confirms that loan amount, age, and income are primary drivers of credit risk.

However, due to the limited size and feature depth of the dataset, the model may not generalize to real-world lending scenarios.
For strategic policy decisions, additional features such as credit history, employment stability, past delinquencies, and financial obligations should be incorporated.

✨ Final Note (Client Impression)

This project not only delivers a high-performing predictive model but also ensures full transparency, making it suitable for applications where fairness, accountability, and regulatory compliance matter.
By enhancing the dataset with richer financial attributes in future iterations, the system can evolve into a production-ready credit risk engine.
