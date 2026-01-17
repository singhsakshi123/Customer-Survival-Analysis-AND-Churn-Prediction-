# Customer Survival Analysis & Churn Prediction (Telco Customers)

## 📌 Project Summary
This project analyzes **customer churn** using:
1) **Exploratory Data Analysis (EDA)** to understand churn patterns  
2) **Survival Analysis** to study *when* customers churn (time-to-event)  
3) **Machine Learning** to predict *whether* a customer will churn  

Dataset: **Telco Customer Churn (`Telco_Cusomer_Churn.csv`)**


## 🎯 Business Goal
- Identify customers likely to churn and the main drivers behind churn  
- Understand churn risk over time (customer lifetime / tenure-based churn behavior)  
- Support retention strategy decisions using interpretable model results  


## 🗂 Dataset Overview
The dataset contains customer demographics, services, contract details, and billing data.

**Target column**
- `Churn` → **0 = No**, **1 = Yes**

**Key features**
- **Customer profile:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`  
- **Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `TechSupport`, `StreamingTV`, etc.  
- **Billing & contract:** `Contract`, `PaymentMethod`, `PaperlessBilling`  
- **Charges:** `MonthlyCharges`, `TotalCharges`  
- **Time feature:** `tenure` (used for survival analysis)


## ⚙️ Data Preprocessing (Common Steps)
- Dropped `customerID`
- Cleaned `TotalCharges` (converted to numeric, handled blanks/missing values)
- Encoded binary columns (Yes/No → 1/0)
- One-hot encoded categorical columns:
  - `InternetService`
  - `Contract`
  - `PaymentMethod`


## 📁 Project Workflow + Files (What each notebook contains)

### 1) `Exploratory Data Analysis.ipynb` (EDA)
**What I did**
- Checked missing values, datatypes, and class imbalance
- Visualized churn across customer segments using countplots and distributions
- Compared churn trends across:
  - contract types
  - internet service types
  - payment methods
  - tenure and monthly charges

**Key insights**
- Customers with **shorter tenure** show higher churn risk  
- **Month-to-month contracts** are strongly associated with churn  
- Customers with **fiber optic internet** show higher churn trend vs other services  
- Billing features like **MonthlyCharges / TotalCharges** strongly affect churn behavior  


### 2) `Customers Survival Analysis.ipynb` (Time-to-Churn Modeling)
**What I did**
- Used **tenure as time duration**
- Modeled churn using **Survival Analysis**
- Applied Cox Proportional Hazards / survival curve based comparisons

**Key insights**
- Survival probability decreases as tenure increases (churn happens more in early months)
- Contract type and service type affect churn timing
- Survival analysis explains **when churn happens**, not just whether it happens  


### 3) `Churn Prediction Model.ipynb` (ML Model Training + Tuning)
**What I did**
- Built churn classification models using:
  - RandomForestClassifier (baseline + tuned)
  - XGBoostClassifier (tuned)
- Tuned hyperparameters using **GridSearchCV / RandomizedSearchCV**
- Improved decision-making by **threshold tuning** instead of using default 0.50
- Evaluated results using:
  - Confusion Matrix
  - Precision / Recall / F1-score (Churn = 1)
  - ROC-AUC
  - PR-AUC (Average Precision)

**Final Test Results (XGBoost - Tuned)**
- **ROC-AUC:** **0.8525**
- **PR-AUC:** **0.6754**
- **Best Threshold:** **0.60**
- **Churn Class (1) Performance**
  - **Precision:** **0.5867**
  - **Recall:** **0.7059**
  - **F1-score:** **0.6408**

**Baseline Test Results (Random Forest)**
- **ROC-AUC:** **0.8483**
- **PR-AUC:** **0.6729**
- **Best Threshold:** **0.47**
- **Churn Class (1) Performance**
  - **Precision:** **0.5814**
  - **Recall:** **0.6872**
  - **F1-score:** **0.6299**

**What these numbers mean**
- XGBoost slightly outperformed Random Forest across metrics:
  - Higher **ROC-AUC** and **PR-AUC** (better overall ranking for churn risk)
  - Higher churn **Recall** (captures more churn customers)
  - Higher churn **F1-score** (better precision-recall balance)
- Since churn is an imbalanced classification problem, **PR-AUC and churn-class F1/Recall** were prioritized over accuracy.



## 🔍 Explainability & Insights from the Final Model
To interpret why the model predicts churn:

### ✅ Feature Importance (Tree-based)
Top churn-driving factors observed:
- `tenure` (most important)
- `MonthlyCharges`
- `TotalCharges`
- `Contract_Two year` (strong retention effect)
- `InternetService_Fiber optic`

### ✅ Permutation Importance
Used to validate feature impact by measuring score drop when a feature is shuffled.

### ✅ SHAP (Explainability)
- **Global SHAP:** identifies which features drive churn overall  
- **Local SHAP:** explains individual predictions (why 1 customer churned)


## 🛠 Tools & Libraries Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- lifelines (Survival Analysis)
- SHAP
- eli5 (Permutation Importance)


## ✅ Final Conclusion
This project delivers a complete churn analytics pipeline:
- EDA explains **what patterns drive churn**
- Survival analysis explains **when churn happens**
- ML model predicts **who will churn** with:
  - **ROC-AUC ~ 0.85**
  - **Churn F1 ~ 0.62**
  - **Churn Recall ~ 0.66**

This can be used for retention actions like contract upgrades, pricing strategies, and targeted customer support.
