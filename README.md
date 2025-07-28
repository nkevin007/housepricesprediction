# 🏠 House Price Prediction with Confidence Range

This project builds and deploys a machine learning model to **predict house sale prices** using key housing features such as living area, quality rating, and basement size. The model is trained on the **Ames Housing Dataset** and deployed via **Streamlit** with a built-in confidence range to account for prediction uncertainty.

----

## 📌 Objective

- Predict the `SalePrice` of homes using supervised regression.
- Identify the most influential features affecting property value.
- Provide an **interactive web app** with an easy-to-understand **confidence interval** to improve user trust.

---

## 📊 Dataset Overview

- **Source**: [Ames Housing Dataset – Kaggle](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)
- **Target Variable**: `SalePrice`
- **Observations**: 1,460 homes
- **Features**: 80+ numerical, categorical, and ordinal attributes

---

## 🧪 Tools & Libraries

- **Python** – Core programming language
- **Streamlit** – Interactive app deployment
- **Jupyter Notebook** – Model development and EDA

### 🔧 Libraries Used

- `pandas`, `numpy` – Data handling
- `matplotlib`, `seaborn` – Visualizations
- `scikit-learn` – Machine learning modeling
- `joblib` – Model serialization
- `streamlit` – Web app interface

---

## ⚙️ Machine Learning Workflow

### 1. Data Preprocessing

- Handled missing values
- Dropped low-variance/uninformative columns
- Focused on numeric features for initial version

### 2. Exploratory Data Analysis (EDA)

- Correlation heatmaps and feature ranking
- Distribution analysis of `SalePrice`
- Visualized top features by importance

### 3. Model Training & Evaluation

- Trained a `RandomForestRegressor`
- Hyperparameter tuning with `GridSearchCV`
- Evaluation metrics:
  - **R² Score**: ~0.89
  - **MSE**: ~841,309,331

---

## ✅ Final Model Summary

| Metric   | Value         |
|----------|---------------|
| R² Score | ~0.89         |
| MSE      | ~841M         |

> 📌 The model explains **~89%** of the variance in sale prices on the test set.

---

## 🔍 Top 5 Most Important Features

| Feature       | Importance |
|---------------|------------|
| OverallQual   | 0.53       |
| GrLivArea     | 0.14       |
| TotalBsmtSF   | 0.04       |
| 2ndFlrSF      | 0.03       |
| BsmtFinSF1    | 0.03       |

---

## 🚀 Streamlit App Features

- ✅ Clean UI to input housing features
- ✅ 💰 Real-time price prediction
- ✅ 📉 95% **confidence interval** using **bootstrapping**
- ✅ 📊 Sidebar chart of top model features
- ✅ 🧠 Explanation of how confidence range works (to educate users)

📸 *Sample output:*

> 💰 Estimated Sale Price: \$240,000  
> 📉 95% Confidence Range: \$220,000 – \$260,000

> 🔍 “This confidence range means the model expects the house to sell between \$220K and \$260K, based on patterns learned from similar homes.”

---

## 💡 Future Improvements

- Include categorical encoding (e.g., Neighborhood, HouseStyle)
- Compare against XGBoost, LightGBM, and Linear Regression
- Add historical price distribution plots
- Add REST API (Flask/FastAPI) or deploy to Hugging Face Spaces

---

## 🛠️ How to Run This Project

### 1. Clone the repository

```bash
git clone https://github.com/nkevin007/housepricesprediction.git
cd housepricesprediction

2. Install dependencies

pip install -r requirements.txt

Or manually:

pip install streamlit pandas scikit-learn matplotlib seaborn joblib

3. Run the Streamlit app

streamlit run app.py

    Open http://localhost:8501 in your browser.

📂 Files in This Repo
File	Description
app.py	Streamlit web app for prediction
predict.py (optional)	Helper script to load model (if used)
random_forest_model.pkl	Trained ML model saved with joblib
expected_features.csv	Column structure for model input
housepricesprediction.ipynb	Full EDA and model training notebook
README.md	Project overview and usage guide

Made by KN
