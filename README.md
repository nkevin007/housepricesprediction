# 🏠 House Price Prediction with Machine Learning

This project builds and evaluates a machine learning model to predict house sale prices based on housing features such as square footage, quality, basement area, and garage size. The model is trained on the **Ames Housing Dataset**, a well-known real estate dataset frequently used in regression modeling competitions.

---

## 📌 Objective

Predict the `SalePrice` of homes using supervised regression and identify the most influential features affecting property value.

---

## 📊 Dataset Overview

- 📁 **Source**: [Kaggle – Ames Housing Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)  
- 🎯 **Target Variable**: `SalePrice`  
- 🏘️ **Observations**: 1,460 homes  
- 🧩 **Features**: 80+ numerical, categorical, and ordinal variables  

---

## 🧪 Tools & Libraries

- **Python** – Core programming language  
- **Jupyter Notebook** – Development environment  

**Libraries Used**:
- `pandas` – Data manipulation  
- `matplotlib` & `seaborn` – Visualization  
- `scikit-learn` – Modeling & evaluation  

---

## ⚙️ Machine Learning Workflow

### 1. Data Preprocessing
- Handled missing values  
- Removed low-variance or uninformative columns  
- Focused on numeric features for simplicity  

### 2. Exploratory Data Analysis (EDA)
- Visualized feature correlation with `SalePrice`  
- Plotted top 15 features by importance  
- Used heatmaps, scatter plots, and bar charts  

### 3. Model Training & Tuning
- Trained a `RandomForestRegressor` model  
- Tuned using `GridSearchCV` with 3-fold cross-validation  
- Evaluated with **Mean Squared Error (MSE)** and **R² Score**

---

## ✅ Final Model Results

| Metric       | Value        |
|--------------|--------------|
| R² Score     | ~0.89        |
| MSE          | ~841,309,331 |

📌 *This means the model explains ~89% of the variance in house prices.*

---

## 🔍 Top 5 Most Important Features

| Feature       | Importance   |
|---------------|--------------|
| `OverallQual` | 0.53         |
| `GrLivArea`   | 0.14         |
| `TotalBsmtSF` | 0.04         |
| `2ndFlrSF`    | 0.03         |
| `BsmtFinSF1`  | 0.03         |

---

## 💡 Future Improvements

- Include **categorical encoding** for richer feature use (e.g., `Neighborhood`)
- Compare against baseline models like **Linear Regression** or **XGBoost**
- Deploy as a web app (e.g., **Streamlit** or **Flask**) for interactive predictions

---

## 🛠️ How to Run This Project

Clone the repository:

```bash
git clone https://github.com/nkevin007/housepricesprediction.git
cd housepricesprediction

Open the notebook:

jupyter notebook housepricesprediction.ipynb

Install required libraries:

pip install pandas scikit-learn matplotlib seaborn

📂 Files in This Repo
File	Description
housepricesprediction.ipynb	Main notebook with full analysis
README.md	Project summary and usage guide
