# 🏠 House Prices Prediction

This project builds and evaluates a machine learning model to predict house prices using data features like square footage, quality, number of rooms, and more.

The goal is to explore real-world regression modeling and identify which features have the most impact on house prices.

## 🔍 Project Overview

- **Dataset**: Ames Housing Dataset
- **Goal**: Predict `SalePrice` based on housing features
- **Model Used**: Random Forest Regressor (tuned with GridSearchCV)
- **Tools**: Python, Pandas, Scikit-learn, Matplotlib, Jupyter Notebook

---

## 📊 Key Features Explored

- Neighborhood
- Overall Quality
- Living Area (GrLivArea)
- Garage Size
- Basement Area
- Year Built

---

## ⚙️ Machine Learning Process

1. **Data Preprocessing**
   - Cleaned missing values
   - Handled categorical variables
   - Scaled features

2. **Exploratory Data Analysis (EDA)**
   - Plotted average sale price by neighborhood
   - Visualized top features by importance

3. **Model Training**
   - Used `RandomForestRegressor`
   - Tuned with `GridSearchCV`

4. **Evaluation**
   - Mean Squared Error (MSE)
   - R² Score

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| MSE    | ~841,309,331 |
| R²     | ~0.89 |

---

## 🔍 Feature Importance (Top 5)

| Feature       | Importance |
|---------------|------------|
| OverallQual   | 0.53       |
| GrLivArea     | 0.14       |
| TotalBsmtSF   | 0.04       |
| 2ndFlrSF      | 0.03       |
| BsmtFinSF1    | 0.03       |

*(Your actual values may vary slightly depending on tuning)*

---

## 💻 How to Run

1. Clone this repo:
