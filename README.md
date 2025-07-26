 🏠 House Price Prediction with Machine Learning

This project builds and evaluates a machine learning model to predict house sale prices based on housing features such as square footage, quality, basement area, and garage size. The model is trained on the Ames Housing Dataset, a well-known real estate dataset frequently used in regression modeling competitions.
📌 Objective

Predict the SalePrice of homes using supervised regression and identify the most influential features affecting property value.
📊 Dataset Overview

    Source: Kaggle – Ames Housing Dataset

    Target Variable: SalePrice

    Observations: 1,460 homes

    Features: 80+ features including numerical, categorical, and ordinal data

🧪 Tools & Libraries

    Python: Core programming language

    Jupyter Notebook: Development environment

    Libraries:

        pandas for data manipulation

        matplotlib and seaborn for visualization

        scikit-learn for modeling and evaluation

⚙️ Machine Learning Workflow

    Data Preprocessing

        Handled missing values

        Removed low-variance columns

        Focused on numeric features for simplicity

    Exploratory Data Analysis

        Visualized correlation with SalePrice

        Plotted top 15 features by importance

        Used heatmaps, scatter plots, and bar charts

    Model Training & Tuning

        Trained a RandomForestRegressor model

        Tuned using GridSearchCV with 3-fold cross-validation

        Evaluated with Mean Squared Error (MSE) and R² Score

✅ Final Model Results
Metric	Value
R² Score	~0.89
Mean Squared Error (MSE)	~841,309,331

    📌 The model explains ~89% of the variance in house prices.

🔍 Top 5 Most Important Features
Feature	Importance
OverallQual	0.53
GrLivArea	0.14
TotalBsmtSF	0.04
2ndFlrSF	0.03
BsmtFinSF1	0.03
💡 Future Improvements

    Include categorical encoding for richer feature use (e.g. Neighborhood)

    Compare against baseline models like Linear Regression or XGBoost

    Deploy as a web app (e.g., Streamlit or Flask) for interactive use

🛠️ How to Run This Project

    Clone the repository:

git clone https://github.com/nkevin007/housepricesprediction.git
cd housepricesprediction

Open the notebook:

jupyter notebook housepricesprediction.ipynb

Run all cells in order. You must have the following libraries installed:

    pip install pandas scikit-learn matplotlib seaborn

📂 Files in This Repo
File	Description
housepricesprediction.ipynb	Main notebook with full analysis
README.md	Project summary and usage guide
