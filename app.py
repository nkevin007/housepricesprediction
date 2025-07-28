import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load('random_forest_model.pkl')

# Load expected feature list
expected_features = pd.read_csv("expected_features.csv", header=None)[0].tolist()

# Title and description
st.set_page_config(page_title="House Price Estimator with Confidence", layout="centered")
st.title("🏡 House Price Estimator with Confidence Range")
st.write("Fill in the house details below to estimate the price along with a confidence range.")

# 📥 Input fields
OverallQual = st.slider("Overall Quality (1–10)", 1, 10, 5)
GrLivArea = st.number_input("Above Ground Living Area (sq ft)", 500, 5000, 1500)
TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", 0, 3000, 800)
SecondFlrSF = st.number_input("2nd Floor Area (sq ft)", 0, 3000, 500)
BsmtFinSF1 = st.number_input("Finished Basement Area (sq ft)", 0, 2000, 400)

# 🔢 Create input row filled with zeros
input_data = pd.DataFrame([np.zeros(len(expected_features))], columns=expected_features)

# ✍️ Replace known values
input_data["OverallQual"] = OverallQual
input_data["GrLivArea"] = GrLivArea
input_data["TotalBsmtSF"] = TotalBsmtSF
input_data["SecondFlrSF"] = SecondFlrSF
input_data["BsmtFinSF1"] = BsmtFinSF1

# Bootstrapped predictions
def bootstrap_predictions(model, X, n=100):
    preds = []
    for _ in range(n):
        # Random subset of trees (bootstrap)
        estimator = np.random.choice(model.estimators_)
        preds.append(estimator.predict(X)[0])
    return np.array(preds)

# 🧠 Predict
if st.button("Estimate Price"):
    central = model.predict(input_data)[0]
    boot_preds = bootstrap_predictions(model, input_data, n=100)
    lower = np.percentile(boot_preds, 2.5)
    upper = np.percentile(boot_preds, 97.5)

    st.success(f"💰 Estimated Sale Price: ${central:,.0f}")
    st.markdown(f"📉 95% Confidence Range: **${lower:,.0f} – ${upper:,.0f}**")

    st.caption("""
    🔍 **What does this mean?**  
    The confidence range shows where the actual sale price is *likely* to fall based on the model.  
    Think of it as the "safe zone" — your house might sell for more or less,  
    but most similar houses with these features sold in this range.
    """)

# 📊 Optional: Feature Importance (in sidebar)
st.sidebar.header("📈 Top Feature Importance")
try:
    importances = model.feature_importances_
    top_k = pd.Series(importances, index=expected_features).sort_values(ascending=False).head(10)
    st.sidebar.bar_chart(top_k)
except AttributeError:
    st.sidebar.warning("Feature importance not available for this model.")
