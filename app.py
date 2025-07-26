import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load('random_forest_model.pkl')

# Load expected feature list
expected_features = pd.read_csv("expected_features.csv", header=None)[0].tolist()

# App title
st.title("🏡 House Price Estimator")

st.write("Fill in details below. We'll estimate the sale price of the house!")

# Sample input fields (based on the most important features)
OverallQual = st.slider("Overall Quality (1–10)", 1, 10, 5)
GrLivArea = st.number_input("Above Ground Living Area (sq ft)", 500, 5000, 1500)
TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", 0, 3000, 800)
SecondFlrSF = st.number_input("2nd Floor Area (sq ft)", 0, 3000, 500)
BsmtFinSF1 = st.number_input("Finished Basement Area (sq ft)", 0, 2000, 400)

# 🔧 Create a full input row with default zeros
input_data = pd.DataFrame([np.zeros(len(expected_features))], columns=expected_features)

# ✍️ Replace just a few known values
input_data["OverallQual"] = OverallQual
input_data["GrLivArea"] = GrLivArea
input_data["TotalBsmtSF"] = TotalBsmtSF
input_data["SecondFlrSF"] = SecondFlrSF
input_data["BsmtFinSF1"] = BsmtFinSF1

# 🧠 Predict when user clicks
if st.button("Estimate Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Sale Price: ${prediction:,.2f}")


