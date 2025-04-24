import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.title("Swiss House Price Predictor")

# Input sliders for key features
st.header("Input Features")
houses_lag1 = st.slider("Houses Asking Price (Lag 1)", 80.0, 200.0, 160.0)
appartments_lag1 = st.slider("Apartments Asking Price (Lag 1)", 80.0, 200.0, 180.0)
# ... Add sliders for all 12 features

# Prepare input
input_data = {
    "houses_asking_price_lag1": houses_lag1,
    "appartments_asking_price_lag1": appartments_lag1,
    # ... Other features
}

# Call API
if st.button("Predict"):
    response = requests.post("http://<pi-ip>:8000/predict", json=input_data)
    prediction = response.json()["predicted_house_asking_price"]
    st.success(f"Predicted House Asking Price: {prediction:.2f}")

# Display MLflow results
st.header("Model Performance")
st.write("Initial Model: Test R² = 0.83")
st.write("Enhanced Model: Test R² = 0.83 (no new features selected)")
st.image("plots/predictions_plot.png", caption="Predictions vs. Actual")

if __name__ == "__main__":
    st.run()