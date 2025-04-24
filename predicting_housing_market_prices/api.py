from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model
model = joblib.load("models/final_model.joblib")

# Define input schema
class PredictionInput(BaseModel):
    houses_asking_price_lag1: float
    appartments_asking_price_lag1: float
    houses_asking_growth: float
    apartment_buildings_transaction_price_lag2: float
    ch_net_migration_lag1: float
    currency_in_circulation_change_lag1: float
    appartments_asking_growth: float
    appartments_transaction_growth: float
    monetary_aggregate_m3_change_lag1: float
    banks_foreign_loans_utilisation_lag2: float
    apartment_buildings_transaction_price_lag1: float
    inflation_lag2: float

@app.post("/predict")
def predict(input: PredictionInput):
    # Convert input to DataFrame
    data = pd.DataFrame([input.dict()])
    # Predict
    prediction = model.predict(data)[0]
    return {"predicted_house_asking_price": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)