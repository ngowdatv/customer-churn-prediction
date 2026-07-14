from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a customer will churn",
    version="1.0"
)

# Load trained model
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")


# Input Schema
class ChurnInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API Running 🚀"}


@app.post("/predict")
def predict(data: ChurnInput):

    input_df = pd.DataFrame([data.model_dump()])

    # One-hot encoding
    input_df = pd.get_dummies(input_df)

    # Match training columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Prediction
    prediction = model.predict(input_df)[0]

    result = "Churn" if prediction == 1 else "No Churn"

    return {
        "Prediction": result,
        "Prediction Value": int(prediction)
    }