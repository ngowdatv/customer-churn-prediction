from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI(
    title="Customer Churn Prediction API",
    description="ML API to predict customer churn",
    version="1.0"
)


# Load ML model and preprocessing pipeline
model = joblib.load("artifacts/model.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")


# Input schema
class CustomerData(BaseModel):
    customerID: str
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


# Home route
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is Running"
    }


# Prediction route
@app.post("/predict")
def predict(data: CustomerData):

    input_data = pd.DataFrame([data.model_dump()])

    processed_data = preprocessor.transform(input_data)

    prediction = model.predict(processed_data)

    result = prediction[0]

    return {
        "prediction": str(result),
        "result": "Customer will churn" if result == 1 else "Customer will stay"
    }