from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a customer will churn",
    version="1.0"
)


# Load model
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")


# Input parameters shown in Swagger
class ChurnInput(BaseModel):

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str

    tenure: int

    InternetService: str
    Contract: str

    MonthlyCharges: float
    TotalCharges: float

    PaymentMethod: str



@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API Running 🚀"
    }



@app.post("/predict")
def predict(data: ChurnInput):

    # Convert input to dataframe
    input_df = pd.DataFrame(
        [data.model_dump()]
    )


    # Encoding
    input_df = pd.get_dummies(input_df)


    # Match model training columns
    input_df = input_df.reindex(
        columns=columns,
        fill_value=0
    )


    # Prediction
    prediction = model.predict(input_df)[0]


    result = "Churn" if prediction == 1 else "No Churn"


    return {
        "Prediction": result,
        "Prediction_Value": int(prediction)
    }