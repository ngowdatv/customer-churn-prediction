from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load model
model = pickle.load(open("artifacts/model.pkl", "rb"))
preprocessor = pickle.load(open("artifacts/preprocessor.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "Customer Churn Model Running 🚀"}

@app.post("/predict")
def predict(data: list):
    data = np.array(data).reshape(1, -1)
    data = preprocessor.transform(data)
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}