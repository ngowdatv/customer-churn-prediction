from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and feature columns
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")


@app.route("/")
def home():
    return "Churn Model Running 🚀"


@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.json

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    return jsonify({"churn": int(prediction)})


if __name__ == "__main__":
    app.run(debug=True)