from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Dummy training (replace with your actual model later)
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
data = data.dropna()

data['Churn'] = data['Churn'].map({'Yes':1, 'No':0})
X = pd.get_dummies(data.drop(['customerID','Churn'], axis=1))
y = data['Churn']

model = RandomForestClassifier()
model.fit(X, y)

@app.route("/")
def home():
    return "Churn Model Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.json
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    return jsonify({"churn": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)