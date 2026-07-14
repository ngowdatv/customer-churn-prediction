import joblib
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("churn.csv/customer_churn.csv")
data = data.dropna()

# Convert target
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

# Prepare features
X = pd.get_dummies(data.drop(["customerID", "Churn"], axis=1))
y = data["Churn"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model and feature names
joblib.dump(model, "model.pkl")
joblib.dump(list(X.columns), "columns.pkl")

print("✅ Model saved successfully!")