import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==============================
# MLflow Configuration
# ==============================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Customer_Churn_Prediction")

# ==============================
# Load Dataset
# ==============================
data = pd.read_csv("churn.csv/customer_churn.csv")
data = data.dropna()

# Encode Target
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

# Features & Target
X = pd.get_dummies(data.drop(["customerID", "Churn"], axis=1))
y = data["Churn"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# ==============================
# MLflow Run
# ==============================
with mlflow.start_run():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Parameters
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    # Metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Save Model
    joblib.dump(model, "model.pkl")
    joblib.dump(list(X.columns), "columns.pkl")

    # Log Model
    mlflow.sklearn.log_model(
        sk_model=model,
        name="RandomForest_Model"
    )

    print("\n==============================")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("==============================")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

print("\n✅ Training Completed Successfully")