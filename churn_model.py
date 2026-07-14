import boto3
import pandas as pd
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. LOAD DATA FROM S3
# -----------------------------
bucket = "customer-churn-bucket-12345"
key = "customer_churn.csv.csv"

s3 = boto3.client("s3")

obj = s3.get_object(Bucket=bucket, Key=key)
data = obj["Body"].read().decode("utf-8")

df = pd.read_csv(StringIO(data))

print("Data Loaded")
print(df.head())

# -----------------------------
# 2. DATA CLEANING
# -----------------------------

# Fix TotalCharges column (important)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop missing values
df = df.dropna()

# Drop useless column
df = df.drop("customerID", axis=1)

# -----------------------------
# 3. FEATURES & TARGET
# -----------------------------
target_column = "Churn"

X = df.drop(target_column, axis=1)
y = df[target_column]

# Convert categorical → numeric
X = pd.get_dummies(X)

# -----------------------------
# 4. TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. MODEL (IMPROVED)
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 6. PREDICTION
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 7. ACCURACY
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n !!!!!!! Model Accuracy:", accuracy)

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# CHURN COUNT
# -----------------------------
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

# -----------------------------
# CHURN vs CONTRACT
# -----------------------------
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Churn by Contract Type")
plt.xticks(rotation=30)
plt.show()

# -----------------------------
# CHURN vs PAYMENT METHOD
# -----------------------------
sns.countplot(x='PaymentMethod', hue='Churn', data=df)
plt.title("Churn by Payment Method")
plt.xticks(rotation=45)
plt.show()

import matplotlib.pyplot as plt

importance = model.feature_importances_
features = X.columns

plt.barh(features, importance)
plt.title("Feature Importance")
plt.show()