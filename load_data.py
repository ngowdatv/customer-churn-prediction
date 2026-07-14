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

print("✅ Data Loaded")
print(df.head())

# -----------------------------
# 2. BASIC INFO
# -----------------------------
print("\nShape:", df.shape)
print("\nColumns:", df.columns)

# -----------------------------
# 3. HANDLE MISSING VALUES
# -----------------------------
df = df.dropna()

# -----------------------------
# 4. FIND TARGET COLUMN
# -----------------------------
# Change this if your column name is different
target_column = "Churn"

if target_column not in df.columns:
    print("❌ 'Churn' column not found. Available columns:")
    print(df.columns)
    exit()

# -----------------------------
# 5. SPLIT DATA
# -----------------------------
X = df.drop(target_column, axis=1)
y = df[target_column]

# Convert categorical → numeric
X = pd.get_dummies(X)

# -----------------------------
# 6. TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 7. MODEL TRAINING
# -----------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# -----------------------------
# 8. PREDICTION
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 9. ACCURACY
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n🎯 Model Accuracy:", accuracy)