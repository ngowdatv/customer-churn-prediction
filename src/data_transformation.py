import os
import sys
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


class DataTransformation:

    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join(
            "artifacts",
            "preprocessor.pkl"
        )

    def get_data_transformer_object(self):

        try:

            numerical_columns = [
                "SeniorCitizen",
                "tenure",
                "MonthlyCharges",
                "TotalCharges"
            ]

            categorical_columns = [
                "gender",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logger.info("Numerical Pipeline Created")
            logger.info("Categorical Pipeline Created")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:

            logger.info("Reading Train and Test Data")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # -----------------------------
            # Clean TotalCharges
            # -----------------------------
            train_df["TotalCharges"] = (
                train_df["TotalCharges"]
                .astype(str)
                .str.strip()
                .replace("", np.nan)
            )

            test_df["TotalCharges"] = (
                test_df["TotalCharges"]
                .astype(str)
                .str.strip()
                .replace("", np.nan)
            )

            train_df["TotalCharges"] = pd.to_numeric(
                train_df["TotalCharges"],
                errors="coerce"
            )

            test_df["TotalCharges"] = pd.to_numeric(
                test_df["TotalCharges"],
                errors="coerce"
            )

            print("Train dtype :", train_df["TotalCharges"].dtype)
            print("Test dtype  :", test_df["TotalCharges"].dtype)

            # -----------------------------
            # Drop customerID
            # -----------------------------
            train_df.drop(columns=["customerID"], inplace=True)
            test_df.drop(columns=["customerID"], inplace=True)

            target_column = "Churn"

            # -----------------------------
            # Features
            # -----------------------------
            X_train = train_df.drop(columns=[target_column])
            X_test = test_df.drop(columns=[target_column])

            # -----------------------------
            # Target Encoding
            # -----------------------------
            y_train = train_df[target_column].map({
                "No": 0,
                "Yes": 1
            })

            y_test = test_df[target_column].map({
                "No": 0,
                "Yes": 1
            })

            preprocessing_obj = self.get_data_transformer_object()

            logger.info("Applying preprocessing")

            X_train = preprocessing_obj.fit_transform(X_train)
            X_test = preprocessing_obj.transform(X_test)

            print("X Train Shape :", X_train.shape)
            print("X Test Shape  :", X_test.shape)

            save_object(
                file_path=self.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            print(
                "Preprocessor Saved :",
                self.preprocessor_obj_file_path
            )

            logger.info("Data Transformation Completed")

            return (
                X_train,
                X_test,
                y_train,
                y_test,
                self.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)