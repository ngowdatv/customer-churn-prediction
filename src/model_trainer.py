import os
import sys
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)

from src.exception import CustomException
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        try:

            mlflow.set_experiment("Customer_Churn")

            with mlflow.start_run():

                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Random Forest": RandomForestClassifier(),
                    "Gradient Boosting": GradientBoostingClassifier(),
                    "XGBoost": XGBClassifier(
                        use_label_encoder=False,
                        eval_metric="logloss"
                    )
                }

                model_report = evaluate_model(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    models
                )

                best_model_score = max(model_report.values())

                best_model_name = list(model_report.keys())[
                    list(model_report.values()).index(best_model_score)
                ]

                best_model = models[best_model_name]

                save_object(
                    self.model_trainer_config.trained_model_file_path,
                    best_model
                )

                # Predictions
                y_pred = best_model.predict(X_test)

                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred)

                # Parameters
                mlflow.log_param("Best Model", best_model_name)
                mlflow.log_param("Train Samples", len(X_train))
                mlflow.log_param("Test Samples", len(X_test))
                mlflow.log_param("Features", X_train.shape[1])

                # Metrics
                mlflow.log_metric("Accuracy", accuracy)
                mlflow.log_metric("Precision", precision)
                mlflow.log_metric("Recall", recall)
                mlflow.log_metric("F1 Score", f1)
                mlflow.log_metric("ROC AUC", roc_auc)

                # Confusion Matrix
                disp = ConfusionMatrixDisplay.from_predictions(
                    y_test,
                    y_pred
                )
                plt.savefig("confusion_matrix.png")
                plt.close()

                mlflow.log_artifact("confusion_matrix.png")

                # Log Model
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="model"
                )

                print(f"\nBest Model : {best_model_name}")
                print(f"Accuracy   : {accuracy:.4f}")
                print(f"Precision  : {precision:.4f}")
                print(f"Recall     : {recall:.4f}")
                print(f"F1 Score   : {f1:.4f}")
                print(f"ROC AUC    : {roc_auc:.4f}")

                return accuracy

        except Exception as e:
            raise CustomException(e, sys)