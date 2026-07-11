import os
import sys
from dataclasses import dataclass

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

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

                mlflow.log_param("Best Model", best_model_name)

                mlflow.log_metric("Accuracy", best_model_score)

                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="model"
                )

                print(f"Best Model: {best_model_name}")
                print(f"Accuracy : {best_model_score}")

                return best_model_score

        except Exception as e:
            raise CustomException(e, sys)