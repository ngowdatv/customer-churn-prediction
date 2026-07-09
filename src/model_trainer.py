import os
import sys

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logger
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

            print("\nModel Accuracy\n")

            for name, score in model_report.items():
                print(f"{name} : {score:.4f}")

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print("\nBest Model :", best_model_name)
            print("Accuracy   :", best_model_score)

            save_object(
                self.model_trainer_config.trained_model_file_path,
                best_model
            )

            print(
                "\nModel Saved :",
                self.model_trainer_config.trained_model_file_path
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)