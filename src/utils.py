import os
import pickle
import numpy as np

from sklearn.metrics import accuracy_score


def save_object(file_path, obj):
    """
    Save any Python object as a pickle file.
    """

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)


def load_object(file_path):
    """
    Load a pickle object.
    """

    with open(file_path, "rb") as file_obj:
        return pickle.load(file_obj)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Train multiple models and return their accuracies.
    """

    report = {}

    for model_name, model in models.items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        score = accuracy_score(y_test, y_pred)

        report[model_name] = score

    return report