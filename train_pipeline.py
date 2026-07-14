from src.data_ingestion import DataIngestion
from src.data_transformation import DataTransformation
from src.model_trainer import ModelTrainer


if __name__ == "__main__":

    ingestion = DataIngestion()

    train_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()

    X_train, X_test, y_train, y_test, _ = transformation.initiate_data_transformation(
        train_path,
        test_path
    )

    trainer = ModelTrainer()

    trainer.initiate_model_trainer(
        X_train,
        X_test,
        y_train,
        y_test
    )