import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logger


class DataIngestion:

    def __init__(self):
        self.raw_data_path = os.path.join("artifacts", "raw.csv")
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.test_data_path = os.path.join("artifacts", "test.csv")

    def initiate_data_ingestion(self):
        logger.info("Entered Data Ingestion")

        try:
            df = pd.read_csv("data/raw/customer_churn.csv")

            os.makedirs("artifacts", exist_ok=True)

            df.to_csv(self.raw_data_path, index=False)

            logger.info("Raw Data Saved")

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)

            logger.info("Train-Test Split Completed")

            return (
                self.train_data_path,
                self.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)