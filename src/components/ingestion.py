import os
import sys
from pandas import read_csv
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exceptions import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig


@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion settings.

    Attributes:
        train_data_path (str): File path for saving the training dataset.
        test_data_path (str): File path for saving the testing dataset.
        raw_data_path (str): File path for saving the raw dataset.
    """
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    """
    Class for handling the data ingestion process in a machine learning pipeline.

    Attributes:
        ingestion_config (DataIngestionConfig): Configuration instance for data ingestion.
    """

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initialize_data_ingestion(self):
        """
        Initializes the data ingestion process by loading, splitting, and saving the dataset.

        Returns:
            tuple: A tuple containing the paths to the saved train and test datasets.

        Raises:
            CustomException: For handling exceptions that occur during the data ingestion process.
        """
        logging.info('Initializing Data Ingestion...')
        try:
            csv_file_path = 'notebook/data/gemstone.csv'
            if not os.path.exists(csv_file_path):
                raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

            df = read_csv(csv_file_path)
            logging.info('CSV file loaded!')

            os.makedirs(os.path.dirname(
                self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,
                      index=False, header=True)
            logging.info('Train-Test-Split Initiated')

            train_data, test_data = train_test_split(
                df, test_size=0.2, shuffle=True, random_state=42)
            train_data.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,
                             index=False, header=True)

            logging.info('Ingestion completed!')
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initialize_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data_path, test_data_path)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initialize_model_trainer(train_arr, test_arr))

    # if os.path.exists(train_data_path) and os.path.exists(test_data_path):
    #     data_transformation = DataTransformation()
    #     data_transformation.initialize_data_transformation(
    #         train_data_path, test_data_path)
    #     logging.info('Data transformation process completed successfully.')
    # else:
    #     logging.error('Train or Test data path does not exist.')
