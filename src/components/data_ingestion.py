import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    """
    Data Ingestion Configuration Class
    Holds the file paths for the raw, train, and test datasets.
    """
    raw_data_path: str = os.path.join('artifacts', 'data', 'gemstone.csv')
    train_data_path: str = os.path.join('artifacts', 'data', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'data', 'test.csv')

class DataIngestion:
    """
    A class used to handle the data ingestion process.

    Attributes
    ----------
    config : DataIngestionConfig
        The configuration object containing file paths.

    Methods
    -------
    initiate_data_ingestion():
        Reads the raw dataset, splits it into training and testing sets,
        and saves them to specified paths.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Parameters
        ----------
        config : DataIngestionConfig
            An object containing configuration settings for data ingestion.
        """
        self.config = config

    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process.

        Reads the dataset from the raw data path, splits it into training and testing datasets,
        and saves them to their respective paths as specified in the configuration.

        Returns
        -------
        tuple
            A tuple containing paths to the train and test data files.
        """
        logging.info('Starting data ingestion process.')
        try:
            df = pd.read_csv(self.config.raw_data_path)
            logging.info('Dataset loaded into pandas DataFrame.')

            # Creating directories for storing processed data
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.test_data_path), exist_ok=True)

            # Splitting the dataset into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)

            logging.info('Data ingestion completed successfully.')
            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            logging.error(f'Error during data ingestion: {e}')
            raise CustomException(e, sys)

if __name__ == '__main__':
    # Main execution block
    # Set up configuration, initiate data ingestion, transformation, and model training
    config = DataIngestionConfig()
    data_ingestion = DataIngestion(config)
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    # Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    # Model Training
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr, test_arr)
