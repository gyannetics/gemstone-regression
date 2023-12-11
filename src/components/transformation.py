from dataclasses import dataclass
import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object
from typing import Tuple

@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation settings.

    Attributes:
        preprocessor_obj_file_path (str): File path for saving the preprocessor object.
        target_column_name (str): Name of the target column in the dataset.
        numerical_columns (list): List of names of numerical columns.
        categorical_columns (list): List of names of categorical columns.
    """
    preprocessor_obj_file_path: str = os.path.join(
        'artifacts', 'preprocessor.pkl')
    target_column_name: str = 'math_score'
    numerical_columns: list = None
    categorical_columns: list = None

    def __post_init__(self):
        if self.numerical_columns is None:
            self.numerical_columns = ['reading_score', 'writing_score']
        if self.categorical_columns is None:
            self.categorical_columns = [
                'gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course'
            ]


class DataTransformation:
    """
    Class for handling the data transformation process in a machine learning pipeline.

    Attributes:
        config (DataTransformationConfig): Configuration instance for data transformation.
    """

    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Creates and returns a data transformation pipeline for preprocessing numerical and categorical data.

        Returns:
            ColumnTransformer: A transformer for preprocessing data with specified numerical and categorical pipelines.
        """
        try:
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info("Setting up data transformation pipelines.")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, self.config.numerical_columns),
                    ("cat_pipeline", cat_pipeline, self.config.categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initialize_data_transformation(self, train_path, test_path) -> Tuple:
        """
        Initializes and applies the data transformation on the provided training and testing datasets.

        Args:
            train_path (str): File path to the training dataset.
            test_path (str): File path to the testing dataset.

        Returns:
            tuple: A tuple containing transformed training array, testing array, and path to the saved preprocessor object.

        Raises:
            FileNotFoundError: If the provided file paths for train or test datasets are not found.
            CustomException: For handling exceptions that occur during the data transformation process.
        """
        try:
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                raise FileNotFoundError(
                    f"Data file not found: {train_path} or {test_path}")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and Test data loaded successfully.")

            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_df = train_df.drop(
                columns=[self.config.target_column_name], axis=1)
            target_feature_train_df = train_df[self.config.target_column_name]

            input_feature_test_df = test_df.drop(
                columns=[self.config.target_column_name], axis=1)
            target_feature_test_df = test_df[self.config.target_column_name]

            logging.info(
                "Applying preprocessing pipeline to training and testing data.")
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,
                              np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,
                             np.array(target_feature_test_df)]

            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info(
                "Preprocessing object saved and data transformation completed.")

            return (
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

# Example usage
# transformation = DataTransformation()
# train_array, test_array, preprocessor_path = transformation.initialize_data_transformation(train_path, test_path)
