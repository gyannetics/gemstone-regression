import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Configuration class for Data Transformation.
    Contains file path for the preprocessor object.
    """
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """
    A class for transforming raw data into a format suitable for model training.
    It handles preprocessing tasks like imputation, scaling, and encoding.
    """

    def __init__(self):
        """
        Initializes the DataTransformation with necessary configuration.
        """
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        """
        Creates a data preprocessing pipeline which includes imputation, encoding,
        and scaling for both numerical and categorical columns.

        Returns
        -------
        preprocessor: ColumnTransformer
            A preprocessing object configured for both numerical and categorical data.
        """
        try:
            # Define columns for different types of transformations
            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

            # Define custom categories for ordinal encoding
            cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            # Create pipelines for numerical and categorical processing
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())                
            ])
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal_encoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                ('scaler', StandardScaler())
            ])

            # Combine pipelines into a single preprocessor
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            return preprocessor
        
        except Exception as e:
            logging.error('Error during data transformation object creation: %s', e)
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        Applies the preprocessing steps on the training and testing datasets.

        Parameters
        ----------
        train_path : str
            Path to the training dataset file.
        test_path : str
            Path to the testing dataset file.

        Returns
        -------
        tuple
            A tuple containing processed training array, testing array,
            and the path to the saved preprocessor object.
        """
        try:
            # Load the datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Train and test data loaded successfully.')

            # Obtain the preprocessing object
            preprocessing_obj = self.get_data_transformation_object()
            logging.info('Preprocessing object obtained.')

            # Prepare the datasets
            target_column_name = 'price'
            drop_columns = [target_column_name, 'id']
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Apply transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine features with target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor object
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)
            logging.info('Preprocessor object saved.')

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        
        except Exception as e:
            logging.error('Error in initiate_data_transformation: %s', e)
            raise CustomException(e, sys)

# if __name__ == '__main__':
#     # Configuration and initiation of data transformation
#     data_transformation = DataTransformation()
#     train_arr, test_arr, _ = data_transformation.initiate_data_transformation('path_to_train.csv', 'path_to_test.csv')
