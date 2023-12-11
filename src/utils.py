import os
import sys
from typing import Dict, Any
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
from src.exceptions import CustomException
from src.logger import logging
import time
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    """
    Save a Python object to a file using dill.

    Args:
        file_path (str): The path to the file where the object should be saved.
        obj: The Python object to save.

    Raises:
        CustomException: For handling exceptions that occur during object saving.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, 
                    models: Dict[str, Any], 
                    params: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate multiple models using GridSearchCV and return their performance scores.

    Args:
        X_train: Training data features.
        y_train: Training data labels.
        X_test: Test data features.
        y_test: Test data labels.
        models: Dictionary of model names and their instances.
        params: Dictionary of model names and their corresponding parameter grid for GridSearchCV.

    Returns:
        A dictionary with model names and their respective test R2 scores.

    Raises:
        CustomException: For handling exceptions that occur during the model evaluation.
    """
    try:
        report = {}

        for model_name, model in tqdm(models.items(), desc="Training models"):
            start_time = time.time()  # Start time
            logging.info(f'Tuning with Hyper Parameters for {model_name}')
            param_grid = params.get(model_name, {})
            gs = GridSearchCV(model, param_grid, cv=3)
            gs.fit(X_train, y_train)

            end_time = time.time()  # End time
            training_time = end_time - start_time  # Calculate training time

            best_model = gs.best_estimator_

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = {
                'train_score': train_model_score,
                'test_score': test_model_score,
                'best_params': gs.best_params_,
                'training_time': training_time,  # Add training time to the report
                'best_estimator': best_model
            }
            # Display current model name
            tqdm.write(
                f"Completed training {model_name}. Time taken: {training_time:.2f} seconds")
            logging.info(
                f"Completed training {model_name}. Time taken: {training_time:.2f} seconds")

        return report

    except Exception as e:
        raise CustomException(e, sys)


def model_metrics(true, predicted):
    try:
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mse)
        r2_square = r2_score(true, predicted)
        return mae, rmse, r2_square
    except Exception as e:
        logging.info('Exception Occured while evaluating metric')
        raise CustomException(e, sys)


def print_evaluated_results(xtrain, ytrain, xtest, ytest, model):
    try:
        ytrain_pred = model.predict(xtrain)
        ytest_pred = model.predict(xtest)

        # Evaluate Train and Test dataset
        model_train_mae, model_train_rmse, model_train_r2 = model_metrics(
            ytrain, ytrain_pred)
        model_test_mae, model_test_rmse, model_test_r2 = model_metrics(
            ytest, ytest_pred)

        # Printing results
        print('Model performance for Training set')
        print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
        print("- R2 Score: {:.4f}".format(model_train_r2))

        print('----------------------------------')

        print('Model performance for Test set')
        print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
        print("- R2 Score: {:.4f}".format(model_test_r2))

    except Exception as e:
        logging.info('Exception occured during printing of evaluated results')
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e, sys)
