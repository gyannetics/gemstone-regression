import os
import sys
from dataclasses import dataclass
from typing import Tuple

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    """
    Configuration for ModelTrainer.

    Attributes:
        trained_model_file_path (str): Path to save the trained model.
    """
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    """
    This class is responsible for training various regression models and identifying 
    the best performing model based on R2 score.

    Attributes:
        model_trainer_config (ModelTrainerConfig): Configuration instance for ModelTrainer.
    """

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initialize_model_trainer(self, train_array, test_array) -> Tuple[str, float]:
        """
        Trains multiple regression models defined in the 'models' dictionary, evaluates them, 
        and selects the best model based on the R2 score. The best model is then saved to a file.

        Args:
            train_array (np.array): Training dataset (features and target).
            test_array (np.array): Test dataset (features and target).

        Returns:
            Tuple[str, float]: The name of the best model and its R2 score on the test dataset.

        Raises:
            CustomException: For handling exceptions that occur during the model training process.
        """
        try:
            logging.info('Splitting Train and Test input data')
            # Splitting the training array into features and target variable
            x_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            # Splitting the testing array into features and target variable
            x_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor()
            }
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "K-Neighbors Regressor": {
                    'n_neighbors': [1, 3, 5, 10, 20],
                    'weights': ['uniform', 'distance']
                },
                "Random Forest Regressor": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],

                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGB Regressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }

            }
            model_report = evaluate_models(
                X_train=x_train,
                y_train=y_train,
                X_test=x_test,
                y_test=y_test,
                models=models,
                params=params
            )
            # Finding the best model based on the test R2 score
            best_model_name, best_model_metrics = max(
                model_report.items(), key=lambda x: x[1]['test_score'])
            best_model_score = best_model_metrics['test_score']
            # Get the trained model instance
            best_trained_model = best_model_metrics['best_estimator']

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(
                f"Best model found: {best_model_name} with score: {best_model_score * 100:.5f}%")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_trained_model  # Save the trained model
            )
            logging.info("Model file saved successfully!")
            # Use the trained model for prediction
            predicted = best_trained_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)

            return best_model_name, r2_square

        except Exception as e:
            raise CustomException(e, sys)


# Example usage:
# trainer = ModelTrainer()
# best_model_name, r2_square = trainer.initialize_model_trainer(train_array, test_array)
