import os
import sys
sys.path.append('F:/project/mlproject')
from dataclasses import dataclass

from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerconfig:
    train_model_file_path=os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerconfig()
        
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('split train and test input data')
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'XGBoost Regressor': XGBRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor()
            }
            
            model_report: dict = evaluate_models(
                x_train=x_train, 
                y_train=y_train, 
                x_test=x_test, 
                y_test=y_test, 
                models=models
            )
            
            # Get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            # Get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException(f'No best model found. Best score: {best_model_score}')
            
            logging.info(f'Best model found on both training and testing dataset: {best_model_name}')
            
            # Refit the best model on training data
            best_model.fit(x_train, y_train)
            
            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(x_test)
            r2_accuracy = r2_score(y_test, predicted)
            
            return r2_accuracy
            
        except Exception as e:
            raise CustomException(e, sys)