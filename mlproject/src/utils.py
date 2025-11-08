import os
import sys
sys.path.append('F:/project/mlproject')
from src.exception import CustomException
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train, y_train, x_test, y_test, models):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            
            # FIX: Use explicit keyword arguments with capital X
            model.fit(X=x_train, y=y_train)
            
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score
            
        return report
        
    except Exception as e:
        raise CustomException(e, sys)