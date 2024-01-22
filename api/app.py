# 1. Library imports
import uvicorn
from fastapi import FastAPI
import pickle
import pandas as pd
from xgboost import XGBClassifier
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 2. Create app and model objects
app = FastAPI()

model_path ='C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/mlartifacts/675174994878903245/f160483f83af458da90360f354519e24/artifacts/XGBClassifier_model__bayes_search__randomundersampler_balancing__debug_run'
model = mlflow.sklearn.load_model(model_path)

@app.post('/predict_from_dict')
def predict_credit_risk(client_infos: dict):
    client_infos = pd.DataFrame.from_dict([client_infos])
    prediction_array, confidence_array = model.predict(client_infos), model.predict_proba(client_infos)
    prediction_value = int(prediction_array[0])
    confidence_value = float(confidence_array[0][prediction_value])
    if confidence_value > 0.8:
        risk_category = "SAFE" if prediction_value == 0 else "NOPE"
    elif confidence_value > 0.6:
        risk_category = "UNCERTAIN" if prediction_value == 0 else "VERY RISKY"
    else:
        risk_category = "RISKY"
    return {
        'prediction': prediction_value,
        'confidence': confidence_value,
        'risk_category': risk_category
    }

@app.post('/global_feature_importance')
def get_global_feature_importance():
    if isinstance(model, XGBClassifier):
        feature_importance_dict = {
            'model_type': 'XGBClassifier',
            'feature_importance': {
                'weight': model.get_booster().get_score(importance_type='weight'),
                'cover': model.get_booster().get_score(importance_type='cover'),
                'gain': model.get_booster().get_score(importance_type='gain')
            }
        }
    elif isinstance(model, RandomForestClassifier):
        feature_importance_dict = {
            'model_type': 'RandomForestClassifier',
            'feature_importance': model.feature_importances_
        }
    return feature_importance_dict

# 4. Run the API with uvicorn (uvicorn app:app --reload)
# first app stands for the pyhton file, second app for the API instance, --reload for automatic refresh
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)