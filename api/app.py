# 1. Library imports
import uvicorn
from fastapi import FastAPI
import pickle
import pandas as pd
from xgboost import XGBClassifier
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np

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
    credit_risk = 'Risky' if prediction_value == 1 else 'Safe'
    return {
        'prediction': prediction_value,
        'confidence': confidence_value,
        'credit_approval': credit_risk
    }

# 4. Run the API with uvicorn (uvicorn app:app --reload)
# first app stands for the pyhton file, second app for the API instance, --reload for automatic refresh
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)