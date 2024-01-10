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

# with open('C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/mlartifacts/984947130656132807/83c10514a1df42e5ba3b30f93971a0ea/artifacts/XGBClassifier_model__bayes_search__randomundersampler_balancing__debug_run/model.pkl', 'rb') as file:
#     model = pickle.load(file)

model_path ='C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/mlartifacts/675174994878903245/f160483f83af458da90360f354519e24/artifacts/XGBClassifier_model__bayes_search__randomundersampler_balancing__debug_run'
model = mlflow.sklearn.load_model(model_path)

with open('C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/train_df_debug.csv', 'rb') as file:
    data = pd.read_csv(file)

# client_id = 108567

# client_infos = data[data['SK_ID_CURR'] == client_id]
# client_infos = client_infos.drop(columns=['SK_ID_CURR', 'index', 'TARGET'])

data.set_index('SK_ID_CURR', inplace=True)
data.drop(columns=['index', 'TARGET'], inplace=True)

# data_dict = data.to_dict(orient='index')

# def predict_risk(self, client_infos):
#     data_in = [[sepal_length, sepal_width, petal_length, petal_length]]
#     prediction = self.model.predict(data_in)
#     probability = self.model.predict_proba(data_in).max()
#     return prediction[0], probability

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
@app.post('/predict')
def predict_credit_risk(client_id: int):
    client_infos = data[data.index == client_id]
    # client_infos = data_dict[client_id]
    # client_infos = pd.DataFrame(client_infos, index=[client_id])
    prediction_array, confidence_array = model.predict(client_infos), model.predict_proba(client_infos)
    prediction_value = int(prediction_array[0])
    confidence_value = float(confidence_array[0][prediction_value])
    # prediction, probability = model.predict(client_infos), model.predict_proba(client_infos)
    credit_risk = 'Risky' if prediction_value == 1 else 'Safe'
    return {
        'prediction': prediction_value,
        'confidence': confidence_value,
        'credit_approval': credit_risk
    }

@app.post('/client_infos')
def return_clients_infos(client_id: int):
    client_infos = data[data.index == client_id]
    client_infos_dict = client_infos.to_dict(orient='records')
    return client_infos_dict

# @app.post('/print_dict')
# def get_received_dict(data_dict: dict):
#     print(data_dict)  # Printing the received dictionary
#     return {'message': 'Dictionary printed in the console'}

# 4. Run the API with uvicorn (uvicorn app:app --reload)
# first app stands for the pyhton file, second app for the API instance, --reload for automatic refresh
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)