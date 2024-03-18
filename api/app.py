# 1. Library imports
import uvicorn
from fastapi import FastAPI
import pickle
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap
import os
from dotenv import load_dotenv

# 2. Create app and model objects
app = FastAPI()

load_dotenv('api.env')

MLFLOW_TRACKING_USERNAME = os.getenv('MLFLOW_TRACKING_USERNAME')
MLFLOW_TRACKING_PASSWORD = os.getenv('MLFLOW_TRACKING_PASSWORD')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_MODEL_URI = os.getenv('MLFLOW_MODEL_URI')
MLFLOW_RUN_ID = os.getenv('MLFLOW_RUN_ID')

model_path_dict = {
    # 'XGBClassifier': 'C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/mlartifacts/196890409778388257/411079214e254096a033a07452c930b4/artifacts/XGBClassifier_model__bayes_search__randomundersampler_balancing__tuning_run',
    # 'RandomForestClassifier': 'C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/mlartifacts/389240121052556420/45d26b172a2d4bb79fc64664c19a0910/artifacts/RandomForestClassifier_model__bayes_search__randomundersampler_balancing__tuning_run ',
    # 'LGBMClassifier': 'C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/mlartifacts/218309939986015518/ffe6693bbf44476384c6c2db9c6552b7/artifacts/LGBMClassifier_model__bayes_search__randomundersampler_balancing__tuning_run',
    # 'debug': 'C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/mlartifacts/675174994878903245/11ae21cdc87645cfb96491b38d5f1b49/artifacts/XGBClassifier_model__bayes_search__randomundersampler_balancing__debug_run',
    'remote': 'https://dagshub.com/zerippeur/mlflow-tracking.mlflow'
}

explainer_path_dict = {
    # 'XGBClassifier': 'C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/mlartifacts/196890409778388257/411079214e254096a033a07452c930b4/artifacts/XGBClassifier_model__bayes_search__randomundersampler_balancing__tuning_run/shap_explainer_XGBClassifier_version_28.pkl',
    # 'RandomForestClassifier': 'C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/mlartifacts/389240121052556420/45d26b172a2d4bb79fc64664c19a0910/artifacts/RandomForestClassifier_model__bayes_search__randomundersampler_balancing__tuning_run/shap_explainer_RandomForestClassifier_version_11.pkl',
    # 'LGBMClassifier': 'C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/mlartifacts/218309939986015518/ffe6693bbf44476384c6c2db9c6552b7/artifacts/LGBMClassifier_model__bayes_search__randomundersampler_balancing__tuning_run/shap_explainer_LGBMClassifier_version_5.pkl',
    # 'debug': 'C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/mlartifacts/675174994878903245/11ae21cdc87645cfb96491b38d5f1b49/artifacts/XGBClassifier_model__bayes_search__randomundersampler_balancing__debug_run/shap_explainer_XGBClassifier_version_26.pkl',
    'remote': 'mlflow-artifacts:/86617b78fa1941f9a1c21aa470029781/8482edf6873043f0901149c7752531ec/artifacts/shap_explainer_LGBMClassifier_version_5.pkl'
}

shap_values_path_dict = {
    # 'XGBClassifier': 'C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/mlartifacts/196890409778388257/411079214e254096a033a07452c930b4/artifacts/XGBClassifier_model__bayes_search__randomundersampler_balancing__tuning_run/shap_values_XGBClassifier_version_28.pkl',
    # 'RandomForestClassifier': 'C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/mlartifacts/389240121052556420/45d26b172a2d4bb79fc64664c19a0910/artifacts/RandomForestClassifier_model__bayes_search__randomundersampler_balancing__tuning_run/shap_values_RandomForestClassifier_version_11.pkl',
    # 'LGBMClassifier': 'C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/mlartifacts/218309939986015518/ffe6693bbf44476384c6c2db9c6552b7/artifacts/LGBMClassifier_model__bayes_search__randomundersampler_balancing__tuning_run/shap_values_LGBMClassifier_version_5.pkl',
    # 'debug': 'C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/mlartifacts/675174994878903245/11ae21cdc87645cfb96491b38d5f1b49/artifacts/XGBClassifier_model__bayes_search__randomundersampler_balancing__debug_run/shap_values_XGBClassifier_version_26.pkl',
    'remote': 'mlflow-artifacts:/86617b78fa1941f9a1c21aa470029781/8482edf6873043f0901149c7752531ec/artifacts/shap_values_LGBMClassifier_version_5.pkl'
}

mode = 'remote'

if mode == 'remote':
    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
    mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(model_uri=MLFLOW_MODEL_URI)
else:
    model_path = model_path_dict[mode]
    model = mlflow.sklearn.load_model(model_path)

api_data_for_shap_initiation = None
explainer = None

@app.get('/model_threshold')
async def get_model_threshold():
    threshold = float(mlflow.get_run(run_id=MLFLOW_RUN_ID).data.params['threshold'])
    threshold_dict = {
        'threshold': threshold
    }
    return threshold_dict

@app.get('/global_feature_importance')
async def get_global_feature_importance():
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
            'feature_importance': {
                'gini': dict(zip(model.feature_names_in_, model.feature_importances_))
            }
        }
    elif isinstance(model, LGBMClassifier):
        feature_importance_dict = {
            'model_type': 'LGBMClassifier',
            'feature_importance': {
                'gain': dict(zip(model.feature_name_, model.feature_importances_.tolist())),
            }
        }
    return feature_importance_dict

@app.post('/predict_from_dict')
def predict_credit_risk(prediction_dict: dict):
    client_infos = pd.DataFrame.from_dict([prediction_dict['client_infos']])
    threshold = prediction_dict['threshold']
    proba_repay_wo_risk = float(model.predict_proba(client_infos)[:,0][0])
    risk_prediction = 0 if proba_repay_wo_risk > 1 - threshold else 1
    if proba_repay_wo_risk > 1 - threshold:
        risk_category = "SAFE"
    elif proba_repay_wo_risk > 1 - threshold - 0.05:
        risk_category = "RISKY"
    else:
        risk_category = "NOPE"
    return {
        'prediction': risk_prediction,
        'confidence': proba_repay_wo_risk,
        'risk_category': risk_category
    }

@app.post('/initiate_shap_explainer')
async def initiate_shap_explainer(data_for_shap_initiation: dict):
    global api_data_for_shap_initiation
    global explainer
    global feature_names
    global shap_values_global
    if api_data_for_shap_initiation is None:
        if mode == 'remote':
            with open(mlflow.artifacts.download_artifacts(explainer_path_dict[mode]), 'rb') as f:
                explainer = pickle.load(f)
            # with open(mlflow.artifacts.download_artifacts(shap_values_path_dict[mode]), 'rb') as f:
                # shap_values_global = pickle.load(f)
            data = pd.DataFrame.from_dict(data_for_shap_initiation, orient='index')
            shap_values_global = explainer.shap_values(data)
        else:
            with open(explainer_path_dict[mode], 'rb') as f:
                explainer = pickle.load(f)  
            with open(shap_values_path_dict[mode], 'rb') as f:
                shap_values_global = pickle.load(f)
        if isinstance(model, XGBClassifier):
            feature_names = model.feature_names_in_.tolist()
            shap_values_global = shap_values_global.tolist()
        elif isinstance(model, RandomForestClassifier):
            feature_names = model.feature_names_in_.tolist()
            shap_values_global = shap_values_global[1].tolist()
        elif isinstance(model, LGBMClassifier):
            feature_names = model.feature_name_
            shap_values_global = shap_values_global[1].tolist()
    
    shap_values_dict = {
        'shap_values': shap_values_global,
        'feature_names': feature_names,
        'expected_value': None
    }
    return shap_values_dict

@app.post('/shap_feature_importance')
def get_shap_feature_importance(shap_feature_importance_dict: dict):
    feature_scale = shap_feature_importance_dict['feature_scale']
    if feature_scale == 'Global':
        shap_values = shap_values_global#.tolist() if isinstance(model, XGBClassifier) else shap_values_global[1].tolist()
        expected_value = None
    elif feature_scale == 'Local':
        client_infos = pd.DataFrame.from_dict(shap_feature_importance_dict['client_infos'], orient='index').T
        shap_values = explainer.shap_values(client_infos).tolist() if isinstance(model, XGBClassifier) else explainer.shap_values(client_infos)[1].tolist()
        expected_value = explainer.expected_value.item() if isinstance(model, XGBClassifier) else explainer.expected_value[1].item()

    shap_values_dict = {
        'shap_values': shap_values,
        'feature_names': feature_names,
        'expected_value': expected_value
    }

    return shap_values_dict

# 4. Run the API with uvicorn (uvicorn app:app --reload)
# first app stands for the pyhton file, second app for the API instance, --reload for automatic refresh
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)