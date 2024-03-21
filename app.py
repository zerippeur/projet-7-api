# Standard library imports
import os
from distutils.util import strtobool

# Third-party imports
import pandas as pd
import pickle
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from lightgbm import LGBMClassifier
from mlflow.sklearn import mlflow

# 2. Create app and model objects
app = FastAPI()

load_dotenv('api.env')

# DEPLOY = strtobool(os.getenv('DEPLOY'))

MLFLOW_TRACKING_USERNAME = os.getenv('MLFLOW_TRACKING_USERNAME')
MLFLOW_TRACKING_PASSWORD = os.getenv('MLFLOW_TRACKING_PASSWORD')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_MODEL_URI = os.getenv('MLFLOW_MODEL_URI')
MLFLOW_RUN_ID = os.getenv('MLFLOW_RUN_ID')
EXPLAINER_PATH = os.getenv('EXPLAINER_PATH')
# PORT = os.environ.get('PORT', 8000)

# Create an instance of the MlflowClient with the artifact_ignore_patterns parameter set
client = mlflow.tracking.MlflowClient(artifact_ignore_patterns=["^input_example"])
mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
model = mlflow.sklearn.load_model(model_uri=MLFLOW_MODEL_URI)

@app.get('/model_threshold')
async def get_model_threshold()-> dict:
    """
    Get the model threshold value.

    Returns:
        dict: A dictionary containing the model threshold value.
    """

    threshold = float(mlflow.get_run(run_id=MLFLOW_RUN_ID).data.params['threshold'])
    threshold_dict = {
        'threshold': threshold
    }
    
    return threshold_dict

@app.get('/global_feature_importance')
async def get_global_feature_importance()-> dict:
    """
    Function to retrieve global feature importance information from different classifier models.

    Returns:
        dict: A dictionary containing the global feature importance information.
    """

    if isinstance(model, LGBMClassifier):
        feature_importance_dict = {
            'model_type': 'LGBMClassifier',
            'feature_importance': {
                'gain': dict(zip(model.feature_name_, model.feature_importances_.tolist())),
            }
        }

    return feature_importance_dict

@app.post('/predict_from_dict')
def predict_credit_risk(prediction_dict: dict)-> dict:
    """
    Function to predict credit risk based on a dictionary input.

    Parameters:
        prediction_dict: dictionary containing client information

    Returns:
        prediction_dict: A dictionary with prediction, confidence, and risk category
    """

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

    prediction_dict = {
        'prediction': risk_prediction,
        'confidence': proba_repay_wo_risk,
        'risk_category': risk_category
    }

    return prediction_dict

@app.post('/initiate_shap_explainer')
async def initiate_shap_explainer(data_for_shap_initiation: dict)-> dict:
    """
    Initiates the SHAP explainer with the given data and returns the SHAP values, feature names, and expected value.
    
    Parameters:
        data_for_shap_initiation (dict): The data used to initiate the SHAP explainer.
        
    Returns:
        shap_values_dict: A dictionary containing the SHAP values, feature names, and expected value.
    """

    global explainer
    global feature_names
    global shap_values_global

    with open(mlflow.artifacts.download_artifacts(EXPLAINER_PATH), 'rb') as f:
        explainer = pickle.load(f)

    data = pd.DataFrame.from_dict(data_for_shap_initiation, orient='index')
    shap_values_global = explainer.shap_values(data)

    if isinstance(model, LGBMClassifier):
        feature_names = model.feature_name_
        shap_values_global = shap_values_global[1].tolist()
    
    shap_values_dict = {
        'shap_values': shap_values_global,
        'feature_names': feature_names,
        'expected_value': None
    }
    return shap_values_dict

@app.post('/shap_feature_importance')
def get_shap_feature_importance(shap_feature_importance_dict: dict)-> dict:
    """
    Generate the SHAP feature importance values based on the input dictionary.

    Parameters:
        shap_feature_importance_dict (dict): A dictionary containing information needed to calculate SHAP feature importance.

    Returns:
        shap_values_dict: A dictionary containing the SHAP values, feature names, and expected value.
    """


    feature_scale = shap_feature_importance_dict['feature_scale']

    if feature_scale == 'Global':
        shap_values = shap_values_global#.tolist() if isinstance(model, XGBClassifier) else shap_values_global[1].tolist()
        expected_value = None

    elif feature_scale == 'Local':
        client_infos = pd.DataFrame.from_dict(shap_feature_importance_dict['client_infos'], orient='index').T
        shap_values = explainer.shap_values(client_infos)[1].tolist()
        expected_value = explainer.expected_value[1].item()

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
    # port=int(PORT)
    # if DEPLOY:
    #     uvicorn.run(app, host='0.0.0.0', port=port)
    # else:
    uvicorn.run(app, host='127.0.0.1', port=8000)