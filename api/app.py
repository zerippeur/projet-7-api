# Standard library imports
import os
# from distutils.util import strtobool

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

# load_dotenv('api.env')

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
MLFLOW_RUN_ID = os.getenv("MLFLOW_RUN_ID")
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI")
ARTIFACT_PATH = os.getenv("ARTIFACT_PATH")
BEST_MODEL_NAME = os.getenv("BEST_MODEL_NAME")
BEST_MODEL_VERSION = os.getenv("BEST_MODEL_VERSION")

mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
# Get the root directory of your FastAPI application
root_dir = os.path.dirname(os.path.abspath(__file__))
print(root_dir)

# Create the 'mlflow' directory
mlflow_dir = os.path.join(root_dir, 'mlflow')
os.makedirs(mlflow_dir, exist_ok=True)

dst_path = "mlflow"
best_model = mlflow.sklearn.load_model(model_uri=MLFLOW_MODEL_URI, dst_path=dst_path)
with open(f"{dst_path}/{ARTIFACT_PATH}/shap_explainer_{BEST_MODEL_NAME}_version_{BEST_MODEL_VERSION}.pkl", 'rb') as f:
    shap_explainer = pickle.load(f)
threshold = float(mlflow.get_run(run_id=MLFLOW_RUN_ID).data.params['threshold'])

if isinstance(best_model, LGBMClassifier):
    feature_names = best_model.feature_name_

os.removedirs(mlflow_dir)

@app.get('/model_threshold')
async def get_model_threshold()-> dict:
    """
    Get the model threshold value.

    Returns:
        dict: A dictionary containing the model threshold value.
    """
    
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

    if isinstance(best_model, LGBMClassifier):
        feature_importance_dict = {
            'model_type': 'LGBMClassifier',
            'feature_importance': {
                'gain': dict(zip(best_model.feature_name_, best_model.feature_importances_.tolist())),
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
    proba_repay_wo_risk = float(best_model.predict_proba(client_infos)[:,0][0])

    if proba_repay_wo_risk > 1 - threshold:
        risk_category = "SAFE"
        risk_prediction = 0

    elif proba_repay_wo_risk > 1 - threshold - 0.05:
        risk_category = "RISKY"
        risk_prediction = 1

    else:
        risk_category = "NOPE"
        risk_prediction = 1

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

    # global explainer
    global shap_values_global

    data = pd.DataFrame.from_dict(data_for_shap_initiation, orient='index')
    shap_values_global = shap_explainer.shap_values(data)


    if isinstance(best_model, LGBMClassifier):
        shap_values_global = shap_values_global.tolist()
    
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
        shap_values = shap_values_global
        expected_value = None

    elif feature_scale == 'Local':
        client_infos = pd.DataFrame.from_dict(shap_feature_importance_dict['client_infos'], orient='index').T
        shap_values = shap_explainer.shap_values(client_infos).tolist()
        expected_value = shap_explainer.expected_value.item()

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