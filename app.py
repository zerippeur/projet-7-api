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
# import pathlib
# from dagshub import get_repo_bucket_client

# 2. Create app and model objects
app = FastAPI()

# load_dotenv('api.env')

# DEPLOY = strtobool(os.getenv('DEPLOY'))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
MLFLOW_RUN_ID = os.getenv("MLFLOW_RUN_ID")
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI")
# MLFLOW_MODEL_URI_VERSION="runs:/f972b892ee1a4f2cac146a26052c1b40/LGBMClassifier_model__bayes_search__randomundersampler_balancing__tuning_run"
# EXPLAINER_PATH="mlflow-artifacts:/86617b78fa1941f9a1c21aa470029781/f972b892ee1a4f2cac146a26052c1b40/artifacts/LGBMClassifier_model__bayes_search__randomundersampler_balancing__tuning_run/shap_explainer_LGBMClassifier_version_18.pkl"
ARTIFACT_PATH = os.getenv("ARTIFACT_PATH")
# DAGSHUB_USER_TOKEN="c83285096695b6e8ec56ba2b07dfbeff75e9cadd"
# DAGSHUB_REPO="mlflow-tracking"
# DAGSHUB_BUCKET="mlflow_bucket"
BEST_MODEL_NAME = os.getenv("BEST_MODEL_NAME")
BEST_MODEL_VERSION = os.getenv("BEST_MODEL_VERSION")

mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
dst_path = "mlflow"
best_model = mlflow.sklearn.load_model(model_uri=MLFLOW_MODEL_URI, dst_path=dst_path)
with open(f"{dst_path}/{ARTIFACT_PATH}/shap_explainer_{BEST_MODEL_NAME}_version_{BEST_MODEL_VERSION}.pkl", 'rb') as f:
    shap_explainer = pickle.load(f)
threshold = float(mlflow.get_run(run_id=MLFLOW_RUN_ID).data.params['threshold'])

# client = mlflow.MlflowClient()

# boto_client = get_repo_bucket_client(f"{MLFLOW_TRACKING_USERNAME}/{DAGSHUB_REPO}")

# boto_client.download_file(
#     Bucket="mlflow-tracking",
#     Key="mlflow_bucket/LGBMClassifier/version-18/best_model.pkl",
#     Filename="best_model.pkl"
# )
# with open("best_model.pkl", 'rb') as f:
#     best_model = pickle.load(f)

# boto_client.download_file(
#     Bucket="mlflow-tracking",
#     Key="mlflow_bucket/LGBMClassifier/version-18/shap_explainer.pkl",
#     Filename="shap_explainer.pkl"
# )
# with open("shap_explainer.pkl", 'rb') as f:
#     shap_explainer = pickle.load(f)

# boto_client.download_file(
#     Bucket="mlflow-tracking",
#     Key="mlflow_bucket/LGBMClassifier/version-18/threshold.pkl",
#     Filename="threshold.pkl"
# )
# with open("threshold.pkl", 'rb') as f:
#     threshold = pickle.load(f)

# for var, object in zip(["best_model", "shap_explainer", "threshold"], ["best_model.pkl", "shap_explainer.pkl", "threshold.pkl"]):
#     boto_client.download_file(
#         Bucket=f"{DAGSHUB_REPO}",
#         Key=f"{DAGSHUB_BUCKET}/{BEST_MODEL_NAME}/version-{BEST_MODEL_VERSION}/{object}",
#         Filename=object
#     )
#     with open(object, 'rb') as f:
#         if var == "best_model":
#             best_model = pickle.load(f)
#         elif var == "shap_explainer":
#             shap_explainer = pickle.load(f)
#         elif var == "threshold":
#             threshold = pickle.load(f)

if isinstance(best_model, LGBMClassifier):
    feature_names = best_model.feature_name_

# List the contents of the root directory
# contents = os.listdir(root_dir)
# print("Contents of root directory:", contents)

# mlflow_dir = pathlib.Path.cwd() / "mlflow"
# mlflow_dir.mkdir(exist_ok=True)
# dst_path=str(mlflow_dir)

# List the contents of the root directory
# contents = os.listdir(root_dir)
# print("Contents of root directory:", contents)

# model = mlflow.sklearn.load_model(model_uri=MLFLOW_MODEL_URI, dst_path=dst_path)

@app.get('/model_threshold')
async def get_model_threshold()-> dict:
    """
    Get the model threshold value.

    Returns:
        dict: A dictionary containing the model threshold value.
    """

    # threshold = float(mlflow.get_run(run_id=MLFLOW_RUN_ID).data.params['threshold'])
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