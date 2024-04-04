import pandas as pd
import numpy as np
import pickle
from lightgbm import LGBMClassifier
from api.app import predict_credit_risk, initiate_shap_explainer
import random
import pytest
from hypothesis import given
from hypothesis.strategies import floats, composite

class MockLGBMClassifier(LGBMClassifier):
    # def predict(self):
    #     return random.sample(range(2), X.shape[0])
    def predict_proba(self, X):
        p = random.random()
        return np.array([[p, 1-p]])
    
class MockModelElements:
    @property
    def best_model(self):
        return MockLGBMClassifier()

@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    monkeypatch.setattr('api.app.model_elements', MockModelElements())

@composite
def mock_prediction_dicts(draw):
    threshold = draw(floats(min_value=0, max_value=1))
    return {'client_infos': 'value', 'threshold': threshold}

@given(mock_prediction_dicts())
def test_predict_keys(mock_prediction_dict):
    prediction_dict = predict_credit_risk(prediction_dict=mock_prediction_dict)
    key_list = ['prediction', 'confidence', 'risk_category']

    assert all(key in prediction_dict.keys() for key in key_list)

@given(mock_prediction_dicts())
def test_predict_types(mock_prediction_dict):
    for _ in range(3):
        prediction_dict = predict_credit_risk(prediction_dict=mock_prediction_dict)
        for key, value_type in zip(['prediction', 'confidence', 'risk_category'], [int, float, str]):
            assert value_type == type(prediction_dict[key])
            if key == 'risk_category':
                assert prediction_dict[key] in ['SAFE', 'RISKY', 'NOPE']

@given(mock_prediction_dicts())
def test_predict_thresholding(mock_prediction_dict):
    threshold = mock_prediction_dict['threshold']
    prediction_dict = predict_credit_risk(prediction_dict=mock_prediction_dict)
    if prediction_dict['confidence'] > 1 - threshold:
        assert (prediction_dict['prediction'] == 0 and prediction_dict['risk_category'] == 'SAFE')
    elif prediction_dict['confidence'] > 1 - threshold - 0.05:
        assert (prediction_dict['prediction'] == 1 and prediction_dict['risk_category'] == 'RISKY')
    else:
        assert (prediction_dict['prediction'] == 1 and prediction_dict['risk_category'] == 'NOPE')

# # run test_predict_thresholding 100 times
# for _ in range(100):
#     test_predict_thresholding(mock_prediction_dict)



# # Loading the mocked data for shap initiation as the dashboard would send them to the app
# with open('data_for_shap_initiation.pkl', 'rb') as f:
#     data_for_shap_initiation = pickle.load(f)

# # Loading the LGBMClassifier model
# with open('best_model.pkl', 'rb') as f:
#     best_model = pickle.load(f)

# # Loading the shap explainer
# with open('shap_explainer.pkl', 'rb') as f:
#     shap_explainer = pickle.load(f)

# # Loading the feature names
# with open('feature_names.pkl', 'rb') as f:
#     feature_names = pickle.load(f)

# # Loading the model's threshold
# with open('threshold.pkl', 'rb') as f:
#     threshold = pickle.load(f)

# def predict_credit_risk(prediction_dict: dict)-> dict:
#     """
#     Function to predict credit risk based on a dictionary input.

#     Parameters:
#         prediction_dict: dictionary containing client information

#     Returns:
#         prediction_dict: A dictionary with prediction, confidence, and risk category
#     """

#     client_infos = pd.DataFrame.from_dict([prediction_dict['client_infos']])
#     threshold = prediction_dict['threshold']
#     proba_repay_wo_risk = float(best_model.predict_proba(client_infos)[:,0][0])

#     if proba_repay_wo_risk > 1 - threshold:
#         risk_category = "SAFE"
#         risk_prediction = 0

#     elif proba_repay_wo_risk > 1 - threshold - 0.05:
#         risk_category = "RISKY"
#         risk_prediction = 1

#     else:
#         risk_category = "NOPE"
#         risk_prediction = 1

#     prediction_dict = {
#         'prediction': risk_prediction,
#         'confidence': proba_repay_wo_risk,
#         'risk_category': risk_category
#     }

#     return prediction_dict

# def initiate_shap_explainer(data_for_shap_initiation: dict)-> dict:
#     """
#     Initiates the SHAP explainer with the given data and returns the SHAP values, feature names, and expected value.
    
#     Parameters:
#         data_for_shap_initiation (dict): The data used to initiate the SHAP explainer.
        
#     Returns:
#         shap_values_dict: A dictionary containing the SHAP values, feature names, and expected value.
#     """

#     data = pd.DataFrame.from_dict(data_for_shap_initiation, orient='index')
#     shap_values_global = shap_explainer.shap_values(data, check_additivity=False)


#     if isinstance(best_model, LGBMClassifier):
#         shap_values_global = shap_values_global.tolist()
    
#     shap_values_dict = {
#         'shap_values': shap_values_global,
#         'feature_names': feature_names,
#         'expected_value': None
#     }
#     return shap_values_dict

# def test_predict_credit_risk():
#     """
#     Test the predict_credit_risk function.

#     Loads mocked client data as the dashboard would send them to the app endpoint and the
#     LGBMClassifier model.

#     Parameters:
#         None

#     Returns:
#         None
#     """
#     # Loading the mocked client infos as the dashboard would send them to the app
#     with open('client_dict_108567.pkl', 'rb') as f:
#         client_dict_108567 = pickle.load(f)

#     with open('client_dict_110084.pkl', 'rb') as f:
#         client_dict_110084 = pickle.load(f)

#     # Testing the predict_credit_risk function
#     prediction_dict_108567 = predict_credit_risk(
#         prediction_dict={
#             'client_infos': client_dict_108567,
#             'threshold': threshold
#         }
#     )
#     prediction_dict_110084 = predict_credit_risk(
#         prediction_dict={
#             'client_infos': client_dict_110084,
#             'threshold': threshold
#         }
#     )

#     keys_list = ['prediction', 'confidence', 'risk_category']
#     assert all(key in prediction_dict_108567.keys() for key in keys_list)
#     assert all(key in prediction_dict_110084.keys() for key in keys_list)

#     for key, value_type in zip(['prediction', 'confidence', 'risk_category'], [int, float, str]):
#         assert value_type == type(prediction_dict_108567[key])
#         assert value_type == type(prediction_dict_110084[key])

#     if prediction_dict_108567['confidence'] > 1 - threshold:
#         assert prediction_dict_108567['prediction'] == 0
#         assert prediction_dict_108567['risk_category'] == 'SAFE'
#     elif prediction_dict_108567['confidence'] > 1 - threshold - 0.05:
#         assert prediction_dict_108567['prediction'] == 1
#         assert prediction_dict_108567['risk_category'] == 'RISKY'
#     else:
#         assert prediction_dict_108567['prediction'] == 1
#         assert prediction_dict_108567['risk_category'] == 'NOPE'

#     if prediction_dict_110084['confidence'] > 1 - threshold:
#         assert prediction_dict_110084['prediction'] == 0
#         assert prediction_dict_110084['risk_category'] == 'SAFE'
#     elif prediction_dict_110084['confidence'] > 1 - threshold - 0.05:
#         assert prediction_dict_110084['prediction'] == 1
#         assert prediction_dict_110084['risk_category'] == 'RISKY'
#     else:
#         assert prediction_dict_110084['prediction'] == 1
#         assert prediction_dict_110084['risk_category'] == 'NOPE'

# def test_initiate_shap_explainer():
#     """
#     Test the initiate_shap_explainer function.

#     Loads mocked client data as the dashboard would send them to the app endpoint and the
#     LGBMClassifier model.

#     Parameters:
#         None

#     Returns:
#         None
#     """

#     # Testing the initiate_shap_explainer function
#     shap_values_dict = initiate_shap_explainer(data_for_shap_initiation=data_for_shap_initiation)

#     assert all(key in shap_values_dict.keys() for key in ['shap_values', 'feature_names', 'expected_value'])
#     if isinstance(best_model, LGBMClassifier):
#         assert shap_values_dict['feature_names'] == best_model.feature_name_