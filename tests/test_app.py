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