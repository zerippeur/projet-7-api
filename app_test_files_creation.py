# Standard library imports
import os

# Third-party imports
import pandas as pd
import pickle
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from mlflow.sklearn import mlflow
from typing import Tuple, Literal, Union
import shap
from sqlalchemy import create_engine, text
import shutil

def get_client_infos(client_id: int, output: Literal['dict', 'df'] = 'df', db_uri: str="postgresql+psycopg2://uercsb3qh549o8:pad42cb3e3e5308e4767ba6c19131d39dfb324c0f5d6fa0a217c92d69ba56d4b3@cfua00420e2gff.cluster-czz5s0kz4scl.eu-west-1.rds.amazonaws.com:5432/d7jon5tp0qcemc") -> Union[dict, pd.DataFrame]:
    """
    Get client infos from database.

    Parameters:
    -----------
    client_id: int
        The client id to get infos from
    output: Literal['dict', 'df']
        The output format of the function. Can be either 'dict' or 'df'
    db_uri: str
        The database URI

    Returns:
    --------
    Union[dict, pd.DataFrame]
        The client infos in the requested format.
    """
    engine = create_engine(db_uri)

    table_names = ['train_df', 'test_df']

    # SQL query to select infos from both tables where the client id matches
    query = text(f'SELECT * FROM {table_names[0]} WHERE "SK_ID_CURR" = :client_id UNION ALL SELECT * FROM {table_names[1]} WHERE "SK_ID_CURR" = :client_id ORDER BY "SK_ID_CURR"')
    with engine.connect() as conn:
        result = pd.read_sql_query(query, conn, params={'client_id': client_id}, index_col='SK_ID_CURR')

    if output == 'dict':
        # Convert the dataframe to a dictionary indexed by the client id
        model_dict = result.drop(columns=['level_0', 'index', 'TARGET']).to_dict(orient='index')
        # Extract the client infos from the dictionary
        client_infos = model_dict[client_id]
        return client_infos
    elif output == 'df':
        # Drop the unnecessary columns and return the resulting dataframe
        client_infos = result.drop(columns=['level_0', 'index', 'TARGET'])
        return client_infos

def get_data_for_shap_initiation(db_uri: str="postgresql+psycopg2://uercsb3qh549o8:pad42cb3e3e5308e4767ba6c19131d39dfb324c0f5d6fa0a217c92d69ba56d4b3@cfua00420e2gff.cluster-czz5s0kz4scl.eu-west-1.rds.amazonaws.com:5432/d7jon5tp0qcemc", limit=3000) -> dict:
    """
    Get data for SHAP initiation.

    This function gets data from the database, merges the train and test sets, and
    then resamples the data using RandomUnderSampler. The resulting dataset is
    returned.

    Parameters:
    -----------
    db_uri: str
        The database URI to connect to.
    limit: int
        The maximum number of samples to include in the resampled dataset.
        Defaults to 3000.

    Returns:
    --------
    data_for_shap_initiation: pandas.DataFrame
        The resampled dataset to use for SHAP initiation.
    """
    engine = create_engine(db_uri)

    query = text(f'SELECT * FROM train_df UNION ALL SELECT * FROM test_df LIMIT {limit*2}')
    result = pd.read_sql_query(query, engine, index_col='SK_ID_CURR')

    data_for_shap_initiation = result.drop(columns=['level_0', 'index', 'TARGET'])

    return data_for_shap_initiation.to_dict(orient="index")

def create_dict() -> None:
    """
    Function to create mocked client information dictionnaries for test files.
    """

    load_dotenv('tests.env')

    HEROKU_DATABASE_URI = os.getenv("DATABASE_URI")
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
    MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
    MLFLOW_RUN_ID = os.getenv("MLFLOW_RUN_ID")
    MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI")
    ARTIFACT_PATH = os.getenv("ARTIFACT_PATH")
    BEST_MODEL_NAME = os.getenv("BEST_MODEL_NAME")
    BEST_MODEL_VERSION = os.getenv("BEST_MODEL_VERSION")

    print(HEROKU_DATABASE_URI)

    client_dict_108567 = get_client_infos(client_id=108567, output='dict', db_uri=HEROKU_DATABASE_URI)
    client_dict_110084 = get_client_infos(client_id=110084, output='dict', db_uri=HEROKU_DATABASE_URI)

    with open('tests/client_dict_108567.pkl', 'wb') as f:
        pickle.dump(client_dict_108567, f)

    with open('tests/client_dict_110084.pkl', 'wb') as f:
        pickle.dump(client_dict_110084, f)

    shap_local_client_dict_108567 = {
        'client_infos': client_dict_108567,
        'feature_scale': 'Local',
    }

    shap_local_client_dict_110084 = {
        'client_infos': client_dict_110084,
        'feature_scale': 'Local',
    }

    with open('tests/shap_local_client_dict_108567.pkl', 'wb') as f:
        pickle.dump(shap_local_client_dict_108567, f)

    with open('tests/shap_local_client_dict_110084.pkl', 'wb') as f:
        pickle.dump(shap_local_client_dict_110084, f)

    data_for_shap_initiation = get_data_for_shap_initiation(db_uri=HEROKU_DATABASE_URI, limit=3000)

    with open('tests/data_for_shap_initiation.pkl', 'wb') as f:
        pickle.dump(data_for_shap_initiation, f)


    mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
    # Get the root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the 'mlflow' directory
    mlflow_dir = os.path.join(root_dir, 'tests/mlflow')
    os.makedirs(mlflow_dir, exist_ok=True)

    dst_path = "tests/mlflow"

    best_model = mlflow.sklearn.load_model(model_uri=MLFLOW_MODEL_URI, dst_path=dst_path)

    with open(f"{dst_path}/{ARTIFACT_PATH}/shap_explainer_{BEST_MODEL_NAME}_version_{BEST_MODEL_VERSION}.pkl", 'rb') as f:
        shap_explainer = pickle.load(f)

    threshold = float(mlflow.get_run(run_id=MLFLOW_RUN_ID).data.params['threshold'])

    if isinstance(best_model, LGBMClassifier):
        feature_names = best_model.feature_name_

    with open(f"tests/best_model.pkl", 'wb') as f:
        pickle.dump(best_model, f)
    
    with open(f"tests/shap_explainer.pkl", 'wb') as f:
        pickle.dump(shap_explainer, f)
    
    with open(f"tests/threshold.pkl", 'wb') as f:
        pickle.dump(threshold, f)
    
    with open(f"tests/feature_names.pkl", 'wb') as f:
        pickle.dump(feature_names, f)

    # Erases the mlflow folder
    shutil.rmtree(mlflow_dir)


def main():
    create_dict()

main()