import requests
import pandas as pd

with open('C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model//train_df_debug.csv', 'rb') as file:
    data = pd.read_csv(file)

client_id = 108567

response = requests.post(f'http://127.0.0.1:8000/predict?client_id={client_id}')
print(response)