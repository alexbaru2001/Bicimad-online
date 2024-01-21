# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 23:24:15 2024

@author: alexb
"""

import requests
import json
import pandas as pd

def request_api():
    url_acceso = "https://openapi.emtmadrid.es/v1/mobilitylabs/user/login/"
    headers = {
        "email": "barraganruizalejandro@gmail.com",
        "password": "Com24Ba25Ru"
    }
    
    response_acceso = requests.get(url_acceso, headers=headers)
    json_response_acceso = response_acceso.json()
    
    accessToken={'accessToken': json_response_acceso['data'][0]['accessToken']}
    
    url='https://openapi.emtmadrid.es/v1/transport/bicimad/stations/'
    
    response = requests.get(url, headers=accessToken)
    json_response = response.json()
    
    data_list = json_response.get('data', [])
    
    # Crear un DataFrame de Pandas con la lista 'data'
    df = pd.DataFrame(data_list)
    return df
