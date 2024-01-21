# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 12:25:18 2024

@author: alexb
"""

import re
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


#verbose nos da unos print por pantalla
def scrape_distritos_data(verbose):
    url_expansion = 'https://es.wikipedia.org/wiki/Anexo:Distritos_de_Madrid'
    #Cambiamos el nombre de las columnas
    distritos_table_columns = [
        'numero',
        'nombre',
        'area',
        'poblacion',
        'densidad']
    #peticion http
    expansion_request = requests.get(url_expansion)
    #status_code es okey si sale 200, si no es asi es que hay algo mal, si da mal que devuelva un dataframe vacio
    if expansion_request.status_code != 200:
        return pd.DataFrame([], columns=distritos_table_columns)
    #Se trata de manejar lo que se ha sacado de la web
    expansion_web = BeautifulSoup(expansion_request.text, 'html.parser')
    #print(expansion_web)
    if (verbose):
        print(expansion_web.prettify())
    #nos quedamos con la tabla
    table = expansion_web.find('table', {'class': 'wikitable sortable'})
    
    for cell in table.find_all('td'):
        span=cell.find('span')
        if span:
            span.decompose()
            
    for cell in table.find_all('th'):
        span=cell.find('span')
        if span:
            span.decompose()
    if (verbose):
        print(table.prettify()) 
        
    distritos_table = pd.read_html(
        str(table), header=0, encoding='utf-8', decimal=',', thousands='.')[0]


    if (verbose):
        print(distritos_table.head())
    distritos_table = distritos_table.iloc[:, :-2]

    distritos_table.columns = distritos_table_columns
    if (verbose):
        print(distritos_table.dtypes)


    if (verbose):
        print(distritos_table.dtypes)
        
    def cambiar_formato(numero_str):
        # Elimino espacios y cambio la coma por un punto
        numero_str_sin_espacios = re.sub(r'\s', '', numero_str)
        numero_str_con_punto = re.sub(r',', '.', numero_str_sin_espacios)
        # Convierto la cadena resultante a un float
        return float(numero_str_con_punto)
    
    for column in distritos_table.columns[2:-1]:
        distritos_table[column]=distritos_table[column].apply(cambiar_formato)
    #Se introduce 32 para poder
    distritos_table['numero']=distritos_table['numero'].fillna(32.0)
    distritos_table['numero']=distritos_table['numero'].astype(int)
    distritos_table['numero']=distritos_table['numero'].astype(str)
    return (distritos_table)


def obtener_info_distritos():
    distritos_data = scrape_distritos_data(False)
    return distritos_data


