# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:49:18 2024

@author: alexb
"""
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import streamlit as st  
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, shape
from shapely import ops
import pyproj
import plotly.express as px
from funciones_aux import *
from streamlit_folium import st_folium
import plotly.express as px

App_title='Estaciones de Bicimad'
App_subtitle='Source: Bicimad'

def main():
    st.set_page_config(page_title=App_title,
                       page_icon="\U0001F6B2",
                       layout="wide")
    st.title(App_title)
    menu()
    
if __name__ == "__main__":
    main()    
