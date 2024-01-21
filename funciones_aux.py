# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:49:50 2024

@author: alexb
"""
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, shape
from shapely import ops
import pyproj
import plotly.express as px
from streamlit_folium import st_folium
import plotly.express as px
import streamlit as st 
from api_bicimad import *
from scraper_distritos_beautiful_soup import *
from plotnine import *
###Cargado de datos########################################################################################################################################################
#Cargamos los dos tipos de fichero que necesitamos para la practica
def cargar_ficheros(fichero):
    if 'zip' in fichero:
        shp_path = fichero

        # Carga el archivo SHP en un GeoDataFrame
        gdf = gpd.read_file(shp_path)
        return gdf
    else:
        return request_api()

###Preparacion de los datos###################################################################################################################################################

#Modificamos la estructura del df para poder representar los datos en un mapa, 
#la opcion añadir_geometry_inverted se incluye para el caso en el que queremos relacionar el df con gdf
def modificar_col_geometry(df,añadir_geometry_inverted=False):
    if añadir_geometry_inverted:
        df['geometry_inverted'] = df['geometry'].apply(lambda x: Point(x['coordinates'][0],x['coordinates'][1]))
        df['geometry'] = df['geometry'].apply(lambda x: Point(x['coordinates'][1],x['coordinates'][0]))
        df['distrito']=None
    else:
        df['latitud'] = df['geometry'].apply(lambda x: x['coordinates'][1])
        df['longitud'] = df['geometry'].apply(lambda x: x['coordinates'][0])
        df['geometry'] = df['geometry'].apply(lambda x: x['type'])
    return df

#Esta funcion se implementa para cambiar la proyeccion del poligono de forma que se pueda relacionar con puntos del mapa 
def cambio_proyeccion(poly):
    poly_crs = pyproj.CRS("EPSG:25830")
    poly = ops.transform(lambda x, y: pyproj.Transformer.from_crs(poly_crs, pyproj.CRS("EPSG:4326"), always_xy=True).transform(x,y), poly)
    return poly


#Esta funcion sirve para crear un df que contenga informacion sobre que distrito corresponde a cada estacion
def estaciones_con_distrito(df,gdf):
    df=modificar_col_geometry(df,True)   
    gdf=gdf.assign(changes=gdf['geometry'])
    gdf['changes']=gdf['changes'].apply(cambio_proyeccion)    
    
    def identificar_distrito(estacion):
        al_reves = gdf['changes'].contains(estacion)
        esta = gdf['NOMBRE'].loc[al_reves]
        return esta.iloc[0] 

    df['distrito']=df['geometry_inverted'].apply(identificar_distrito)
    gdf=gdf.iloc[:, :-1]
    df_con_distrito=df
    return df_con_distrito  

#Se agrupa por distrito el numero de estaciones
def num_estaciones_por_distrito(df_con_distrito):
    df_estaciones_por_distrito=df_con_distrito['distrito'].value_counts().reset_index()
    return df_estaciones_por_distrito

#Se añade informacion en gdf sobre la cantidad de estaciones que hay en cada distrito
def modificar_gdf(gdf,df_estaciones_por_distrito):
    gdf_con_estaciones=gdf.merge(df_estaciones_por_distrito,left_on='NOMBRE', right_on='distrito',how='left')
    gdf_con_estaciones['COD_DIS_TX']=gdf_con_estaciones['count']
    gdf_con_estaciones=gdf_con_estaciones.iloc[:, :-2]
    gdf_con_estaciones = gdf_con_estaciones.rename(columns={'COD_DIS_TX': 'Estaciones'})
    return gdf_con_estaciones

#Se realiza un filtrado de las bases inferiores a un numero dado
def cantidad_estaciones(df,num):
    df_cantidad=df.loc[df['total_bases']>=num]
    return df_cantidad

def estaciones_con_info_extra(df,gdf,df_scrapeado):
    df_distrito=estaciones_con_distrito(df,gdf)
    bicis_distrito=df_distrito.groupby('distrito')[['distrito','total_bases']].sum('total_bases').reset_index().rename(columns={'total_bases': 'count'})
    gdf_con_estaciones=modificar_gdf(gdf,bicis_distrito)
    gdf_distritos=gdf_con_estaciones.rename(columns={'Estaciones': 'bicis'})[['COD_DIS','bicis','NOMBRE']]
    df_con_scrapper=df_scrapeado.merge(gdf_distritos, left_on='numero', right_on='COD_DIS')[['nombre','area','poblacion','densidad','bicis']]
    return df_con_scrapper



def distrito_nivel_ocupacion(df_distrito,opcion,filtro=0):    
    low=df_distrito.loc[df_distrito['light']==filtro]['distrito'].value_counts().reset_index().rename(columns={'count': 'light'})
    result=(df_distrito['distrito'].value_counts().reset_index()).merge(low,on='distrito')
    result['porcentaje']=result['light']/result['count']
    if opcion=='max':
        return result.loc[result['porcentaje']==result['porcentaje'].max()][['distrito','porcentaje']]
    else:
        return result.loc[result['porcentaje']==result['porcentaje'].min()][['distrito','porcentaje']]



###Display################################################################################################################################################################

#Se representan de manera simple todas las estaciones en un mapa
def mostrar_estaciones_bicimad_de_golpe(df):
    def agregar_marcador(fila):
        folium.Marker(location=[fila['latitud'], fila['longitud']], popup=fila['name']).add_to(map)
    map=folium.Map(location=[df['latitud'].mean(),df['longitud'].mean()], zoom_start=11, scrollWheelZoom=False, tiles='CartoDB positron')
    # Aplicar la función a cada fila del DataFrame
    df.apply(agregar_marcador, axis=1)
    st_map=st_folium(map, width=700, height=450)    

#Se emplea la funcion cluster para representar de manera más cómoda todas las estaciones en el mapa
def mostrar_estaciones_bicimad_cluster(df):
    def agregar_marcador1(fila):
        folium.Marker(location=[fila['latitud'], fila['longitud']], popup=fila['name']).add_to(marker_cluster)

    map=folium.Map(location=[df['latitud'].mean(),df['longitud'].mean()], zoom_start=11, scrollWheelZoom=False, tiles='CartoDB positron')
    marker_cluster = MarkerCluster().add_to(map)
    # Aplicar la función a cada fila del DataFrame
    df.apply(agregar_marcador1, axis=1)
    st_map=st_folium(map, width=700, height=450)    
    

#Se representa en el mapa los distritos de madrid
def mostrar_distritros(gdf):
    def style_function(feature):
        return {
            'fillColor': 'blue',  # Color de relleno
            'color': 'black',      # Color de borde
            'weight': 1,           # Grosor de la línea
            'opacity': 0.5,         # Opacidad de la línea
            'fillOpacity': 0.3
        }
    map=folium.Map(location=[40.416767, -3.681854], zoom_start=10, scrollWheelZoom=False, tiles='CartoDB positron')
    folium.GeoJson(gdf,style_function=style_function,
                       popup = folium.GeoJsonPopup(fields = ['NOMBRE'],
                                aliases=['Distrito: '],
                                localize=True,
                                labels=True,
                                parse_html=False)).add_to(map)
    st_map=st_folium(map, width=700, height=450)

#Funcion similar a mostrar_distritros que añade una capa en la que se representan en cluster las estaciones de bici
def mostrar_distritros_con_estaciones(gdf,df):
    def agregar_marcador1(fila):
        folium.Marker(location=[fila['latitud'], fila['longitud']], popup=fila['name']).add_to(marker_cluster)

    map=folium.Map(location=[df['latitud'].mean(),df['longitud'].mean()], zoom_start=11, scrollWheelZoom=False, tiles='CartoDB positron')
    marker_cluster = MarkerCluster().add_to(map)
    # Aplicar la función a cada fila del DataFrame
    df.apply(agregar_marcador1, axis=1)
    folium.GeoJson(gdf,popup = folium.GeoJsonPopup(fields = ['NOMBRE'],
                                aliases=['Distrito: '],
                                localize=True,
                                labels=True,
                                parse_html=False)).add_to(map)
    st_map=st_folium(map, width=700, height=450)    
    

#Representa por escala de colores en el mapa la cantidad de estaciones por distrito
def mostrar_mapa_cloropleth(df_estaciones_por_distrito,gdf_con_estaciones):
    map=folium.Map(location=[40.42514422131318, -3.6833399438991723], zoom_start=10, scrollWheelZoom=False, tiles='CartoDB positron')
    folium.Choropleth(
        geo_data=gdf_con_estaciones,
        name='choropleth',
        data = df_estaciones_por_distrito,
        columns=['distrito', 'count'],
        key_on='feature.properties.NOMBRE',
        fill_color='YlGn',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Numero de estaciones de bici'
    ).add_to(map)
    
    def style_function(feature):
        return {
            'fillColor': 'green',  # Color de relleno
            'color': 'black',      # Color de borde
            'weight': 1,           # Grosor de la línea
            'opacity': 0.5,         # Opacidad de la línea
            'fillOpacity': 0.0
        }    
    folium.GeoJson(gdf_con_estaciones,
                   style_function=style_function,
                   popup = folium.GeoJsonPopup(fields = ['NOMBRE','Estaciones'],
                                               aliases=['Distrito:','Estaciones: '],
                                               localize=True,
                                               labels=True,
                                               parse_html=False
                                                        )).add_to(map)
    st_map=st_folium(map, width=700, height=450)
    
    
    
def mostrar_mapa_densidad(df):
    fig = px.density_mapbox(df, lat = 'latitud', lon = 'longitud', z = 'total_bases',
                            radius = 10,
                            center = dict(lat = df['latitud'].mean(), lon = df['longitud'].mean()),
                            zoom = 10,
                            mapbox_style = 'open-street-map')    
    return fig


def mostrar_grafico_cant_bicis(df,gdf,percent):
    df_copia_grafico = df.copy(deep=True)
    gdf_copia_grafico = gdf.copy(deep=True)
    df_con_distrito,df_estaciones_por_distrito,_=df_con_distrito_y_gdf_con_estaciones(df_copia_grafico,gdf_copia_grafico)
    total_bicis=df_con_distrito.groupby('distrito')['total_bases'].sum().reset_index()
    total_bicis_estacion=total_bicis.merge(df_estaciones_por_distrito,on='distrito',how='left')
    columna=None
    if percent=='ratio':
        total_bicis_estacion=total_bicis_estacion.assign(changes=total_bicis_estacion['total_bases']/total_bicis_estacion['count'])
        columna='changes'
    else:
        columna='total_bases'
    total_bicis_estacion=total_bicis_estacion.sort_values(by=columna)
    fig = px.bar(total_bicis_estacion, x='distrito', y=columna, 
             labels={'distrito': 'Distrito', columna: 'Bicis'})

    # Puedes ajustar el ángulo del texto del eje x
    fig.update_xaxes(tickangle=45, tickmode='array')
    
    return fig

def mostrar_densidad_tamaño_estación(df_distrito,distrito):
    df_distrito=df_distrito.rename(columns={'distrito': 'Distrito'})
    return (ggplot(df_distrito.loc[df_distrito['Distrito'].isin(distrito)], aes(x='total_bases', fill='Distrito')) + 
      geom_density(alpha=0.5)+
      scale_x_continuous(breaks=range(0, 41, 5), minor_breaks=[])+
      scale_y_continuous(minor_breaks=[])+
      theme_bw()+
      theme(legend_title=element_text(weight='bold'),
            axis_text=element_text(weight='bold'),
           plot_title = element_text(color = "black", face="bold"),
           axis_title = element_text(size = 14, color = "black", face="bold"),
           panel_background = element_blank(),
           axis_line = element_line(color = "black", size=1),
           axis_ticks = element_line(color = "black", size=1))+
      labs(x='Numero de bicis por estación', y='Densidad', title='Densidad tipo de estaciones', color='Distritos')
    )


def nivel_ocupacion(df_distrito,distrito, formato='dodge'):
    if formato=='fill':
        return (ggplot(df_distrito.loc[df_distrito['distrito'].isin(distrito)], aes(x='distrito', fill='factor(light)')) + 
        geom_bar(position=formato) + 
        scale_fill_manual(values=["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00"],name = "Nivel de ocupación",labels = ['Bajo','Medio','Alto','No disponible'])+
        theme(axis_text_x = element_text(angle = 45, hjust = 1,color = "black"),
              panel_grid_major_x=element_blank(),
              panel_background=element_rect(fill='#d4d4d4'),
              panel_grid_major_y=element_line(size=0.5))+
        labs(x='Estaciones', y='Distritos')
        )
    else:
        return (ggplot(df_distrito.loc[df_distrito['distrito'].isin(distrito)], aes(x='distrito', fill='factor(light)')) + 
        geom_bar(position=formato) + 
        scale_fill_manual(values=["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00"],name = "Nivel de ocupación",labels = ['Bajo','Medio','Alto','No disponible'])+
        theme(axis_text_x = element_text(angle = 45, hjust = 1,color = "black"),
              panel_grid_major_x=element_blank(),
              panel_grid_minor_y=element_blank(),
              panel_background=element_rect(fill='#d4d4d4'),
              panel_grid_major_y=element_line(size=0.5))+
        scale_y_continuous(breaks=range(0, 41, 5),
                           minor_breaks=range(0, 41, 5))+
        labs(x='Estaciones', y='Distritos')+
        expand_limits(y=(0, 41))
        )

def mostrar_personas_bici(df_con_scrapper):
    df_con_scrapper['personas por bici']=df_con_scrapper['poblacion']/df_con_scrapper['bicis']
    fig = px.bar(df_con_scrapper, y='personas por bici', x='nombre', 
             labels={'nombre': 'Distrito', 'personas por bici': 'Personas por Bici'},
             color_discrete_sequence=['#4682B4'])

    fig.update_xaxes(tickangle=20, tickmode='array')
    fig.update_xaxes(title_standoff=20)
    # Poner el nombre del eje x en negrita
    fig.update_xaxes(title_text='Distritos', title_font=dict(size=20, family='Arial', color='black'))
    
    # Poner el nombre del eje y en negrita
    fig.update_yaxes(title_text='Personas por Bici', title_font=dict(size=20, family='Arial', color='black'))
    return fig


def mostrar_hectareas_bici(df_con_scrapper):
    df_con_scrapper['hectareas por bici']=(df_con_scrapper['area']*100)/df_con_scrapper['bicis']
    
    fig = px.bar(df_con_scrapper, y='hectareas por bici', x='nombre', 
                 labels={'nombre': 'Distrito', 'hectareas por bici': 'Hectareas por Bici'},
                 color_discrete_sequence=['#4682B4'])
    
    fig.update_xaxes(tickangle=20, tickmode='array')
    fig.update_xaxes(title_standoff=20)
    # Poner el nombre del eje x en negrita
    fig.update_xaxes(title_text='Distritos', title_font=dict(size=20, family='Arial', color='black'))
    
    # Poner el nombre del eje y en negrita
    fig.update_yaxes(title_text='Hectareas por Bici', title_font=dict(size=20, family='Arial', color='black'))  
    return fig

def mostrar_hectareas(df_con_scrapper):
    df_con_scrapper['hectareas']=df_con_scrapper['area']*100

    fig = px.bar(df_con_scrapper, y='hectareas', x='nombre', 
                 labels={'nombre': 'Distrito', 'hectareas': 'Hectareas'},
                 color_discrete_sequence=['#4682B4'])
    
    fig.update_xaxes(tickangle=20, tickmode='array')
    fig.update_xaxes(title_standoff=20)
    # Poner el nombre del eje x en negrita
    fig.update_xaxes(title_text='Distritos', title_font=dict(size=20, family='Arial', color='black'))
    
    # Poner el nombre del eje y en negrita
    fig.update_yaxes(title_text='Hectáreas', title_font=dict(size=20, family='Arial', color='black'))
    return fig



###Funciones principales########################################################################################################################################


def cargar_datos():
    df=cargar_ficheros('response.json')
    gdf=cargar_ficheros('Distritos.zip')
    df_scrapeado=scrape_distritos_data(False)
    return df,gdf,df_scrapeado

def botones_para_modificar_df(df, distritos=False):
    if distritos=='Solo distritos':
        return df
    else:
        opciones=['Activas','Inactivas']
        seleccion=st.multiselect('Estado de la estacion:', opciones)
        if len(seleccion)==1:
            if 'Activas' in seleccion:
                seleccion_1=0
            else:
                seleccion_1=1
                
            df=df.loc[df['no_available']==seleccion_1]
        min_df=df['total_bases'].loc[(df['total_bases']==min(df['total_bases']))].iloc[0]
        max_df=df['total_bases'].loc[(df['total_bases']==max(df['total_bases']))].iloc[0]
        num = st.slider('Número mínimo de bicis por estación',min_df,max_df)
        df=cantidad_estaciones(df,num)
        return df


def estaciones_option(df):
    col1, col2 = st.columns([1, 3])
    with col1:
      st.header("Opciones")
      sidebar_estaciones = st.selectbox("Tipo de mapa de las estaciones:",
           ("Simple", "Cluster")
       )
      df=botones_para_modificar_df(df)
      
    with col2:
      df=modificar_col_geometry(df)
      if sidebar_estaciones=='Simple':
          st.header("Mapa de las estaciones simple") 
          mostrar_estaciones_bicimad_de_golpe(df)
      if sidebar_estaciones=='Cluster':
          st.header("Mapa de estaciones cluster")
          mostrar_estaciones_bicimad_cluster(df)

def distrito_option(gdf,df):
    col1, col2 = st.columns([1, 3])
    with col1:
      st.header("Opciones")
      sidebar_distritos= st.selectbox('Tipo de mapa de las estaciones:',
           ("Solo distritos", "Distritos con estaciones"))
      df=botones_para_modificar_df(df,sidebar_distritos)
      df=modificar_col_geometry(df)
      
    with col2:
        if sidebar_distritos=='Solo distritos':
            st.header("Solo distritos")
            mostrar_distritros(gdf)
        if sidebar_distritos=='Distritos con estaciones':
            st.header("Distritos con estaciones")
            mostrar_distritros_con_estaciones(gdf,df)

def df_con_distrito_y_gdf_con_estaciones(df,gdf):
    df_con_distrito=estaciones_con_distrito(df,gdf)
    df_estaciones_por_distrito=num_estaciones_por_distrito(df_con_distrito)
    gdf_con_estaciones=modificar_gdf(gdf,df_estaciones_por_distrito)
    return df_con_distrito,df_estaciones_por_distrito,gdf_con_estaciones
        
        
def analisis_option(df,gdf):
    col1, col2 = st.columns([1, 3])
    with col1:
      st.header("Opciones")
      sidebar_distritos_estacion= st.selectbox('Tipo de mapa de las estaciones:',
           ("Mapa con escala de color", "Mapa de densidad"))
      df=botones_para_modificar_df(df)
      
    with col2:
        if sidebar_distritos_estacion=='Mapa de densidad':
            st.header("Mapa de densidad")
            df_copia_profunda = df.copy(deep=True)
            df_adecuado=modificar_col_geometry(df_copia_profunda)
            st.plotly_chart(mostrar_mapa_densidad(df_adecuado))
        
        if sidebar_distritos_estacion=='Mapa con escala de color':
           st.header("Mapa con escala de color")
           df_copia = df.copy(deep=True)
           gdf_copia = gdf.copy(deep=True)
           _,df_estaciones_por_distrito,gdf_con_estaciones=df_con_distrito_y_gdf_con_estaciones(df_copia,gdf_copia)
           mostrar_mapa_cloropleth(df_estaciones_por_distrito,gdf_con_estaciones)
    
def bicis_por_distrito(df,gdf_copia1):
    col1, col2 = st.columns([1, 3])
    with col1:
      df=botones_para_modificar_df(df)
      opcion_grafico = st.radio(
           "Elige como ver el grafico",
      ('bicis totales', 'ratio'))
      
    with col2:   
        st.plotly_chart(mostrar_grafico_cant_bicis(df,gdf_copia1,opcion_grafico))
        
def densidad_tipos_estaciones(df,gdf,opciones):
    col1, col2 = st.columns([1, 3])
    with col1:
      seleccion=st.multiselect('Selecione distritos:', opciones)
      
    with col2:
        if seleccion==[]:
            st.header('Secciona al menos algún distrito')
        else:
            df_distrito=estaciones_con_distrito(df,gdf)
            plot=mostrar_densidad_tamaño_estación(df_distrito,seleccion)
            st.pyplot(ggplot.draw(plot))
            
def nivel_de_ocupacion_de_las_estaciones(df,gdf,opciones):
    col1, col2 = st.columns([1, 3])
    df_distrito=estaciones_con_distrito(df,gdf)
    df_distrito_1=df_distrito.copy(deep=True)
    with col1:
      seleccion1=st.multiselect('Selecione distritos:', opciones)
      tipo_de_grafico=st.radio(
           "Elige como ver el grafico",
      ('dodge','stack','fill'))
    with col2:  
        if len(seleccion1)<2:
            st.header('Secciona al menos dos distritos')
        else: 
            plot=nivel_ocupacion(df_distrito,seleccion1, formato=tipo_de_grafico)
            st.pyplot(ggplot.draw(plot))
    mapeo_opciones = {0:'Ocupacion baja', 1:'Ocupacion media', 2:'Ocupacion alta'}
    opcion_conclusion = st.radio(
         "Elige que insight ver",
    (0, 1,2),
    format_func=lambda opcion: mapeo_opciones[opcion])
    st.write(f"El distrito con mayor porcentaje es {distrito_nivel_ocupacion(df_distrito_1,'max',opcion_conclusion).iloc[0, 0]} con un {str(round(distrito_nivel_ocupacion(df_distrito_1,'max',opcion_conclusion).iloc[0, 1],4)*100)[:5]}%")
    st.write(f"El distrito con menor porcentaje es {distrito_nivel_ocupacion(df_distrito_1,'min',opcion_conclusion).iloc[0, 0]} con un {str(round(distrito_nivel_ocupacion(df_distrito_1,'min',opcion_conclusion).iloc[0, 1],4)*100)[:5]}%")        


def bicis_por_hectarea(df_con_scrapper):
    col1, col2 = st.columns([1, 3])
    with col1:
      opcion_grafico = st.radio(
           "Elige como ver el grafico",
      ('Por hectareas', 'Hectareas por bici'))
      
    with col2:   
        if opcion_grafico=='Hectareas por bici':
            st.plotly_chart(mostrar_hectareas_bici(df_con_scrapper))
        else:
            st.plotly_chart(mostrar_hectareas(df_con_scrapper))



def menu():
    #Indicamos seleccion
    opcion_principal=st.sidebar.radio('¿Qué ver?',('Mapas', 'Grafico'))
    #Cargamos los datos  
    df,gdf,df_scrapeado=cargar_datos()
    
    #Botones para filtrar
    #df=botones_para_modificar_df(df)
    df_copia1 = df.copy(deep=True)
    gdf_copia1 = gdf.copy(deep=True)
    df_con_scrapper=estaciones_con_info_extra(df_copia1,gdf_copia1,df_scrapeado)
    if opcion_principal=='Mapas':
        menu=st.sidebar.selectbox("Elige mapa un tipo de mapa",
             ("Estaciones", "Distritos","Analisis"))
        if menu=="Estaciones":
           estaciones_option(df) 
        if menu=="Distritos":
           distrito_option(gdf,df)
        if menu=="Analisis":
           analisis_option(df,gdf)
    else:
        menu=st.sidebar.selectbox("Elige un tipo de grafico",
             ('Bicis por distrito',"Densidad tipos de estaciones", "Bicis por persona","Bicis por hectárea",'Nivel de ocupacion de las estaciones'))
        opciones=['Villa de Vallecas', 'Arganzuela', 'Chamartín', 'Usera',
                  'Fuencarral - El Pardo', 'Carabanchel', 'Hortaleza', 'Latina',
                  'San Blas - Canillejas', 'Ciudad Lineal', 'Moncloa - Aravaca',
                  'Centro', 'Retiro', 'Salamanca', 'Tetuán', 'Villaverde',
                  'Puente de Vallecas', 'Vicálvaro', 'Barajas', 'Chamberí',
                  'Moratalaz']
        if menu=='Bicis por distrito':
            bicis_por_distrito(df,gdf_copia1)
        if menu=='Densidad tipos de estaciones':
            densidad_tipos_estaciones(df,gdf,opciones)            
        if menu=='Bicis por persona':
            st.plotly_chart(mostrar_personas_bici(df_con_scrapper))
        if menu=='Bicis por hectárea':
            bicis_por_hectarea(df_con_scrapper)
        if menu=='Nivel de ocupacion de las estaciones':
            nivel_de_ocupacion_de_las_estaciones(df,gdf,opciones)



























