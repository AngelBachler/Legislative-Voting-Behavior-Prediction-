from pandas import json_normalize
import pandas as pd
from .common_utils import safe_serialize
from zeep import Client
from tqdm.notebook import tqdm
import logging
import requests
import xml.etree.ElementTree as ET

def get_votaciones(client, year: int):
    res = client.service.retornarVotacionesXAnno(year)
    if res:
        d = safe_serialize(res) or {}
        df = json_normalize(d)
        return df

def get_detalle(client, idx: int):
    res = client.service.retornarVotacionDetalle(idx)
    if res:
        d = safe_serialize(res) or {}
        df = json_normalize(d)
        df_explode = df.explode(["Votos.Voto"], ignore_index=True)
        df_detalle = json_normalize(df_explode["Votos.Voto"])
        df_detalle = pd.concat([df_explode, df_detalle], axis=1)
        return df_detalle
    return pd.DataFrame()

def build_detalle_periodo(nombre_periodo: str):
    wsdl_url = "https://opendata.camara.cl/camaradiputados/WServices/WSLegislativo.asmx?WSDL"
    client = Client(wsdl_url)

    star_year = nombre_periodo.split("-")[0].strip()
    end_year = nombre_periodo.split("-")[1].strip()
    

    # Construir DataFrame completo de detalles de votaciones
    detalles_list = []
    for year in tqdm(range(int(star_year), int(end_year))):
        try:
            logging.info(f"Procesando año: {year}")
            df_votaciones = get_votaciones(client, year)
            if df_votaciones is None or df_votaciones.empty:
                continue
            
            for idx in tqdm(df_votaciones["Id"].unique()):
                try:
                    df_detalle = get_detalle(client, idx)
                    detalles_list.append(df_detalle)
                except Exception as e:
                    logging.warning(f"Error en votacion {idx}: {e}")
            
        except Exception as e:
            logging.error(f"Error al procesar año {year}: {e}")
    detalles_periodo_df = pd.concat(detalles_list, ignore_index=True)
    return detalles_periodo_df


def procesar_votacion_detalle(xml_content):
    # 1. Parsear el XML
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return pd.DataFrame()

    # 2. Manejar el Namespace (CRÍTICO)
    # El XML tiene xmlns="http://tempuri.org/", así que todas las etiquetas
    # tienen este prefijo invisible. Lo definimos aquí.
    ns = {'ns': 'http://tempuri.org/'}

    # 3. Extraer Datos de Cabecera (Contexto de la votación)
    # Usamos .find() con el prefijo del namespace 'ns:'
    votacion_id = root.find('ns:ID', ns).text
    fecha_hora = root.find('ns:Fecha', ns).text
    boletin = root.find('ns:Boletin', ns).text
    resultado = root.find('ns:Resultado', ns).text
    tipo_votacion = root.find('ns:Tipo', ns).text
    
    # Extraer info de la Sesión (Anidada)
    sesion = root.find('ns:Sesion', ns)
    sesion_id = sesion.find('ns:ID', ns).text if sesion is not None else None
    sesion_num = sesion.find('ns:Numero', ns).text if sesion is not None else None
    sesion_tipo = sesion.find('ns:Tipo', ns).text if sesion is not None else None
    tramite = root.find('ns:Tramite', ns).text if sesion is not None else None
    informe = root.find('ns:Informe', ns).text if sesion is not None else None

    info = {
        # --- Contexto General ---
        'votacion_id': votacion_id,
        'fecha_hora': fecha_hora,
        'boletin': boletin,
        'sesion': sesion_num,
        'sesion_id' : sesion_id,
        'resultado_general': resultado,
        'tipo_votacion': tipo_votacion,
        'tipo_sesion' : sesion_tipo,
        'tramite' : tramite,
        'informe' : informe
    }  

    return info
    