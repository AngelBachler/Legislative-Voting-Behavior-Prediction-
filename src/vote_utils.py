from pandas import json_normalize
import pandas as pd
from .common_utils import safe_serialize
from zeep import Client
from tqdm.notebook import tqdm
import logging

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
    