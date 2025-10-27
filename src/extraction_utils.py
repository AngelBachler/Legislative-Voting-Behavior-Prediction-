import pandas as pd
from pandas import json_normalize
import requests
import time
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from .common_utils import safe_serialize, fetch_html


def get_legislaturas(client):
    res = client.service.retornarPeriodosLegislativos()
    d = safe_serialize(res) or {}
    df = pd.json_normalize(d, sep=".")
    df_explode = df.explode("Legislaturas.Legislatura").reset_index(drop=True)
    df_legislaturas = json_normalize(df_explode["Legislaturas.Legislatura"])
    df_legislaturas = pd.concat([df_explode, df_legislaturas], axis=1)
    return df_legislaturas

def get_diputados(client, periodo_id):
    res = client.service.retornarDiputadosXPeriodo(periodo_id)
    d = safe_serialize(res) or {}
    df = pd.json_normalize(d, sep=".")
    df_explode = df.explode("Diputado.Militancias.Militancia").reset_index(drop=True)
    df_diputados = json_normalize(df_explode["Diputado.Militancias.Militancia"])
    df_diputados = pd.concat([df_explode, df_diputados], axis=1)
    return df_diputados



def scrape_bcn_list(base_url, params, pagina_inicial=1, pagina_final=20):
    """
    Scrapea el índice de parlamentarios de la BCN página por página.

    Args:
        base_url (str): La URL base del índice.
        params (dict): Parámetros de la query (ej. para "ex" o "en ejercicio").
        pagina_inicial (int): Página por la que empezar.
        pagina_final (int): Página máxima a revisar.

    Returns:
        pd.DataFrame: Un DataFrame con ['nombre_en_lista', 'url_bcn', 'fuente_params']
    """
    data = []
    # Usar una sesión es más eficiente para múltiples requests
    with requests.Session() as s:
        for n_pagina in range(pagina_inicial, pagina_final + 1):
            params = {**params, "pagina": str(n_pagina)}

            try:
                time.sleep(0.5) # Ser respetuosos con el servidor
                status, text, url = fetch_html(base_url, params=params)

                soup = BeautifulSoup(text, "html.parser")

                items = []
                for li in soup.select("#contenedorResultados li"):
                    a = li.find("a", href=True)
                    if not a:
                        continue
                    url = urljoin(url, a["href"])
                    nombre = a.get_text(strip=True)
                    items.append({"nombre_en_lista": nombre, "url_wiki": url})
                if not items:
                    logging.info(f"Fin del listado en página {n_pagina}")
                    break
                
                for it in items:
                    it["pagina"] = n_pagina
                data.extend(items)
                logging.info(f"Scrapeada página {n_pagina} con {len(items)} items.")
            except requests.RequestException as e:
                logging.error(f"Error al scrapear página {n_pagina} con params {params}: {e}")
                break # Detener si hay un error de red

    return pd.DataFrame(data)