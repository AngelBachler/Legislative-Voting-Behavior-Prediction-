from zeep import Client
import pandas as pd
from pandas import json_normalize
import re
import requests
import time
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def listar_ops(wsdl_url):
    """
    Lista las operaciones disponibles en un servicio SOAP dado su WSDL.
    Útil para explorar endpoints de la Cámara de Diputados o del Senado.
    """
    c = Client(wsdl=wsdl_url)
    for svc in c.wsdl.services.values():
        for port in svc.ports.values():
            binding = port.binding
            for name, op in binding._operations.items():
                args = []
                if op.input and op.input.body and op.input.body.type:
                    args = [elt[0] for elt in op.input.body.type.elements]  # nombres de parámetros
                print(f"{name}({', '.join(args)})")

def sanitize_filename(name):
    """Limpia nombres de carpeta/archivo."""
    if pd.isna(name) or str(name).strip() == "":
        return "sin_periodo"
    return re.sub(r"[^\w\-]+", "_", str(name)).strip("_")

def safe_serialize(obj):
    """
    Serializa objetos Zeep o SOAP a estructuras Python básicas (dict, list, str, etc.)
    para poder normalizarlos con pandas.json_normalize sin errores.
    """
    # tipos base
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [safe_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}

    # intentar con zeep.helpers.serialize_object
    try:
        from zeep.helpers import serialize_object
        ser = serialize_object(obj)
        if ser is not None:
            return safe_serialize(ser)  # recursivo por si hay anidaciones
    except Exception:
        pass

    # objetos Zeep suelen tener __values__
    vals = getattr(obj, "__values__", None)
    if isinstance(vals, dict):
        return {k: safe_serialize(v) for k, v in vals.items()}

    # fallback genérico
    try:
        return {k: safe_serialize(v) for k, v in vars(obj).items()}
    except Exception:
        return obj

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

def fetch_html(url: str, params: dict = None):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.status_code, r.text, r.url

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
                text, url = fetch_html(base_url, params=params)

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