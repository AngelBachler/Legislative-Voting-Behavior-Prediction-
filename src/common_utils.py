from zeep import Client
import re
import pandas as pd
import unicodedata
import requests
import logging
from ollama import Client as OllamaClient 

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
        
def normalize_string(s):
    """
    Normaliza un string para una mejor comparación (minúsculas, sin acentos, espacios).
    Es vital para un 'fuzzy matching' preciso.
    """
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn')
    return s

def fetch_html(url: str, params: dict = None):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.status_code, r.text, r.url

def get_ollama_client(host='http://127.0.0.1:11434'):
    """
    Inicializa y prueba la conexión con el cliente de Ollama.

    Args:
        host (str, optional): La dirección del servidor de Ollama. 
                                Por defecto 'http://127.0.0.1:11434'.

    Returns:
        ollama.Client: El objeto cliente si la conexión es exitosa.
        None: Si la conexión falla.
    """
    try:
        # 1. Crear el cliente
        client = OllamaClient(host=host)
        
        # 2. Probar la conexión (ligero y rápido)
        #    'client.list()' verifica que el servidor responde.
        client.list() 
        
        logging.info(f"Cliente Ollama conectado exitosamente en {host}")
        return client
        
    except Exception as e:
        # 3. Manejar el error si Ollama no está corriendo
        logging.error(f"ERROR: No se pudo conectar con el servidor de Ollama en {host}.")
        logging.error(f"   Asegúrese de que Ollama esté en ejecución. Detalle: {e}")
        return None