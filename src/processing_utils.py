# --- Contenido para: src/processing_utils.py ---

import pandas as pd
import logging
from pathlib import Path
from rapidfuzz import process, fuzz
from typing import List, Dict, Any
import numpy as np
import re

# Importar nuestra función de normalización de texto
try:
    from .common_utils import normalize_string
except ImportError:
    logging.error("No se pudo importar 'normalize_string' desde 'common_utils'")
    # Definición de respaldo (mala práctica, pero asegura que funcione)
    import unicodedata
    def normalize_string(s):
        if not isinstance(s, str): return ""
        s = s.lower().strip()
        s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        return s

# --- 1. FUNCIÓN DE CARGA ---

def load_all_bio_files(data_dir_raw: Path) -> pd.DataFrame:
    """
    Encuentra todos los archivos 'diputados_bio.csv' en las subcarpetas de 01_raw,
    los carga y los concatena en un solo DataFrame.
    """
    logging.info("Buscando archivos 'diputados_bio.csv'...")
    
    # Usar glob para encontrar todos los archivos
    bio_files = list(data_dir_raw.glob('*/diputados_bio.csv'))
    
    if not bio_files:
        logging.error(f"No se encontraron archivos 'diputados_bio.csv' en {data_dir_raw}")
        return pd.DataFrame()

    logging.info(f"Encontrados {len(bio_files)} archivos. Cargando...")
    
    lista_df = []
    for f in bio_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            # Añadir una columna para saber de qué período/archivo vino
            df['fuente_periodo'] = f.parent.name
            lista_df.append(df)
        except Exception as e:
            logging.error(f"Error al cargar el archivo {f}: {e}")
            
    if not lista_df:
        logging.error("No se pudo cargar ningún DataFrame de biografías.")
        return pd.DataFrame()
        
    df_full = pd.concat(lista_df, ignore_index=True)
    logging.info(f"DataFrame consolidado creado con {len(df_full)} filas.")
    return df_full

# --- 2. FUNCIONES DE ESTANDARIZACIÓN (NORMALIZACIÓN) ---

MAPA_UNIVERSIDADES = {
    "Universidad de Chile": ["u de chile", "universidad chile"],
    "Pontificia Universidad Católica de Chile": ["puc", "universidad catolica", "catolica de chile"],
    "Universidad de Concepción": ["u de concepcion", "udec"],
    "Universidad de Santiago de Chile": ["usach", "universidad de santiago"],
    
}

def standardize_education(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza las columnas 'universidad' y 'maximo_nivel_educativo'
    usando fuzzy matching y mapeo simple.
    """
    df = df_in.copy()
    
    # --- Estandarizar Nivel Educativo ---
    if 'maximo_nivel_educativo' in df.columns:
        logging.info("Estandarizando 'maximo_nivel_educativo'...")
        # Mapeo simple
        level_map = {
            "enseñanza básica": "Básica",
            "enseñanza media": "Media",
            "educación universitaria": "Universitaria",
            "magíster": "Magíster",
            "doctor/a": "Doctorado",
            "doctorado": "Doctorado"
        }
        df['educacion_nivel_clean'] = df['maximo_nivel_educativo'].str.lower().map(level_map)
        # Llenar vacíos (si 'universidad' existe, asumir 'Universitaria')
        df.loc[(df['educacion_nivel_clean'].isna()) & (df['universidad'].notna()), 'educacion_nivel_clean'] = "Universitaria"

    # --- Estandarizar Universidad (con Fuzzy Matching) ---
    if 'universidad' in df.columns:
        logging.info("Estandarizando 'universidad'...")
        # 1. Crear la lista de 'choices' (opciones canónicas)
        choices = list(MAPA_UNIVERSIDADES.keys())
        
        # 2. Crear un mapa de normalización (ej. "u de chile" -> "Universidad de Chile")
        reverse_map = {}
        for key, values in MAPA_UNIVERSIDADES.items():
            for v in values:
                reverse_map[v] = key
        
        # 3. Aplicar
        def find_uni(uni_str):
            if not isinstance(uni_str, str):
                return "Desconocida"
            
            norm_str = normalize_string(uni_str)
            
            # 3.1. Match Perfecto (con mapa de normalización)
            if norm_str in reverse_map:
                return reverse_map[norm_str]
            
            # 3.2. Match Difuso (Fuzzy)
            match, score, _ = process.extractOne(norm_str, choices, scorer=fuzz.token_sort_ratio)
            
            if score >= 90:
                return match
            else:
                return "Otra / Desconocida"

        # (usar .progress_apply si está en un notebook, pero esto es un módulo)
        df['universidad_clean'] = df['universidad'].apply(find_uni)
        
    return df

def standardize_civil_status(status_series: pd.Series) -> pd.Series:
    """
    Limpia y estandariza la columna 'estado_civil' usando mapeo condicional.
    
    Args:
        status_series (pd.Series): La columna 'estado_civil' cruda.
        
    Returns:
        pd.Series: La columna con categorías estandarizadas.
    """
    if not isinstance(status_series, pd.Series):
        logging.error("Input no es una pd.Series. Retornando input.")
        return status_series

    # 1. Normalizar strings para facilitar matching
    norm_col = status_series.apply(normalize_string)

    # 2. Definir las condiciones (de más específico a más general)
    conditions = [
        norm_col.str.contains('conviviente', na=False),
        norm_col.str.contains('casad', na=False),
        norm_col.str.contains('divorciad', na=False),
        norm_col.str.contains('separad', na=False),
        norm_col.str.contains('viud', na=False),
        norm_col.str.contains('solter', na=False),
        
        # 3. Marcar datos irrelevantes o malos para eliminar
        norm_col.str.contains(r'padre de|null', na=False, regex=True)
    ]

    # 4. Definir las categorías limpias correspondientes
    choices = [
        'Conviviente Civil',
        'Casado/a',
        'Divorciado/a',
        'Separado/a',
        'Viudo/a',
        'Soltero/a',
        np.nan  # Marcar como Nulo (NaN)
    ]

    # 5. Aplicar la lógica
    # np.select(condiciones, opciones, default=valor_si_ninguno)
    # Usamos 'default=np.nan' para que cualquier cosa que no calce
    # (como strings vacíos) también se marque como Nulo.
    clean_series = np.select(conditions, choices, default=np.nan)

    # 6. Reemplazar los Nulos (NaN) con una categoría 'Desconocido'
    return pd.Series(clean_series, index=status_series.index).fillna('Desconocido')

def standardize_location(loc_series: pd.Series) -> pd.DataFrame:
    """
    Limpia y estandariza la columna 'lugar_nacimiento'.
    Extrae las features 'ciudad_nac' y 'pais_nac'.
    
    Args:
        loc_series (pd.Series): La columna 'lugar_nacimiento' cruda.
        
    Returns:
        pd.DataFrame: Un DataFrame con las nuevas columnas ('ciudad_nac', 'pais_nac').
    """
    if not isinstance(loc_series, pd.Series):
        logging.error("Input no es una pd.Series.")
        return pd.DataFrame(columns=['ciudad_nac', 'pais_nac'])

    # 1. Crear un DataFrame de trabajo
    df = pd.DataFrame({'lugar_nac_raw': loc_series})

    # 2. Normalizar texto (minúsculas, sin acentos)
    # .astype(str) maneja los 'nan' temporalmente
    df['norm'] = df['lugar_nac_raw'].astype(str).apply(normalize_string)
    df['norm'] = df['norm'].replace(['nan', ''], np.nan) # Restaurar NaNs

    # 3. Definir patrones de limpieza
    
    # Patrón para ruido (ej. "comuna de", "region de", etc.)
    JUNK_REGEX = r'(comuna de |region de |provincia de |ex oficina salitrera |sede de )'
    
    # Patrón para países (extraído de su lista)
    PAISES_LIST = [
        'españa', 'estados unidos', 'holanda', 'bolivia', 'francia', 
        'suecia', 'suiza', 'argentina', 'austria'
    ]
    PAISES_REGEX = f"({'|'.join(PAISES_LIST)})"

    # 4. Extraer País
    # Extraer el país si se menciona explícitamente
    df['pais_nac'] = df['norm'].str.extract(PAISES_REGEX, flags=re.IGNORECASE, expand=False)
    # Si no se extrajo un país, asumir 'Chile'
    df['pais_nac'] = df['pais_nac'].fillna('Chile')

    # 5. Limpiar el string de 'lugar'
    
    # 5a. Remover el ruido (prefijos)
    df['clean'] = df['norm'].str.replace(JUNK_REGEX, '', regex=True, flags=re.IGNORECASE)
    # 5b. Remover el país (si lo encontramos)
    df['clean'] = df['clean'].str.replace(PAISES_REGEX, '', regex=True, flags=re.IGNORECASE)

    # 6. Extraer la Ciudad/Pueblo
    # Nuestra heurística es: tomar la primera parte del string antes de una coma
    # ej. "melipilla, santiago" -> "melipilla"
    # ej. "madrid, " -> "madrid"
    df['ciudad_nac'] = df['clean'].str.split(',').str[0]
    
    # 7. Limpieza final
    df['ciudad_nac'] = df['ciudad_nac'].str.strip()
    
    # Estandarizar "Santiago"
    df.loc[df['ciudad_nac'].str.contains('santiago', na=False), 'ciudad_nac'] = 'Santiago'
    
    # Capitalizar para prolijidad
    df['pais_nac'] = df['pais_nac'].str.capitalize()
    df['ciudad_nac'] = df['ciudad_nac'].str.capitalize()
    
    # 8. Manejar los 'Desconocido'
    df.loc[df['lugar_nac_raw'].isna(), ['pais_nac', 'ciudad_nac']] = 'Desconocido'
    df['ciudad_nac'] = df['ciudad_nac'].fillna('Desconocido')
    
    # 9. Retornar solo las nuevas columnas
    return df[['ciudad_nac', 'pais_nac']]

# --- 3. FUNCIÓN DE DEDUPLICACIÓN ---

def deduplicate_deputies(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Toma el DataFrame consolidado (con duplicados) y crea un
    registro maestro único por diputado.
    """
    # Usaremos el 'Diputado.Id' como la clave única
    if 'Diputado.Id' not in df_full.columns:
        logging.error("No se encuentra 'Diputado.Id', no se puede deduplicar.")
        return df_full
        
    logging.info(f"Deduplicando... Filas antes: {len(df_full)}")
    
    # Estrategia: Ordenar por 'match_score' (calidad de la bio)
    # y 'fecha_ingreso' (más reciente), y luego tomar el MEJOR registro.
    df_sorted = df_full.sort_values(
        by=['match_score', 'fecha_ingreso'], 
        ascending=[False, False]
    )
    
    # 'keep='first'' se queda con el mejor registro (el primero después de ordenar)
    df_master = df_sorted.drop_duplicates(subset=['Diputado.Id'], keep='first')
    
    logging.info(f"Filas después de deduplicar: {len(df_master)}")
    return df_master