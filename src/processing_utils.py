import pandas as pd
import logging
from pathlib import Path
from rapidfuzz import process, fuzz
import numpy as np
import re
import ast
from tqdm import tqdm
from bertopic import BERTopic

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

def extract_last_colegio(colegio_str: str) -> str | None:
    """
    Toma un string que representa una lista de colegios (del LLM)
    y extrae el ÚLTIMO colegio de la lista (asumiendo egreso).
    
    Input: "['Colegio A', 'Colegio B']"
    Output: "Colegio B"
    """
    # 1. Manejar NaNs o valores que no sean string
    if not isinstance(colegio_str, str):
        return None  # Se convertirá en NaN

    try:
        # 2. Parsear de forma segura el string a una lista de Python
        # ast.literal_eval es el método seguro para parsear
        # estructuras de Python (listas, dicts) desde strings.
        colegios_list = ast.literal_eval(colegio_str)
        
        # 3. Validar y extraer
        if isinstance(colegios_list, list) and len(colegios_list) > 0:
            # Retorna el último ítem, limpio de espacios
            return str(colegios_list[-1]).strip()
        else:
            # Para listas vacías '[]'
            return None
            
    except (ValueError, SyntaxError):
        # Para strings malformados o que no son listas
        logging.warning(f"Error al parsear string de colegio: {colegio_str}")
        return None

UNI_MAP = {
    # Chilenas (Canónico: [variaciones normalizadas])
    'Pontificia Universidad Católica de Chile': ['pontificia universidad catolica de chile', 'universidad catolica de chile', 'puc', 'universidad catolica', 'pontificia universidad catolica', 'pontificia universidad catolica de santiago', 'pontifica universidad catolica de chile'],
    'Universidad de Chile': ['universidad de chile', 'u de chile', 'uchile', 'escuela de teatro de la universidad de chile'],
    'Universidad de Santiago de Chile': ['universidad de santiago de chile', 'universidad de santiago', 'usach', 'universidad tecnica del estado', 'ute'],
    'Universidad de Concepción': ['universidad de concepcion', 'udec'],
    'Universidad Técnica Federico Santa María': ['universidad tecnica federico santa maria', 'utfsm', 'universidad federico santa maria'],
    'Universidad Austral de Chile': ['universidad austral de chile', 'universidad austral de valdivia'],
    'Universidad Adolfo Ibáñez': ['universidad adolfo ibanez', 'uai'],
    'Universidad de Los Andes': ['universidad de los andes (chile)', 'universidad de los andes'],
    'Universidad del Desarrollo': ['universidad del desarrollo', 'udd'],
    'Universidad Diego Portales': ['universidad diego portales', 'udp'],
    'Universidad Andrés Bello': ['universidad andres bello', 'unab', 'nacional andres bello'],
    'Universidad Católica del Norte': ['universidad catolica del norte'],
    'Universidad de La Frontera': ['universidad de la frontera', 'ufro'],
    'Universidad de Valparaíso': ['universidad de valparaiso', 'uv'],
    'Universidad de Artes, Ciencias y Comunicación (UNIACC)': ['universidad de artes y las ciencias sociales',
                                                               'universidad de artes y ciencias de la comunicacion'],
    # Institutos
    'Universidad Tecnológica de Chile INACAP': ['inacap', 'instituto nacional de capacitacion', 'universidad tecnologica (inacap)', 'instituto profesional inacap', 'instituto inacap',
                                                'universidad tecnológica'],
    'Instituto Profesional Agrario Adolfo Matthei': ['instituto superior de agricultura adolfo matthei', 'instituto profesional agrario adolfo matthei'],
    'Instituto Profesional Duoc UC': ['duoc uc'],
    
    # Extranjeras (Agrupadas)
    'Universidad Extranjera': [
        'universidad de california', 'georgetown', 'complutense de madrid', 'universidad de cuenca', 'universitat basel', 'universidad de bilbao-espana', 
        'universita degli studi di milano', 'instituto tecnico de estocolmo', 'universidad nacional autonoma de mexico', 'escuela nacional de antropologia e historia de mexico', 
        'universidad internacional de cataluna, espana', 'universidad de heidelberg', 'universidad mayor de san simon (umss)', 'madrid, espana', 
        'cochabamba, bolivia', 'chicago, estados unidos', 'francia', 'malmo, suecia', 'zurich, suiza', 
        'buenos aires, argentina', 'viena - hietzing, austria', 'washington dc, estados unidos'
    ],
    
    # Ruido
    'Desconocida': ['nombre universidad', 'icce santiago', 'nan', '']
}

# --- 2. LISTA CANÓNICA (Fallback para Fuzzy Match) ---
# (Esta es TU lista de 'UNIVERSIDADES')
UNIVERSIDADES_CANONICAS = [
    "Pontificia Universidad Católica de Chile", "Universidad de Chile",
    "Universidad de Santiago de Chile", "Universidad de Concepción",
    "Universidad Católica de Valparaíso", "Universidad Adolfo Ibáñez",
    "Universidad Técnica Federico Santa María", "Universidad de Valparaíso",
    "Universidad Tecnológica de Chile INACAP", "Universidad de Los Andes",
    "Universidad del Desarrollo", "Universidad Alberto Hurtado",
    "Universidad Andrés Bello", "Universidad Autónoma de Chile",
    "Universidad Arturo Prat", "Universidad Austral de Chile",
    "Universidad de La Frontera", "Universidad de Magallanes",
    "Universidad de Talca", "Universidad Católica del Norte",
    "Universidad Católica del Maule", "Universidad Católica de Temuco",
    "Universidad Católica de la Santísima Concepción", "Universidad Bernardo O'Higgins",
    "Universidad Central de Chile", "Universidad de Antofagasta",
    "Universidad de Atacama", "Universidad de La Serena",
    "Universidad del Bío-Bío", "Universidad de Playa Ancha de Ciencias de la Educación",
    "Universidad de Tarapacá", "Universidad de Viña del Mar",
    "Universidad Diego Portales", "Universidad Finis Terrae",
    "Universidad Mayor", "Universidad Metropolitana de Ciencias de la Educación",
    "Universidad Miguel de Cervantes", "Universidad San Sebastián",
    "Instituto Profesional AIEP", "Instituto Profesional ARCOS",
    "Instituto Profesional de Chile", "Instituto Profesional Duoc UC",
    "Instituto Profesional Escuela de Contadores Auditores de Santiago",
    "Instituto Profesional Agrario Adolfo Matthei", "Instituto Nacional de Capacitación Profesional (INACAP)",
    "Instituto Profesional Iplacex", "Instituto Profesional Santo Tomás",
    "Instituto Profesional Virginio Gómez", "Instituto Profesional CIISA",
    "Instituto Profesional Galdámez", "Universidad de California",
    "Universidad de Cuenca", "Universidad de Georgetown",
    "Universidad Complutense de Madrid", "Universität Basel",
    "Universidad de Bilbao-España", "Università degli Studi di Milano",
    "Universidad Real", "Instituto Técnico de Estocolmo",
    "Universidad Nacional Autónoma de México",
    "Escuela Nacional de Antropología e Historia de México",
    "Universidad Internacional de Cataluña, España",
    "Universidad de Heidelberg", "Universidad Mayor de San Simón",
    "Universidad Católica Raúl Silva Henríquez",
    "Escuela de Comunicación Mónica Herrera",
    "Universidad Contemporánea de Arica",
    "Instituto Bancario Guillermo Subercaseaux", "Universidad Pedro de Valdivia", "Instituto Vicente Pérez Rosales",
    "Universidad de Los Lagos", "Instituto Nacional de Capacitación Profesional", "Universidad de La República",
    "Universidad Gabriela Mistral", "Universidad Mariano Egaña",
    "Universidad Católica de la Santísima Concepción", "Universidad Bolivariana", "Universidad de Artes, Ciencias y Comunicación (UNIACC)",
    "Universidad Academia de Humanismos Cristiano", "Instituto Profesional Diego Portales", "Universidad Iberoamericana de Ciencias y Tecnología",
    "Universidad de Antofagasta", 
    "Universidad del Pacífico",    # Añadir extranjeras genéricas
    "Universidad Extranjera"
]


# --- 3. FUNCIÓN AUXILIAR DE MATCHING ---
def _find_best_uni_match(raw_entry: str, choices: list, reverse_map: dict) -> str:
    """
    Toma un string crudo y usa una estrategia HÍBRIDA para encontrar el mejor match canónico.
    """
    if not isinstance(raw_entry, str):
        return "Desconocida"
    
    # 1. Extraer la *primera* universidad de la lista
    primary_entry = raw_entry.split(',')[0].split('(')[0].strip()
    norm_entry = normalize_string(primary_entry)
    
    if norm_entry == "" or norm_entry == "nan":
         return "Desconocida"

    # 2. JERARQUÍA 1: Buscar en el Mapa Manual (Rápido y Preciso)
    match = reverse_map.get(norm_entry)
    if match:
        return match # ¡Encontrado!

    mejores_candidatos = process.extract(
    query=norm_entry,
    choices=choices,
    scorer=fuzz.WRatio,
    processor=normalize_string,
    limit=5
)

    mejor_nombre = None
    mejor_score_final = 0

    for candidato_nombre, score_paso1, _ in mejores_candidatos:
        
        score_paso2 = fuzz.partial_token_set_ratio(
            normalize_string(norm_entry),
            normalize_string(candidato_nombre)
        )
        
        score_final = max(score_paso1, score_paso2)
        
        if score_final > mejor_score_final:
            mejor_score_final = score_final
            mejor_nombre = candidato_nombre
    if mejor_score_final >= 90:
        return mejor_nombre
    else:
        return "Otra / Desconocida"

# --- 4. FUNCIÓN PRINCIPAL REFACTORIZADA ---
def standardize_education(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    (VERSIÓN HÍBRIDA Y OPTIMIZADA)
    Estandariza 'universidad' usando Mapeo Manual y Mapeo Fuzzy "Una Sola Vez".
    """
    df = df_in.copy()
    
    # --- A. Estandarizar Nivel Educativo (sin cambios) ---
    if 'maximo_nivel_educativo' in df.columns:
        logging.info("Estandarizando 'maximo_nivel_educativo'...")
        level_map = {
            "enseñanza básica": "Básica", "enseñanza media": "Media",
            "educación universitaria": "Universitaria", "magíster": "Magíster",
            "doctor/a": "Doctorado", "doctorado": "Doctorado"
        }
        df['educacion_nivel_clean'] = df['maximo_nivel_educativo'].str.lower().map(level_map)
        df.loc[(df['educacion_nivel_clean'].isna()) & (df['universidad'].notna()), 'educacion_nivel_clean'] = "Universitaria"

    # --- B. Estandarizar Universidad ---
    if 'universidad' in df.columns:
        logging.info("Estandarizando 'universidad' con estrategia 'map-once'...")
        
        # 1. Crear el mapa de búsqueda manual (rápido)
        reverse_map = {}
        for key, values in UNI_MAP.items():
            for v in values:
                reverse_map[v] = key
        
        # 2. Obtener los valores únicos (LA OPTIMIZACIÓN)
        unique_universities = df['universidad'].unique()
        logging.info(f"Se encontraron {len(unique_universities)} valores únicos de universidad.")

        # 3. Construir el Mapeo (solo en los únicos)
        logging.info("Construyendo mapa de traducción (Manual + Fuzzy)...")
        map_dict = {}
        for raw_name in tqdm(unique_universities, desc="Creando Mapa Fuzzy"):
            map_dict[raw_name] = _find_best_uni_match(
                raw_name, 
                UNIVERSIDADES_CANONICAS, 
                reverse_map
            )

        # 4. Aplicar el mapeo a todas las filas (muy rápido)
        logging.info("Aplicando mapa a todas las filas...")
        df['universidad_clean'] = df['universidad'].map(map_dict)

        # 5. Crear 'universidad_tipo'
        logging.info("Clasificando tipo de universidad...")
        def get_tipo(uni_clean):
            if uni_clean in ["Desconocida", "Otra / Desconocida", "Universidad Extranjera"]:
                return uni_clean
            if "Instituto" in uni_clean or "INACAP" in uni_clean or "AIEP" in uni_clean or "Duoc" in uni_clean:
                return "Instituto/Técnico"
            return "Universidad Chilena"
        
        df['universidad_tipo'] = df['universidad_clean'].apply(get_tipo)
        
    return df

CAREER_MAP_REGEX = {
    # --- 1. CATEGORÍAS MÁS ESPECÍFICAS (VAN PRIMERO) ---
    
    'Geografía': r'geograf[ií]a|geograf(o|a)',
    'Asistente Social': r'asistente social|trabaj(o|adora) social', # Movido desde Ciencias Sociales
    'Licenciatura en matemáticas': r'licenciatura en matem[áa]ticas|matem[áa]ticas', # Movido desde Ciencias Sociales
    'Teología': r'teolog(o|a)|teologia', # Movido desde Ciencias Sociales
    'Biología Marina': r'biolog(o|a) marino|ciencias del mar|oceanograf[ií]a',
    'Marketing': r'marketing|comercio internacional|publicidad y marketing',
    'Ingeniería en Metalurgia': r'ingenier([íi]|)(o|a|) en metalurgia|metalurgista',

    # --- 2. CATEGORÍAS GENERALES AMPLIAS ---
    
    'Ingeniería Comercial': r'economia|ingenier([íi]|)(o|a|) comercial|ciencias de la administracion|ciencias economicas|gestion de empresas|administracion de empresas|administracion y direccion de empresas|ingenier(o|a) en administraci[óo]n|direcci[óo]n y gesti[óo]n servicios|contador|auditor|gestion gerencial|administraci[óo]n bancaria y contabilidad',
    'Ingeniería Civil': r'ingenier([íi]|)(o|a|) civil|(?<!ejecucion\s)ingenier([íi]|)(o|a|) industrial|obras civiles|construccion civil|ingenier[íi]a civil industrial',
    'Ingeniería (Ejecución/Técnica)': r'ingenier([íi]|)(o|a|) en ejecucion|ejecucion|tecnico|laboratista|geomensura|quimico industrial|electrico|electronica|mecanico|tecnica en quimica',
    'Derecho': r'derecho|abogad(o|a)|jur[íi]dica|leyes|ciencias jur[íi]dicas',
    'Salud': r'm[eé]dic(o|a)|cirujan(o|a)|medicina|odontolog|enfermer(o|a)|obstetricia|matr[oó]n|kinesiolog|fonoaudiolog|psicolog[ií]a|psiquiatr[ií]a|enfermer(o|a)|enfermeria',
    'Educación / Pedagogía': r'profesor(|a)|pedagog[ií]a|educador(a)|educacion|licenciado en educacion|ense[ñn]anza',
    'Agronomía / Veterinaria': r'agronomo|agronomia|forestal|agricola|veterinaria|pecuaria',
    
    # (Llave consolidada y corregida)
    'Administración Pública': r'administraci[óo]n publica|administrador(|a) p[úu]blic(o|a)|ciencias politicas y administrativas|oficial civil|programas sociales', 
    
    'Periodismo / Comunicaciones': r'periodis|comunicaci[oó]n social|comunicador social|relaciones publicas|publicidad',
    'Arquitectura / Diseño': r'arquitectura|arquitect(o|a)|dise[ñn]o|dise[ñn]ador industrial',
    'Arte / Actuación': r'arte|actor|actriz|actuacion|teatro|musica|danza|gestion cultural',

    # --- 3. CATEGORÍA "CATCH-ALL" (VA AL FINAL) ---
    
    # (Limpiada de los duplicados que ahora están arriba)
    'Ciencias Sociales': r'soci[óo]log(o|a)|ciencias pol[íi]ticas|sociolog[ií]a|antropolog[ií]a|ciencia politica|cientista politico|historia|filosof[ií]a|letras|literatura|bachillerato en humanidades'
}

# --- 2. FUNCIÓN DE LIMPIEZA AUXILIAR ---
def _clean_raw_career(raw_entry: str) -> list:
    """
    Toma un string crudo (que puede ser una lista-como-string)
    y lo convierte en una lista limpia de strings.
    
    Input: "['Derecho', 'Magíster en...']" -> Output: ["derecho", "magister en..."]
    Input: "Psicología, Derecho" -> Output: ["psicologia", "derecho"]
    """
    if not isinstance(raw_entry, str):
        return []
    
    # Intentar parsear como lista (ej. "['...']")
    try:
        parsed_list = ast.literal_eval(raw_entry)
        if isinstance(parsed_list, list):
            return [normalize_string(item) for item in parsed_list]
    except (ValueError, SyntaxError):
        # No era una lista-como-string, tratar como string simple
        pass
    
    # Es un string simple, normalizar y separar por comas
    # ej. "Psicología, Derecho"
    entries = re.split(r',|;', raw_entry) # Separar por coma o punto y coma
    return [normalize_string(entry) for entry in entries if entry.strip()]


# --- 3. FUNCIÓN DE ESTANDARIZACIÓN (NUEVA) ---
def standardize_career(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza la columna 'carrera' usando Mapeo Regex "Una Sola Vez".
    Crea 'carrera_clean_1' y 'carrera_clean_2' (para multi-carreras).
    """
    df = df_in.copy()
    
    if 'carrera' not in df.columns:
        logging.warning("No se encontró la columna 'carrera'. Saltando.")
        return df

    logging.info("Estandarizando 'carrera' con estrategia 'map-once'...")

    # 1. Obtener valores únicos (LA OPTIMIZACIÓN)
    unique_careers_raw = df['carrera'].unique()
    logging.info(f"Se encontraron {len(unique_careers_raw)} valores únicos de carrera.")

    # 2. Construir el Mapeo (solo en los únicos)
    logging.info("Construyendo mapa de traducción de carreras (Regex)...")
    map_dict = {}

    for raw_entry in tqdm(unique_careers_raw, desc="Mapeando Carreras"):
        
        # 2a. Limpiar el string a una lista (ej. ["psicologia", "derecho"])
        cleaned_list = _clean_raw_career(raw_entry)
        
        # 2b. Buscar matches de regex en esa lista
        matches = []
        for term in cleaned_list:
            if not term:
                continue
            
            found = False
            for category, regex_pattern in CAREER_MAP_REGEX.items():
                if re.search(regex_pattern, term):
                    if category not in matches: # Evitar duplicados
                        matches.append(category)
                    found = True
                    break # Asumir que un término solo pertenece a una categoría
        
        # 2c. Asignar al mapa
        if not matches:
            map_dict[raw_entry] = ["Desconocida"]
        else:
            map_dict[raw_entry] = matches
    
    # 3. Aplicar el mapeo a todas las filas (muy rápido)
    logging.info("Aplicando mapa a todas las filas...")
    # Esto crea una columna donde cada celda es una LISTA de carreras (ej. ['Salud', 'Derecho'])
    df['carrera_clean_list'] = df['carrera'].map(map_dict)

    # 4. (Ingeniería de Features) Dividir en columnas separadas
    # Esto es mucho más útil para el modelamiento
    df['carrera_clean_1'] = df['carrera_clean_list'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "Desconocida")
    df['carrera_clean_2'] = df['carrera_clean_list'].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else None)

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

def create_age_features(df: pd.DataFrame, 
                        start_date_col: str, 
                        birth_date_col: str) -> pd.DataFrame:
    """
    Calcula la 'edad' y el 'rango_etario' de un diputado al
    inicio de un período.

    Args:
        df (pd.DataFrame): El DataFrame que contiene las fechas.
        start_date_col (str): Nombre de la col. con la fecha de inicio del período.
        birth_date_col (str): Nombre de la col. con la fecha de nacimiento limpia.

    Returns:
        pd.DataFrame: El DataFrame original con 2 nuevas columnas: 'edad' y 'rango_etario'.
    """
    logging.info(f"Creando features 'edad' y 'rango_etario'...")
    
    # 1. Asegurar que sean datetime
    df[start_date_col] = pd.to_datetime(df[start_date_col], errors='coerce')
    df[birth_date_col] = pd.to_datetime(df[birth_date_col], errors='coerce')

    # 2. Calcular la edad (en años)
    time_diff = df[start_date_col] - df[birth_date_col]
    
    seconds_in_year = 365.25 * 24 * 60 * 60
    
    with np.errstate(invalid='ignore'): # Ignorar si hay NaT
        df['edad'] = time_diff.dt.total_seconds() / seconds_in_year

    # Redondear y convertir a entero (Int64 maneja NaNs)
    df['edad'] = df['edad'].astype(float).round().astype('Int64')

    # 3. Crear el 'rango_etario'
    bins = [18, 29, 39, 49, 59, 69, 110] # 18-29, 30-39, etc.
    labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70+']

    df['rango_etario'] = pd.cut(
        df['edad'],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True
    )
    
    # Reemplazar 'NaN' en la categoría con 'Desconocido'
    df['rango_etario'] = df['rango_etario'].cat.add_categories('Desconocido').fillna('Desconocido')
    
    return df

VOTE_MAP = {
    'aprueba': 1.0,
    'rechaza': 0.0,
    'abstencion': 0.5,
    'abstiene': 0.5,
    'pareo': np.nan,    # El pareo no es un voto, lo marcamos como Nulo
    'no vota': np.nan,  # 'No vota' tampoco
    'articulo 5': 0.5 # (Asumimos Abstención, puede cambiarlo)
}

VOTE_MAP = {
    'afirmativo': 1.0,
    'en contra': 0.0,
    'abstencion': 0.5,
    'abstiene': 0.5,
    'pareo': np.nan,    
    'no vota': np.nan,  
    'articulo 5': 0.5 
}

# --- REEMPLAZAR LA VERSIÓN ANTIGUA CON ESTA ---
def process_votaciones_chunk(df_raw: pd.DataFrame, periodo_nombre: str) -> pd.DataFrame:
    """
    (VERSIÓN CORREGIDA)
    Limpia, mapea, y selecciona columnas de un chunk (un 'detalle.csv')
    usando los nombres de columna reales.
    """
    df = df_raw.copy()
    
    # --- (NUEVO) PASO 1: FILTRAR SOLO Proyectos de Ley ---
    logging.info(f"Filtrando 'Proyectos de Ley' para {periodo_nombre}...")
    
    tipo_col_raw = None
    if 'Tipo._value_1' in df.columns:
        tipo_col_raw = df['Tipo._value_1']
    
    if tipo_col_raw is not None:
        # Convertir a string para estar seguros
        tipo_votacion_series = tipo_col_raw.astype(str).str.lower()
        
        # Filtro: quedarse solo con filas que contengan 'proyecto de ley'
        keep_mask = tipo_votacion_series.str.contains('proyecto de ley', na=False)
        df = df[keep_mask]
        
        if df.empty:
            logging.info(f"No se encontraron votaciones de 'Proyectos de Ley' en el chunk {periodo_nombre}.")
            return pd.DataFrame() # Retornar vacío
        
        logging.info(f"Se encontraron {len(df)} votaciones de 'Proyectos de Ley'.")
    else:
        logging.warning(f"No se encontraron columnas 'Tipo' en {periodo_nombre}. No se pudo filtrar.")
    
    # 1. Map Votos (Usando las columnas correctas)
    #    Vamos a consolidar 'OpcionVoto.Valor' y 'OpcionVoto._value_1'
    if 'OpcionVoto.Valor' not in df.columns and 'OpcionVoto._value_1' not in df.columns:
        logging.warning(f"No se encontró 'OpcionVoto.Valor' o '_value_1' en {periodo_nombre}.")
        return pd.DataFrame()

    # Priorizar la columna de texto 'Valor' si existe
    if 'OpcionVoto.Valor' in df.columns:
        df['voto_valor'] = df['OpcionVoto.Valor']

    # 2. Extract Boletin ID (Crucial para NLI)
    #    (Usando la columna 'Descripcion')
    if 'Descripcion' in df.columns:
        df['boletin_id'] = df['Descripcion'].astype(str).str.extract(
            r"Bolet[ií]n\s*N[°º]?\s*(\d+-\d+)"
        )[0]
    else:
        df['boletin_id'] = np.nan

    # 3. Select and Rename (Usando los nombres de columna reales)
    cols_to_keep = {
        # Columna Cruda     -> Columna Limpia
        'Id':               'votacion_id',
        'Fecha':            'fecha_votacion',
        'TotalSi':        'total_si',
        'TotalNo':        'total_no',
        'TotalAbstencion':'total_abstenciones',
        'TotalDispensado':'total_dispensado',
        'Quorum._value_1': 'quorum',
        'Diputado.Id':      'diputado_id',
        'voto_valor':       'voto_valor',     # Creada en Paso 1
        'boletin_id':       'boletin_id'     # Creada en Paso 2
    }
    
    # (Verificar si faltan columnas en este chunk)
    final_cols = {}
    for col_raw, col_clean in cols_to_keep.items():
        if col_raw in df.columns:
            final_cols[col_raw] = col_clean
        elif col_raw in df.index: # Para 'voto_valor' y 'boletin_id'
             final_cols[col_raw] = col_clean
        else:
            logging.warning(f"Col. faltante '{col_raw}' en {periodo_nombre}. Se rellenará con NaT/NaN.")
            df[col_raw] = np.nan # Añadirla como NaN para que no falle el rename
            final_cols[col_raw] = col_clean
            
    # Filtrar solo las columnas que existen
    existing_cols_raw = [col for col in final_cols.keys() if col in df.columns]
    df_clean = df[existing_cols_raw]
    df_clean = df_clean.rename(columns=final_cols)
    
    # 4. Convert types
    df_clean['fecha_votacion'] = pd.to_datetime(df_clean['fecha_votacion'], errors='coerce')
    df_clean['diputado_id'] = pd.to_numeric(df_clean['diputado_id'], errors='coerce').astype('Int64')
    df_clean['votacion_id'] = pd.to_numeric(df_clean['votacion_id'], errors='coerce').astype('Int64')
    df_clean['boletin_id'] = df_clean['boletin_id'].astype(str).replace('nan', np.nan) 
    
    # 5. Add period key
    df_clean['periodo'] = periodo_nombre
    
    # 6. Drop rows with no vote (e.g., Pareo o Nulos) o llaves nulas
    df_clean = df_clean.dropna(subset=['voto_valor', 'diputado_id', 'votacion_id'])
    
    return df_clean


def load_all_bulletin_files(data_dir_raw: Path) -> pd.DataFrame:
    """
    Encuentra todos los 'boletines.csv' en las subcarpetas de 01_raw,
    los carga, los concatena y DEDUPLICA por 'boletin_id'.
    """
    logging.info("Buscando archivos 'boletines.csv'...")
    
    # Usar glob para encontrar todos los archivos
    bulletin_files = list(data_dir_raw.glob('*/boletines.csv'))
    
    if not bulletin_files:
        logging.error(f"No se encontraron archivos 'boletines.csv' en {data_dir_raw}")
        return pd.DataFrame()

    logging.info(f"Encontrados {len(bulletin_files)} archivos. Cargando...")
    
    lista_df = []
    for f in bulletin_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            lista_df.append(df)
        except Exception as e:
            logging.error(f"Error al cargar el archivo {f}: {e}")
            
    if not lista_df:
        logging.error("No se pudo cargar ningún DataFrame de boletines.")
        return pd.DataFrame()
        
    df_full = pd.concat(lista_df, ignore_index=True)
    logging.info(f"DataFrame consolidado creado con {len(df_full)} filas.")
    
    # --- (PASO CRÍTICO) Deduplicar ---
    # Un mismo boletín puede haber sido extraído en múltiples períodos.
    # Nos quedamos con la primera aparición.
    logging.info("Deduplicando por 'boletin_id'...")
    df_clean = df_full.drop_duplicates(subset=['boletin_id'], keep='first')
    logging.info(f"DataFrame deduplicado tiene {len(df_clean)} boletines únicos.")
    
    return df_clean

