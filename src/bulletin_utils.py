import pandas as pd
import logging
import requests
import json
from tqdm.notebook import tqdm
from pathlib import Path
from zeep import Client
import xmltodict

from .extraction_utils import sanitize_filename 


def get_boletin(boletin_id: str) -> pd.DataFrame:
    
    BASE_URL = "https://tramitacion.senado.cl/wspublico/tramitacion.php"
    params = {"boletin": boletin_id}
    
    try:
        # 1. Obtener el XML
        response = requests.get(BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        
        # 2. Parsear el XML
        # 'force_list' asegura que 'autor' y 'materia' sean siempre listas,
        # incluso si solo hay uno. Esto evita errores de parseo.
        data_dict = xmltodict.parse(response.text, 
                                    force_list=('autor', 'materia'))
        
        # 3. Navegar a la raíz del proyecto
        proyecto = data_dict.get('proyectos', {}).get('proyecto')
        if not proyecto:
            logging.warning(f"No se encontró <proyecto> en la respuesta para {boletin_id}")
            return pd.DataFrame()

        # 4. Extraer Metadatos (Descripción)
        descripcion = proyecto.get('descripcion', {})
        out = {
            'boletin_id': descripcion.get('boletin'),
            'titulo': descripcion.get('titulo'),
            'fecha_ingreso': descripcion.get('fecha_ingreso'),
            'iniciativa': descripcion.get('iniciativa'),
            'camara_origen': descripcion.get('camara_origen'),
            'etapa': descripcion.get('etapa'),
            'leynro': descripcion.get('leynro'),
            'link_mensaje_mocion': descripcion.get('link_mensaje_mocion')
        }
        
        # 5. Extraer Autores (Aplanar lista)
        autores_list = []
        if proyecto.get('autores') and proyecto['autores'].get('autor'):
            # Gracias a 'force_list', 'autor' siempre es una lista
            for autor in proyecto['autores']['autor']:
                autores_list.append(autor.get('PARLAMENTARIO'))
        out['autores_json'] = json.dumps(autores_list) # Guardar como string JSON

        # 6. Extraer Materias (Aplanar lista)
        materias_list = []
        if proyecto.get('materias') and proyecto['materias'].get('materia'):
            # 'materia' siempre es una lista
            for mat in proyecto['materias']['materia']:
                materias_list.append(mat.get('DESCRIPCION'))
        
        # Guardamos ambas versiones:
        # 'materias_str' para el LLM (como pedía su prompt)
        out['materias_str'] = "; ".join(materias_list) 
        # 'materias_json' para análisis futuro
        out['materias_json'] = json.dumps(materias_list) 
        
        # 8. Convertir a DataFrame de una fila
        df = pd.DataFrame([out])
        df['boletin_id_consultado'] = boletin_id
        return df

    except requests.RequestException as e:
        logging.error(f"Error de red en get_boletin (XML) para {boletin_id}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error de parseo en get_boletin (XML) para {boletin_id}: {e}", exc_info=True)
        return pd.DataFrame()

    except Exception as e:
        logging.error(f"Error en get_boletin para {boletin_id}: {e}", exc_info=True)
        return pd.DataFrame()
    
def extraer_materia(client: Client, materias: str, model_name: str) -> dict | None:
    
    MATERIAS_SYSTEM_PROMPT = """
Eres un analista experto del Congreso Nacional de Chile. Tu tarea es clasificar una lista de materias legislativas en una o más categorías temáticas generales.

**Reglas Estrictas:**
1.  **JSON Únicamente:** Tu respuesta debe ser **SOLAMENTE** un objeto JSON válido, sin texto introductorio (como "Aquí está el JSON:").
2.  **Multi-Etiqueta:** Un proyecto puede pertenecer a múltiples ámbitos. Asigna *todos* los que apliquen. (Ej. "REAJUSTE DE REMUNERACIONES" es ["Trabajo y Previsión", "Economía y Hacienda"]).
3.  **Categorías Válidas:** Usa **SOLAMENTE** las siguientes 13 categorías:
    * Educación
    * Salud
    * Trabajo y Previsión
    * Economía y Hacienda
    * Seguridad y Justicia
    * Medio Ambiente y Energía
    * Vivienda y Urbanismo
    * Gobierno y Política
    * Relaciones Exteriores
    * Cultura y Deporte
    * Transporte y Telecomunicaciones
    * Derechos Humanos y Género
    * no cumple
4.  **"no cumple":** Usa `["no cumple"]` *solo si* ninguna otra categoría aplica.

**Formato JSON Requerido:**
{
  "materias_originales": "<texto exacto recibido>",
  "ambitos_detectados": ["<ámbito1>", "<ámbito2>", ...]
}
"""
    # --- 2. Validación de Entradas (Guard Clauses) ---
    if not client:
        logging.warning("extraer_materia_llm: Cliente Ollama no es válido (None).")
        return None
    
    if not isinstance(materias, list) or len(list) < 1:
        # No molestar al LLM si el texto es muy corto o inválido
        logging.info("extraer_materia_llm: Texto de biografía omitido (demasiado corto o inválido).")
        return None

    # --- 3. Ejecución y Parseo (Bloque Try/Except) ---
    try:
        # Truncar el texto para asegurar que quepa en el contexto del modelo
        texto_truncado = materias
        
        # 3.1. Llamar al LLM
        response = client.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': MATERIAS_SYSTEM_PROMPT},
                {'role': 'user', 'content': texto_truncado}
            ],
            options={'temperature': 0.0} # Queremos respuestas fácticas, no creativas
        )
        
        # 3.2. Obtener la respuesta (string que *contiene* JSON)
        json_string = response['message']['content'].strip()
        
        # 3.3. Limpiar el JSON (Remover markdown y texto extra)
        if json_string.startswith('```json'):
            json_string = json_string[7:].strip()
        if json_string.endswith('```'):
            json_string = json_string[:-3].strip()
        
        # Asegurarse de que empieza con { (a veces añaden texto antes)
        json_start_index = json_string.find('{')
        if json_start_index != -1:
            json_string = json_string[json_start_index:]
            
        # 3.4. Parsear el string a un diccionario Python
        data_dict = json.loads(json_string)

        return data_dict

    except Exception as e:
        # Manejar errores (ej. Ollama desconectado, LLM devuelve texto inválido, etc.)
        logging

# --- FUNCIÓN ORQUESTADORA PRINCIPAL ---
def build_boletines_periodo(periodo_nombre: str, data_dir_raw: Path) -> pd.DataFrame | None:
    """
    Orquesta la extracción de boletines para un solo período.
    Lee 'detalle.csv', extrae IDs de boletín, los descarga y procesa.
    """
    try:
        # 1. Definir rutas
        nombre_carpeta = sanitize_filename(periodo_nombre)
        ruta_input_detalle = data_dir_raw / nombre_carpeta / "detalle.csv"
        
        if not ruta_input_detalle.exists():
            logging.warning(f"No se encontró {ruta_input_detalle}. Saltando extracción de boletines para {periodo_nombre}.")
            return None

        # 2. Cargar detalle.csv y extraer IDs de boletín
        logging.info(f"Cargando {ruta_input_detalle} para extraer IDs de boletín.")
        
        # (Usar 'low_memory=False' es una buena práctica para CSVs grandes)
        df_det = pd.read_csv(ruta_input_detalle, low_memory=False)
        
        # (Lógica de extracción de ID de su celda 4)
        boletines = (
            df_det.loc[df_det["Tipo._value_1"] == "Proyecto de Ley", "Descripcion"]
            .astype(str).str.extract(r"Bolet[ií]n\s*N[°º]?\s*(\d+)")[0]
            .dropna()
            .drop_duplicates()
        )
        
        if boletines.empty:
            logging.info(f"No se encontraron boletines en {periodo_nombre}.")
            return pd.DataFrame() # Devolver DF vacío

        logging.info(f"Encontrados {len(boletines)} boletines únicos para {periodo_nombre}. Iniciando descarga...")

        # 3. Descargar todos los boletines (con barra de progreso)
        lista_boletines_data = []
        for b_id in tqdm(boletines, desc=f"Boletines {periodo_nombre}"):
            try:
                # Llamar a la función auxiliar que ya definimos
                boletin_data_df = get_boletin(b_id) 
                if not boletin_data_df.empty:
                    lista_boletines_data.append(boletin_data_df)
            except Exception as e:
                logging.warning(f"Error al descargar boletín {b_id}: {e}")
        
        if not lista_boletines_data:
            logging.warning(f"No se pudo descargar ningún dato de boletín para {periodo_nombre}.")
            return pd.DataFrame()
            
        df_boletines = pd.concat(lista_boletines_data, ignore_index=True)

        # 4. Procesar materias (como en su celda 4)
        if 'materias' in df_boletines.columns:
            logging.info("Procesando materias y ámbitos...")
            df_boletines["materias_norm"] = df_boletines["materias"].apply(extraer_materia)
            df_boletines["ambitos"] = df_boletines["materias_norm"].apply(
                lambda x: x.get("ambitos_detectados") if isinstance(x, dict) else ["no cumple"]
            )
            df_boletines = df_boletines.drop(columns=["materias", "materias_texto", "materias_norm"], errors='ignore')
        
        return df_boletines

    except Exception as e:
        logging.error(f"Error fatal en build_boletines_periodo para {periodo_nombre}: {e}", exc_info=True)
        return None