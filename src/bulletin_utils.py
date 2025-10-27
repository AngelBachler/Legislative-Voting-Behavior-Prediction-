import pandas as pd
import logging
import requests
import json
from tqdm.notebook import tqdm
from pathlib import Path
import xmltodict

from .common_utils import sanitize_filename 


def get_boletin(boletin_id: str) -> pd.DataFrame:
    
    BASE_URL = "https://tramitacion.senado.cl/wspublico/tramitacion.php"
    params = {"boletin": boletin_id}
    
    try:
        # 1. Obtener el XML
        response = requests.get(BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        
        # --- INICIO DE LA CORRECCIÓN ---
        # 2. LIMPIAR el XML antes de parsear.
        # Reemplazamos los ampersands (&) inválidos que envía el servidor.
        xml_text = response.text.replace("&", "&amp;")
        # --- FIN DE LA CORRECCIÓN ---

        # 3. Parsear el XML (ahora limpio)
        data_dict = xmltodict.parse(
            xml_text,  # <-- Usamos la variable limpia
            force_list=('autor', 'materia')
        )
        
        # 4. Navegar a la raíz del proyecto
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
        
        # 5. Extraer Autores 
        autores_list = []
        if proyecto.get('autores') and proyecto['autores'].get('autor'):
            # Gracias a 'force_list', 'autor' siempre es una lista
            for autor in proyecto['autores']['autor']:
                autores_list.append(autor.get('PARLAMENTARIO'))
        out['autores_json'] = json.dumps(autores_list) # Guardar como string JSON

        # 6. Extraer Materias
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
    
def build_boletines_periodo(periodo_nombre: str, data_dir_raw: Path) -> pd.DataFrame | None:

    try:
        nombre_carpeta = sanitize_filename(periodo_nombre)
        ruta_input_detalle = data_dir_raw / nombre_carpeta / "detalle.csv"
        
        if not ruta_input_detalle.exists():
            logging.warning(f"No se encontró {ruta_input_detalle}.")
            return None

        logging.info(f"Cargando {ruta_input_detalle} para extraer IDs de boletín.")
        df_det = pd.read_csv(ruta_input_detalle, low_memory=False)
        
        # --- FIX AL REGEX ---
        boletines = (
            df_det.loc[df_det["Tipo._value_1"] == "Proyecto de Ley", "Descripcion"]
            .astype(str).str.extract(r"Bolet[ií]n\s*N[°º]?\s*(\d+)(?:-\d+)?")[0]
            .dropna()
            .drop_duplicates()
        )
        
        if boletines.empty:
            logging.info(f"No se encontraron boletines en {periodo_nombre}.")
            return pd.DataFrame() 

        logging.info(f"Encontrados {len(boletines)} boletines únicos. Iniciando descarga...")

        lista_boletines_data = []
        for b_id in tqdm(boletines, desc=f"Boletines {periodo_nombre}"):
            boletin_data_df = get_boletin(b_id) # Llama a la función de XML
            if not boletin_data_df.empty:
                lista_boletines_data.append(boletin_data_df)
        
        if not lista_boletines_data:
            logging.warning(f"No se pudo descargar ningún dato de boletín.")
            return pd.DataFrame()
            
        df_boletines = pd.concat(lista_boletines_data, ignore_index=True)

        return df_boletines

    except Exception as e:
        logging.error(f"Error fatal en build_boletines_periodo: {e}", exc_info=True)
        return None