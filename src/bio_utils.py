import logging
import json
from bs4 import BeautifulSoup
from rapidfuzz import process, fuzz  
from ollama import Client
from .common_utils import fetch_html, normalize_string
import re


def find_best_match_bcn(nombre_query, lista_choices, threshold=70):
    """
    Busca el 'nombre_query' en la 'lista_choices' usando una estrategia
    combinada de 'token_set_ratio' y 'partial_token_set_ratio'.

    Esta función está optimizada para usar el 'processor' de rapidfuzz,
    aplicando 'normalize_string' a ambas cadenas antes de comparar.

    Args:
        nombre_query (str): El nombre que se busca (ej. "Juan Pérez").
        lista_choices (list): La lista de nombres donde buscar (ej. ["Pérez, Juan A.", ...]).
        threshold (int): El puntaje mínimo (0-100) para considerar una coincidencia.

    Returns:
        tuple: (mejor_nombre_encontrado, puntaje_final) o (None, 0) si no supera el umbral.
    """
    if not nombre_query or not lista_choices:
        return None, 0
    
    mejores_candidatos = process.extract(
        query=nombre_query,
        choices=lista_choices,
        scorer=fuzz.token_set_ratio,
        processor=normalize_string,  
        limit=5
    )
    
    mejor_nombre = None
    mejor_score_final = 0

    for candidato_nombre, score_paso1, _ in mejores_candidatos:
        
        score_paso2 = fuzz.partial_token_set_ratio(
            normalize_string(nombre_query),
            normalize_string(candidato_nombre)
        )
        
        score_final = max(score_paso1, score_paso2)
        
        if score_final > mejor_score_final:
            mejor_score_final = score_final
            mejor_nombre = candidato_nombre

    if mejor_score_final >= threshold:
        return mejor_nombre, mejor_score_final
    else:
        return None, mejor_score_final
    
def get_section_paragraphs(soup: BeautifulSoup, title_patterns: list) -> list:

    if not isinstance(title_patterns, (list, tuple)):
        title_patterns = [title_patterns]

    paras = []
    try:
        for box in soup.select("div.box_contenidos"):
            h4 = box.find("h4")
            if not h4:
                continue
            
            title = normalize_string(h4.get_text(" ", strip=True))
            
            if any(re.search(p, title, flags=re.IGNORECASE) for p in title_patterns):
                
                all_paragraphs = box.find_all("p")
                
                if all_paragraphs:
                    for p in all_paragraphs:
                        t = p.get_text(" ", strip=True) 
                        if t:
                            paras.append(t)
                
                else:
                    content_div = h4.find_next_sibling("div")
                    if content_div:
                        t = content_div.get_text(" ", strip=True)
                        if t:
                            paras.extend([line.strip() for line in t.split('\n') if line.strip()])
                if paras:
                    break 
                        
    except Exception as e:
        logging.error(f"Error al parsear 'get_section_paragraphs': {e}")
    
    return paras

def _extract_district_from_trajectory(soup: BeautifulSoup, periodos_validos: list) -> int:

    try:
        for td in soup.find_all("td", class_="trayectoria_align"):
            texto_cargo = td.get_text(" ", strip=True)

            match_periodo = re.search(r"(\d{4})\s*[-–]\s*[A-Za-z]*\s*(\d{4})", texto_cargo)
            if not match_periodo:
                continue
            
            periodo_encontrado = f"{match_periodo.group(1)}-{match_periodo.group(2)}"

            if periodo_encontrado not in periodos_validos:
                continue

            distrito_num = None

            span_distrito = td.find("span", {"property": "bcnbio:representingPlaceNamed"})
            if span_distrito:
                m = re.search(r"(\d+)", span_distrito.get_text(" ", strip=True))
                if m:
                    distrito_num = int(m.group(1))

            if distrito_num is None:
                for div in td.find_all("div"):
                    texto_div = div.get_text(" ", strip=True)
                    if "Distrito" in texto_div:
                        m = re.search(r"(\d+)", texto_div)
                        if m:
                            distrito_num = int(m.group(1))
                            break 
            
            if distrito_num:
                return distrito_num 

    except Exception as e:
        logging.error(f"Error al parsear 'extract_district_from_trajectory': {e}")
    
    return None 

def scrape_bcn_bio_data(url: str, periodos_validos: list) -> dict:

    out = {
        "status": None,
        "distrito": None,
        "familia_juventud_parrafos": [],
        "estudios_vida_laboral_parrafos": [],
        "bio_texto_completo": ""
    }

    status, text, html = fetch_html(url) 
    out["status"] = status
    if status != 200 or not html:
        return out 

    soup = BeautifulSoup(text, "html.parser")

    out["distrito"] = _extract_district_from_trajectory(soup, periodos_validos)

    fam_parrafos = get_section_paragraphs(
        soup, 
        [r"Familia\s+y\s+Juventud", r"Familia", r"Juventud"]
    )

    est_parrafos = get_section_paragraphs(
        soup, 
        [r"Estudios\s+y\s+vida\s+laboral", r"Estudios", r"Vida\s+laboral"]
    )
    
    out["familia_juventud_parrafos"] = fam_parrafos
    out["estudios_vida_laboral_parrafos"] = est_parrafos

    out["bio_texto_completo"] = " ".join(fam_parrafos + est_parrafos)

    return out

def extract_bio_data_llm(client: Client, texto_biografia: str, model_name: str) -> dict | None:
    """
    Usa un cliente de Ollama para extraer datos biográficos estructurados (JSON)
    desde un texto de biografía, usando un prompt de sistema específico.

    Args:
        client (ollama.Client): El cliente de Ollama (generado por 'get_ollama_client').
        texto_biografia (str): El texto crudo de la biografía.
        model_name (str): El nombre del modelo a usar (ej. 'llama3:instruct').

    Returns:
        dict: Un diccionario con los 11 campos extraídos.
        None: Si el texto es inválido, la conexión falla o el JSON es incorrecto.
    """
    # --- 1. Prompt del Sistema (El "cerebro" de la extracción) ---
    SYSTEM_PROMPT = """
Eres un asistente experto en genealogía e historia política de Chile, actuando como un extractor de datos JSON.
Tu tarea es leer la siguiente biografía y extraer SOLAMENTE la siguiente
información en formato JSON.

**Reglas Estrictas:**
1.  Responde ÚNICAMENTE con el objeto JSON. No añadas texto introductorio.
2.  Si un dato no se encuentra en el texto, usa `null`.
3.  `numero_total_hijos`: Debe ser un número entero (int). Si dice "dos hijas y un hijo", el valor es `3`.
4.  `colegios` y `trabajo`: Deben ser una lista de strings `[]`. Si no hay, usa `[]`.
5.  `universidad` y `carrera`: Extraer solo si el texto implica que completó los estudios (ej. "se tituló", "juró como abogado").
6.  `maximo_nivel_educativo`: Infiere el nivel más alto. Valores válidos: ["Enseñanza Básica", "Enseñanza Media", "Educación Universitaria", "Magíster", "Doctor/a", null].

**Formato JSON Requerido:**
{
  "lugar_nacimiento": "Ciudad, Región o País",
  "fecha_nacimiento": "YYYY-MM-DD",
  "padre": "Nombre completo del padre",
  "madre": "Nombre completo de la madre",
  "estado_civil": "Soltero/a | Casado/a | Viudo/a | Divorciado/a | null",
  "numero_total_hijos": 0,
  "colegios": ["Nombre Colegio 1", "Nombre Colegio 2"],
  "universidad": "Nombre Universidad (solo si completó)",
  "carrera": "Nombre Carrera (solo si completó)",
  "maximo_nivel_educativo": "Educación Universitaria",
  "trabajo": ["Puesto/Profesión 1", "Puesto/Profesión 2"]
}
"""

    # --- 2. Validación de Entradas (Guard Clauses) ---
    if not client:
        logging.warning("extract_bio_data_llm: Cliente Ollama no es válido (None).")
        return None
    
    if not isinstance(texto_biografia, str) or len(texto_biografia.strip()) < 50:
        # No molestar al LLM si el texto es muy corto o inválido
        logging.info("extract_bio_data_llm: Texto de biografía omitido (demasiado corto o inválido).")
        return None

    # --- 3. Ejecución y Parseo (Bloque Try/Except) ---
    try:
        # Truncar el texto para asegurar que quepa en el contexto del modelo
        texto_truncado = texto_biografia[:4000] 
        
        # 3.1. Llamar al LLM
        response = client.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
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