from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util

MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def encode_candidates(mineduc_df):
    """Codifica la lista de colegios de MINEDUC una sola vez."""
    print("Codificando la base de datos de MINEDUC (esto se hace 1 vez)...")
    
    # Asegúrate de que el df tenga un índice reseteado para que .iloc funcione
    mineduc_df = mineduc_df.reset_index(drop=True)
    
    candidate_list = mineduc_df['colegio_merge_key'].tolist()
    emb_candidates = MODEL.encode(candidate_list, convert_to_tensor=True)
    
    print("Codificación de candidatos completada.")
    return emb_candidates, mineduc_df

def find_dependencia_fast(nombre_query, mineduc_df, emb_candidates, threshold=0.65):
    """
    Encuentra el mejor match usando embeddings pre-calculados.
    """
    if not isinstance(nombre_query, str) or nombre_query.strip() == "":
        return None, 0, None

    # 1. Codificar solo el nombre que buscamos (el query)
    emb_query = MODEL.encode(nombre_query, convert_to_tensor=True)
    
    # 2. Calcular similitud (emb_candidates ya está en la GPU)
    cosine_scores = util.cos_sim(emb_query, emb_candidates)
    best_idx = int(cosine_scores.argmax())
    
    # 3. Obtener los resultados del índice posicional
    best_score = float(cosine_scores[0, best_idx])
    
    # 4. Filtro de confianza
    if best_score < threshold:
        return None, best_score * 100, None
        
    match_name = mineduc_df.iloc[best_idx]['colegio_merge_key']
    dependencia = mineduc_df.iloc[best_idx]['COD_DEPE']

    return match_name, best_score * 100, dependencia