from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util

def find_dependencia(nombre, mineduc):
    if not isinstance(nombre, str) or nombre.strip() == "":
        return None, 0, None
    result = process.extractOne(nombre, mineduc['colegio_merge_key'], scorer=fuzz.WRatio)
    if result:
        match, score, idx = result
        if score < 90:
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            emb_query = model.encode(nombre, convert_to_tensor=True)
            emb_candidates = model.encode(mineduc['colegio_merge_key'], convert_to_tensor=True)
            cosine_scores = util.cos_sim(emb_query, emb_candidates)
            best_idx = int(cosine_scores.argmax())
            best_match = mineduc['colegio_merge_key']
            best_score = cosine_scores[0][best_idx].item()
            print(f"Cosine similarity score: {best_score:.4f} for '{nombre}' vs '{best_match[best_idx]}'")
            return best_match[best_idx], best_score * 100, mineduc.iloc[best_idx]["COD_DEPE"]
        dep = mineduc.iloc[idx]["COD_DEPE"]
        return match, score, dep
    return None, 0, None