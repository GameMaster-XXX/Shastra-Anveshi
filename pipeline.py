import numpy as np
import os, re, json, traceback, dotenv
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
from unidecode import unidecode

# --- Custom Modules ---
from milvus_utils import dense_milvus_search, get_parents_by_ids, get_child_chunks_by_shloka_ids
from query_processor import (
    detect_query_language, translate_query_to_sanskrit_pivot, 
    classify_query_intent, extract_entities_for_graph, llm_listwise_rerank, generate_localized_not_found
)
from neo4j_utils import neo4j_retriever 
from retriver import print_retrieved_chunks
from generator import construct_prompt_with_citations, call_llm_api

# Load configuration
dotenv.load_dotenv()
CHILD_COLLECTION_NAME = "gita_children_vyakrath_161"
PARENT_COLLECTION_NAME = "gita_parent_vyakrath_161"
MODEL_ID = "krutrim-ai-labs/Vyakyarth"

def devnagari_to_ascii(text):
    indian_digits = '०१२३४५६७८९'
    ascii_digits = '0123456789'
    translation_table = str.maketrans(indian_digits, ascii_digits)
    return text.translate(translation_table)

def extract_coordinate(query):
    clean_query = devnagari_to_ascii(query)
    pattern_a = re.compile(r'(\d+)\s*[\.:\-/]\s*(\d+)')
    pattern_b = re.compile(r'(?:chapter|ch|अध्याय|सर्ग)\s*(\d+).*?(?:verse|shloka|श्लोक|मन्त्र)\s*(\d+)', re.IGNORECASE)
    
    match_b = pattern_b.search(clean_query)
    if match_b: return int(match_b.group(1)), str(match_b.group(2))
    match_a = pattern_a.search(clean_query)
    if match_a: return int(match_a.group(1)), str(match_a.group(2))
    return None, None

def is_raw_sanskrit(text):
    return any(p in text for p in ["।", "॥"])

# --- NEW: Normalization Helper ---
def get_query_embedding(embedder, text):
    """Encodes and L2 normalizes for IP metric compatibility."""
    emb = embedder.encode([text])[0]
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.astype(np.float32)

def rrf_fusion(results_list, k=60, shloka_boost=1.3):
    fused_scores = {}
    metadata_map = {}
    for results in results_list:
        for rank, hit in enumerate(results):
            chunk_id = f"{hit['parent_id']}_{hit['text'][:40]}"
            score = 1.0 / (k + rank + 1)
            if hit.get('chunk_type') == 'shloka':
                score *= shloka_boost
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + score
            metadata_map[chunk_id] = hit
    sorted_chunks = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [metadata_map[cid] for cid, _ in sorted_chunks]

def get_refined_parents(top_children, limit=7):
    parent_relevance = {}
    for child in top_children:
        pid = child['parent_id']
        # Fallback for llm_score if reranker failed
        score = child.get('llm_score', 1.0) 
        parent_relevance[pid] = parent_relevance.get(pid, 0) + score
    sorted_parents = sorted(parent_relevance.items(), key=lambda x: x[1], reverse=True)
    return [p[0] for p in sorted_parents[:limit]]

def perform_expansion_retrieval(collection, initial_hits):
    expanded_ids = set()
    for hit in initial_hits:
        try:
            refs = json.loads(hit.get('cross_references', '[]'))
            if isinstance(refs, list):
                expanded_ids.update(refs)
        except: continue
    if not expanded_ids: return []
    return get_parents_by_ids(collection, list(expanded_ids))

def run_cli_pipeline():
    print(f"Initializing Vyakyarth Model: {MODEL_ID}")
    embedder = SentenceTransformer(MODEL_ID, trust_remote_code=True)
    connections.connect(host=os.getenv("MILVUS_HOST", "localhost"), port="19530")
    
    child_collection = Collection(CHILD_COLLECTION_NAME)
    parent_collection = Collection(PARENT_COLLECTION_NAME)
    child_collection.load()
    parent_collection.load()
    
    print("Robust Acharya Pipeline (v67.5) - Normalized Retrieval & Fallback Active.\n")

    while True:
        try:
            user_query = input("Ask the Acharya (or 'q'): ").strip()
            if user_query.lower() == 'q': break
            if not user_query: continue

            ch_num, sh_num = extract_coordinate(user_query)
            lang = detect_query_language(user_query)
            
            # 1. Coordinate Handling
            if ch_num and sh_num:
                expr = f"chapter == {ch_num} and shloka_no == '{sh_num}'"
                coord_hits = child_collection.query(expr=expr, output_fields=["parent_id", "text", "chapter", "shloka_no"])
                if coord_hits:
                    pids = list({h['parent_id'] for h in coord_hits})
                    parent_docs = get_parents_by_ids(parent_collection, pids)
                    prompt, used = construct_prompt_with_citations(parent_docs, user_query, "", lang)
                    print(f"\nResponse:\n{call_llm_api(prompt)}\n")
                    continue
                else:
                    print(f"\nAcharya: {generate_localized_not_found(user_query, lang)}\n")
                    continue

            # 2. Semantic Prep
            intent = classify_query_intent(user_query)
            sanskrit_pivot = translate_query_to_sanskrit_pivot(user_query, lang)
            
            # SOFT FILTER: Only apply if confidence in intent is high
            filter_expr = None
            if any(w in user_query.lower() for w in ["how to", "process of", "steps for"]):
                filter_expr = 'doctrinal_intent == "Instruction"'

            # 3. Multi-Channel Vector Search (with Normalization)
            # We wrap the query in the same [TEXT] format used in ingestion
            aug_query = f"[TEXT]: {sanskrit_pivot} > [METADATA_SUMMARY]: > [DOCTRINAL_KEYWORDS]:"
            
            q_emb_raw = get_query_embedding(embedder, user_query)
            q_emb_pivot = get_query_embedding(embedder, aug_query)
            
            hits_a = dense_milvus_search(child_collection, q_emb_raw, top_k=50, filter_expr=filter_expr)
            hits_b = dense_milvus_search(child_collection, q_emb_pivot, top_k=50, filter_expr=filter_expr)
            
            entities = extract_entities_for_graph(sanskrit_pivot)
            keywords = " ".join(entities) if entities else sanskrit_pivot
            hits_c = dense_milvus_search(child_collection, get_query_embedding(embedder, keywords), top_k=30)

            vector_hits = rrf_fusion([hits_a, hits_b, hits_c])

            # 4. Graph Stream
            kg_chunks = []
            if not is_raw_sanskrit(user_query):
                kg_ids = neo4j_retriever.get_shlokas_by_entities(entities)
                kg_chunks = get_child_chunks_by_shloka_ids(child_collection, kg_ids)

            # 5. Reranking
            all_child_candidates = vector_hits[:60] + kg_chunks
            top_children = llm_listwise_rerank(all_child_candidates, sanskrit_pivot, user_query, intent=intent)

            # --- FALLBACK LOGIC ---
            if not top_children and vector_hits:
                print("--- Reranker found no matches. Falling back to top vector hits. ---")
                top_children = vector_hits[:5]

            final_parent_ids = get_refined_parents(top_children, limit=7)
            final_parent_docs = get_parents_by_ids(parent_collection, final_parent_ids)
            
            # Allied Verses
            allied_parents = perform_expansion_retrieval(parent_collection, top_children[:5])
            seen_pids = {p['parent_id'] for p in final_parent_docs}
            for allied in allied_parents:
                if allied['parent_id'] not in seen_pids:
                    final_parent_docs.append(allied)

            # 6. Final Generation
            if final_parent_docs:
                prompt, used_chunks = construct_prompt_with_citations(final_parent_docs, user_query, "", lang)
                print_retrieved_chunks(used_chunks)
                print(f"\nResponse:\n{call_llm_api(prompt)}\n")
            else:
                print("Acharya could not find the exact context even after fallback.")

        except Exception:
            traceback.print_exc()

if __name__ == "__main__":
    run_cli_pipeline()