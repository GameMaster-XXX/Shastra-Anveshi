import streamlit as st
import numpy as np
import os
import re
import traceback
import time
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from unidecode import unidecode 

# --- Custom Modules ---
from milvus_utils import dense_milvus_search, get_parents_by_ids, get_child_chunks_by_shloka_ids
from query_processor import (
    detect_query_language, 
    translate_query_to_sanskrit_pivot, 
    classify_query_intent, 
    extract_entities_for_graph, 
    llm_listwise_rerank
)
from neo4j_utils import neo4j_retriever 
from generator import construct_prompt_with_citations 

# --- Configuration ---
load_dotenv(override=True)
CHILD_COLLECTION_NAME = "gita_children_vyakrath_113"
PARENT_COLLECTION_NAME = "gita_parent_vyakrath_113"
FEEDBACK_COLLECTION_NAME = "Sashi_feedback"
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MODEL_ID = "krutrim-ai-labs/Vyakyarth"

# --- Pipeline Utilities ---
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

def rrf_fusion(results_list, k=60, shloka_boost=1.2):
    fused_scores = {}
    metadata_map = {}
    for results in results_list:
        for rank, hit in enumerate(results):
            chunk_id = f"{hit['parent_id']}_{hit['text'][:30]}"
            score = 1.0 / (k + rank + 1)
            if hit.get('chunk_type') == 'shloka': score *= shloka_boost
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + score
            metadata_map[chunk_id] = hit
    return [metadata_map[cid] for cid, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]

# --- Feedback Collection Functions ---
def create_feedback_collection(name, dim):
    if utility.has_collection(name): return Collection(name)
    fields = [
        FieldSchema(name="feedback_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="timestamp", dtype=DataType.INT64),
        FieldSchema(name="rating", dtype=DataType.INT64), # 0 = Auto-logged
        FieldSchema(name="feedback_text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="user_query", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="llm_response", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="retrieved_chunks_json", dtype=DataType.JSON),
        FieldSchema(name="query_embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description="Feedback for RAG system")
    col = Collection(name=name, schema=schema)
    col.create_index(field_name="query_embedding", index_params={"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}})
    return col

def insert_feedback(rating, feedback_text, user_query, llm_response, retrieved_chunks, query_embedding):
    try:
        if not connections.has_connection("default"):
            connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(FEEDBACK_COLLECTION_NAME)
        safe_chunks = [{"text": c.get("text"), "chapter": c.get("chapter"), "shloka_no": c.get("shloka_no")} for c in retrieved_chunks]
        entities = [[int(time.time())], [int(rating)], [str(feedback_text)], [str(user_query)], [str(llm_response)], [safe_chunks], [query_embedding.tolist()]]
        collection.insert(entities)
        collection.flush()
        return True
    except: return False

@st.cache_resource
def load_resources():
    try:
        embedder = SentenceTransformer(MODEL_ID, trust_remote_code=True)
        dim = embedder.get_sentence_embedding_dimension()
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        child_col = Collection(CHILD_COLLECTION_NAME)
        parent_col = Collection(PARENT_COLLECTION_NAME)
        child_col.load(); parent_col.load()
        f_col = create_feedback_collection(FEEDBACK_COLLECTION_NAME, dim)
        f_col.load()
        return embedder, child_col, parent_col, f_col
    except Exception as e:
        st.error(f"Failed to load resources: {e}")
        return None, None, None, None

# --- RAG Pipeline Logic ---
def get_rag_chunks(user_query, embedder, child_collection, parent_collection):
    if not all([embedder, child_collection, parent_collection]):
        return [], user_query, None
    try:
        lang = detect_query_language(user_query)
        ch_num, sh_num = extract_coordinate(user_query)
        q_emb = embedder.encode([user_query])[0]
        
        # 1. Coordinate Exact Match
        if ch_num and sh_num:
            coord_hits = child_collection.query(f"chapter == {ch_num} and shloka_no == '{sh_num}'", ["parent_id"])
            if coord_hits:
                parents = get_parents_by_ids(parent_collection, [h['parent_id'] for h in coord_hits])
                _, used_chunks = construct_prompt_with_citations(parents, user_query, "", lang)
                return used_chunks, user_query, q_emb

        # 2. Semantic Search Strategy
        intent = classify_query_intent(user_query)
        sanskrit_pivot = translate_query_to_sanskrit_pivot(user_query, lang)
        
        h_a = dense_milvus_search(child_collection, q_emb, top_k=30)
        h_b = dense_milvus_search(child_collection, embedder.encode([sanskrit_pivot])[0], top_k=30)
        
        v_hits = rrf_fusion([h_a, h_b])
        
        kg_ids = neo4j_retriever.get_shlokas_by_entities(extract_entities_for_graph(sanskrit_pivot))
        kg_chunks = get_child_chunks_by_shloka_ids(child_collection, kg_ids) if kg_ids else []
        
        all_c = v_hits[:50] + kg_chunks
        top_children = llm_listwise_rerank(all_c, sanskrit_pivot, user_query, intent=intent)
        
        p_ids = list({c['parent_id'] for c in top_children[:7]})
        final_parents = get_parents_by_ids(parent_collection, p_ids)
        
        _, used_chunks = construct_prompt_with_citations(final_parents, user_query, "", lang)
        return used_chunks, user_query, q_emb
    except:
        traceback.print_exc()
        return [], user_query, None

# --- Streamlit UI ---
st.set_page_config(page_title="Shastra Anveshi", page_icon="🔍")
st.title("🕉️ Welcome to Shastra Anveshi")
st.caption("🔍 Efficient Retrieval of Bhagavad Gita Shlokas")

embedder, child_collection, parent_collection, feedback_collection = load_resources()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ready. Enter a keyword or Chapter/Shloka (e.g. '2.47')."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if prompt := st.chat_input("Type Here...."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    clean_prompt = unidecode(prompt).lower().strip()
    greeting_pattern = r"^\s*(hlo|hi|hello|hey|namaska[ra]+m?|namaste|greetings|good\s*morning|bye|thank\s*you|thanks)\s*[!.]*\s*$"
    
    if re.match(greeting_pattern, clean_prompt, re.IGNORECASE):
        resp = "🙏 **Namaste!** Please ask specific questions or provide a Shloka number (e.g., 'Chapter 3 Shloka 10')."
        with st.chat_message("assistant"): st.markdown(resp)
        st.session_state.messages.append({"role": "assistant", "content": resp})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Retrieving Relevant Shlokas..."):
                chunks, original_query, query_emb = get_rag_chunks(prompt, embedder, child_collection, parent_collection)
                
                full_response_markdown = ""
                if chunks:
                    full_response_markdown += f"### ✅ Found {len(chunks)} relevant shlokas\n\n"
                    for chunk in chunks:
                        full_response_markdown += f"**📖 Chapter {chunk.get('chapter')}, Shloka {chunk.get('shloka_no')}**\n"
                        full_response_markdown += f"```text\n{chunk.get('text')}\n```\n---\n"
                    st.markdown(full_response_markdown)
                else:
                    full_response_markdown = "❌ No relevant documents found."
                    st.markdown(full_response_markdown)

        # --- AUTO-LOGGING TO MILVUS (Rating 0) ---
        if query_emb is not None:
            insert_feedback(
                rating=0, 
                feedback_text="Auto-logged", 
                user_query=original_query, 
                llm_response="[RETRIEVAL_ONLY]", 
                retrieved_chunks=chunks, 
                query_embedding=query_emb
            )

        st.session_state.messages.append({"role": "assistant", "content": full_response_markdown})
        
        if original_query and query_emb is not None:
            st.session_state.last_response_data = {"query": original_query, "response": "[RETRIEVAL_ONLY]", "chunks": chunks, "embedding": query_emb}
            st.session_state.feedback_pending = True
        st.rerun()

# --- Feedback Form ---
if st.session_state.get("feedback_pending", False):
    with st.form("feedback_form"):
        st.markdown("### Rate Retrieval Quality")
        rating = st.radio("Rating", [1, 2, 3, 4, 5], index=4, horizontal=True)
        feedback_text = st.text_area("Feedback on results:")
        if st.form_submit_button("Submit"):
            data = st.session_state.last_response_data
            # Log the user-provided rating
            insert_feedback(rating, feedback_text, data["query"], data["response"], data["chunks"], data["embedding"])
            st.toast("Feedback Saved!", icon="✅")
            st.session_state.feedback_pending = False
            st.rerun()