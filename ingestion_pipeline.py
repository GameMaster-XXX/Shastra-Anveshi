# # ingestion_pipeline.py
# import os, glob, re, json, torch, hashlib, pickle
# from sentence_transformers import SentenceTransformer
# from pymilvus import connections

# from data_ingestion import load_file
# from parser import parse_shlokas, create_parent_and_child_documents
# from milvus_utils import (
#     create_child_collection, create_parent_collection,
#     insert_child_chunks, insert_parent_chunks, create_indices, VYAKYARTH_DIM
# )

# # --- CONFIGURATION ---
# FOLDER_PATH = "Chapters" 
# CHILD_COL = "gita_children_vyakrath_161" 
# PARENT_COL = "gita_parent_vyakrath_161"
# MODEL_ID = "krutrim-ai-labs/Vyakyarth"
# ONTOLOGY_PATH = "gita_refined_ontology_final.json"
# CACHE_PATH = os.path.join("cache_pipeline_2", "file_metadata_v113_aug.pkl")

# # Load Ontology
# with open(ONTOLOGY_PATH, 'r', encoding='utf-8') as f:
#     GITA_ONTOLOGY = json.load(f)

# def get_ontology_data(chapter, shloka):
#     return GITA_ONTOLOGY.get(str(chapter), {}).get(str(shloka), {})

# def process_chapter_file(file_path, model):
#     filename = os.path.basename(file_path)
#     chapter_no = int(re.search(r'(\d+)', filename).group(1)) if re.search(r'(\d+)', filename) else 0
    
#     lines = load_file(file_path)
#     units = parse_shlokas(lines, chapter_no=chapter_no, source_file=file_path)
    
#     # 1. Split logic: 650 chars size, 75 overlap as per your parser.py signature
#     new_parents, new_children = create_parent_and_child_documents(
#         units, 
#         child_chunk_size=650, 
#         child_chunk_overlap=75
#     )
    
#     if not new_children: return

#     connections.connect(alias="default", host="localhost", port="19530")
#     try:
#         child_col = create_child_collection(CHILD_COL)
#         parent_col = create_parent_collection(PARENT_COL)
#         create_indices(child_col)
#         create_indices(parent_col)
#         child_col.load(); parent_col.load()

#         augmented_texts = []
#         for child in new_children:
#             ont = get_ontology_data(child['chapter'], child['shloka_no'])
            
#             # Map ontology metadata to child fields
#             child['ontology_summary'] = ont.get('summary', '')
#             child['doctrinal_keywords'] = ", ".join(ont.get('keywords', []))
#             child['cross_references'] = json.dumps(ont.get('cross_references', []))
#             child['doctrinal_intent'] = ont.get('intent', 'General')

#             # --- RECOMMENDED CHUNK FORMAT ---
#             # [TEXT]: ... > [METADATA_SUMMARY]: ... > [DOCTRINAL_KEYWORDS]: ...
#             aug_text = (
#                 f"[TEXT]: {child['text']} > "
#                 f"[METADATA_SUMMARY]: {child['ontology_summary']} > "
#                 f"[DOCTRINAL_KEYWORDS]: {child['doctrinal_keywords']}"
#             )
#             augmented_texts.append(aug_text)

#         print(f"Embedding augmented chunks for {filename}...")
#         dense_embs = model.encode(augmented_texts)
        
#         insert_child_chunks(child_col, new_children, dense_embs)
#         insert_parent_chunks(parent_col, new_parents)
#         print(f"Ingested {filename}")
        
#     finally:
#         connections.disconnect(alias="default")

# if __name__ == "__main__":
#     model = SentenceTransformer(MODEL_ID, trust_remote_code=True)
#     all_files = sorted(glob.glob(os.path.join(FOLDER_PATH, "*.*")))
#     for f in all_files:
#         process_chapter_file(f, model)

import os, glob, re, json, torch, hashlib, pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import connections, utility,Collection

from data_ingestion import load_file
from parser import parse_shlokas, create_parent_and_child_documents
from milvus_utils import (
    create_child_collection, create_parent_collection,
    insert_child_chunks, insert_parent_chunks, create_indices, VYAKYARTH_DIM
)

# --- CONFIGURATION ---
FOLDER_PATH = "Chapters" 
CHILD_COL = "gita_english_children_vyakrath_161" 
PARENT_COL = "gita_english_parent_vyakrath_161"
MODEL_ID = "krutrim-ai-labs/Vyakyarth"
ONTOLOGY_PATH = "gita_refined_ontology_final.json"

# Load Ontology
with open(ONTOLOGY_PATH, 'r', encoding='utf-8') as f:
    GITA_ONTOLOGY = json.load(f)

def get_ontology_data(chapter, shloka):
    return GITA_ONTOLOGY.get(str(chapter), {}).get(str(shloka), {})

# Use a global counter to ensure global_index uniqueness across files
current_global_idx = 1000  

def process_chapter_file(file_path, model):
    global current_global_idx
    filename = os.path.basename(file_path)
    
    # Extract chapter number robustly
    match = re.search(r'(\d+)', filename)
    chapter_no = int(match.group(1)) if match else 0
    
    print(f"\n>>> Processing {filename} (Adhyaya {chapter_no})")
    
    lines = load_file(file_path)
    units = parse_shlokas(lines, chapter_no=chapter_no, source_file=file_path)
    
    # 1. Generate Parent/Child Documents
    new_parents, new_children = create_parent_and_child_documents(
        units, 
        child_chunk_size=650, 
        child_chunk_overlap=75
    )
    
    if not new_children: 
        print(f"Skipping {filename}: No children generated.")
        return

    # Assign persistent global indices
    for child in new_children:
        child['global_index'] = current_global_idx
        current_global_idx += 1

    # 2. Augment text and Embed
    augmented_texts = []
    for child in new_children:
        ont = get_ontology_data(child['chapter'], child['shloka_no'])
        child['ontology_summary'] = ont.get('summary', '')
        child['doctrinal_keywords'] = ", ".join(ont.get('keywords', []))
        child['cross_references'] = json.dumps(ont.get('cross_references', []))
        child['doctrinal_intent'] = ont.get('intent', 'General')

        aug_text = (
            # f"[TEXT]: {child['text']} > "
            f"[METADATA_SUMMARY]: {child['ontology_summary']} > "
            f"[DOCTRINAL_KEYWORDS]: {child['doctrinal_keywords']}"
        )
        augmented_texts.append(aug_text)

    print(f"Embedding {len(augmented_texts)} chunks for {filename}...")
    
    # 3. Encode & Normalize (CRITICAL FIX)
    dense_embs = model.encode(augmented_texts, convert_to_numpy=True)
    norms = np.linalg.norm(dense_embs, axis=1, keepdims=True)
    dense_embs = dense_embs / np.where(norms > 0, norms, 1.0)
    dense_embs = dense_embs.astype(np.float32)

    # 4. Insert to Milvus
    child_col = create_child_collection(CHILD_COL)
    parent_col = create_parent_collection(PARENT_COL)
    
    insert_child_chunks(child_col, new_children, dense_embs)
    insert_parent_chunks(parent_col, new_parents)
    
    # Ensure indices are created after the first insertion
    create_indices(child_col)
    create_indices(parent_col)
    
    print(f"Successfully ingested {filename}")

if __name__ == "__main__":
    print(f"Loading Model: {MODEL_ID}")
    model = SentenceTransformer(MODEL_ID, trust_remote_code=True)
    
    # Ensure connections are open
    connections.connect(alias="default", host="localhost", port="19530")
    
    all_files = sorted(glob.glob(os.path.join(FOLDER_PATH, "*.*")))
    
    for f in all_files:
        try:
            process_chapter_file(f, model)
        except Exception as e:
            print(f"Error processing {f}: {e}")
    
    # Final load to memory for searching
    Collection(CHILD_COL).load()
    Collection(PARENT_COL).load()
    print("\nIngestion Pipeline Complete.")