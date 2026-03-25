# # milvus_utils.py
# from pymilvus import connections, FieldSchema, CollectionSchema, Collection, DataType, utility
# import os

# MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
# MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
# VYAKYARTH_DIM = 768

# connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

# def create_child_collection(name, dim=VYAKYARTH_DIM):
#     if utility.has_collection(name):
#         return Collection(name)
    
#     fields = [
#         FieldSchema(name="global_index", dtype=DataType.INT64, is_primary=True, auto_id=False),
#         FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=64),
#         FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
#         FieldSchema(name="chapter", dtype=DataType.INT64),
#         FieldSchema(name="shloka_no", dtype=DataType.VARCHAR, max_length=512),
#         FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),
#         FieldSchema(name="has_bhashya", dtype=DataType.BOOL),
#         FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=512),
#         FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
#         FieldSchema(name="ontology_summary", dtype=DataType.VARCHAR, max_length=2000),
#         FieldSchema(name="doctrinal_keywords", dtype=DataType.VARCHAR, max_length=1000),
#         FieldSchema(name="cross_references", dtype=DataType.VARCHAR, max_length=1000),
#         FieldSchema(name="doctrinal_intent", dtype=DataType.VARCHAR, max_length=100),
#     ]
#     schema = CollectionSchema(fields, description="Gita Vyakyarth Augmented Dense Only")
#     return Collection(name=name, schema=schema)

# def create_parent_collection(name, dim=VYAKYARTH_DIM):
#     if utility.has_collection(name):
#         return Collection(name)
    
#     fields = [
#         FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
#         FieldSchema(name="dummy_embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
#         FieldSchema(name="chapter", dtype=DataType.INT64),
#         FieldSchema(name="shloka_no", dtype=DataType.VARCHAR, max_length=512),
#         FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=512),
#         FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535) 
#     ]
#     schema = CollectionSchema(fields, description="Gita Parent Docs")
#     return Collection(name=name, schema=schema)

# def create_indices(collection):
#     if "embedding" in [f.name for f in collection.schema.fields]:
#         if not collection.has_index(index_name="dense_idx"):
#             collection.create_index(
#                 field_name="embedding", 
#                 index_params={"metric_type": "IP", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}}, 
#                 index_name="dense_idx"
#             )
#     elif "dummy_embedding" in [f.name for f in collection.schema.fields]:
#         if not collection.has_index(index_name="parent_dummy_idx"):
#             collection.create_index(
#                 field_name="dummy_embedding", 
#                 index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1}}, 
#                 index_name="parent_dummy_idx"
#             )

# def insert_child_chunks(collection, child_chunks, dense_embs, batch_size=256):
#     n = len(child_chunks)
#     for i in range(0, n, batch_size):
#         j = min(i+batch_size, n)
#         batch = child_chunks[i:j]
#         entities = [
#             [int(x["global_index"]) for x in batch],
#             [str(x["parent_id"]) for x in batch],
#             dense_embs[i:j].tolist(),
#             [int(x.get("chapter", 0)) for x in batch],
#             [str(x.get('shloka_no', '')) for x in batch],
#             [str(x.get('chunk_type', 'bhashya')) for x in batch],
#             [bool(x.get("has_bhashya", False)) for x in batch],
#             [str(x.get("source_file", "")) for x in batch],
#             [str(x.get("text", "")) for x in batch],
#             [str(x.get("ontology_summary", "")) for x in batch],
#             [str(x.get("doctrinal_keywords", "")) for x in batch],
#             [str(x.get("cross_references", "")) for x in batch],
#             [str(x.get("doctrinal_intent", "")) for x in batch]
#         ]
#         collection.insert(entities)

# def insert_parent_chunks(collection, parent_chunks, dim=VYAKYARTH_DIM, batch_size=512):
#     n = len(parent_chunks)
#     dummy_vec = [[0.0] * dim for _ in range(n)]
#     for i in range(0, n, batch_size):
#         j = min(i+batch_size, n)
#         batch = parent_chunks[i:j]
#         entities = [
#             [str(x["parent_id"]) for x in batch],
#             dummy_vec[i:j],
#             [int(x.get("chapter", 0)) for x in batch],
#             [str(x.get('shloka_no', '')) for x in batch],
#             [str(x.get("source_file", "")) for x in batch],
#             [str(x.get("text", "")) for x in batch]
#         ]
#         collection.insert(entities)

# def dense_milvus_search(collection, dense_vec, top_k=50, filter_expr=None):
#     search_params = {"metric_type": "IP", "params": {"ef": 128}}
#     try:
#         results = collection.search(
#             data=[dense_vec],
#             anns_field="embedding",
#             param=search_params,
#             limit=top_k,
#             expr=filter_expr,
#             output_fields=["parent_id", "text", "chapter", "shloka_no", "chunk_type", "cross_references", "doctrinal_intent", "ontology_summary"]
#         )
#         hits = results[0]
#         return [{"parent_id": h.entity.get("parent_id"), "score": h.distance, "text": h.entity.get("text"),
#                  "chapter": h.entity.get("chapter"), "shloka_no": h.entity.get("shloka_no"),
#                  "chunk_type": h.entity.get("chunk_type"), "cross_references": h.entity.get("cross_references"),
#                  "doctrinal_intent": h.entity.get("doctrinal_intent"), "ontology_summary": h.entity.get("ontology_summary")} for h in hits]
#     except Exception as e:
#         print(f"Search Error: {e}")
#         return []

# def get_parents_by_ids(collection, parent_ids):
#     """Retrieves full parent blocks by their unique parent_id."""
#     if not parent_ids: return []
#     quoted_ids = [f'"{pid}"' for pid in parent_ids]
#     expr = f"parent_id in [{','.join(quoted_ids)}]"
#     try:
#         return collection.query(expr=expr, output_fields=["parent_id", "text", "chapter", "shloka_no"])
#     except Exception as e:
#         print(f"Query Error in get_parents_by_ids: {e}")
#         return []

# def get_child_chunks_by_shloka_ids(collection, shloka_ids):
#     """Retrieves child chunks for specific Chapter_Shloka IDs (Graph Fallback)."""
#     if not shloka_ids: return []
#     conditions = [f'(chapter == {sid.split("_")[1]} and shloka_no == "{sid.split("_")[2]}" and chunk_type == "shloka")' 
#                   for sid in shloka_ids if len(sid.split("_")) >= 3]
#     if not conditions: return []
#     return collection.query(expr=" or ".join(conditions), output_fields=["parent_id", "text", "chapter", "shloka_no", "chunk_type", "cross_references", "doctrinal_intent"])
from pymilvus import connections, FieldSchema, CollectionSchema, Collection, DataType, utility
import os
import numpy as np

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
VYAKYARTH_DIM = 768

connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

def create_child_collection(name, dim=VYAKYARTH_DIM):
    if utility.has_collection(name):
        return Collection(name)
    
    fields = [
        FieldSchema(name="global_index", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="chapter", dtype=DataType.INT64),
        FieldSchema(name="shloka_no", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="has_bhashya", dtype=DataType.BOOL),
        FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="ontology_summary", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="doctrinal_keywords", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="cross_references", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="doctrinal_intent", dtype=DataType.VARCHAR, max_length=100),
    ]
    schema = CollectionSchema(fields, description="Gita Vyakyarth Augmented Dense Only")
    return Collection(name=name, schema=schema)

def create_parent_collection(name, dim=VYAKYARTH_DIM):
    if utility.has_collection(name):
        return Collection(name)
    
    fields = [
        FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="dummy_embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="chapter", dtype=DataType.INT64),
        FieldSchema(name="shloka_no", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535) 
    ]
    schema = CollectionSchema(fields, description="Gita Parent Docs")
    return Collection(name=name, schema=schema)

def create_indices(collection):
    """Creates HNSW index for the dense vectors."""
    if "embedding" in [f.name for f in collection.schema.fields]:
        if not collection.has_index():
            index_params = {
                "metric_type": "IP", 
                "index_type": "HNSW", 
                "params": {"M": 16, "efConstruction": 200}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
    elif "dummy_embedding" in [f.name for f in collection.schema.fields]:
        if not collection.has_index():
            index_params = {
                "metric_type": "L2", 
                "index_type": "IVF_FLAT", 
                "params": {"nlist": 1}
            }
            collection.create_index(field_name="dummy_embedding", index_params=index_params)

def insert_child_chunks(collection, child_chunks, dense_embs, batch_size=100):
    """Inserts chunks into Milvus in small batches."""
    n = len(child_chunks)
    for i in range(0, n, batch_size):
        batch = child_chunks[i : i + batch_size]
        embs = dense_embs[i : i + batch_size]
        
        entities = [
            [int(x["global_index"]) for x in batch],
            [str(x["parent_id"]) for x in batch],
            embs.tolist(),
            [int(x.get("chapter", 0)) for x in batch],
            [str(x.get('shloka_no', '')) for x in batch],
            [str(x.get('chunk_type', 'bhashya')) for x in batch],
            [bool(x.get("has_bhashya", False)) for x in batch],
            [str(x.get("source_file", "")) for x in batch],
            [str(x.get("text", "")) for x in batch],
            [str(x.get("ontology_summary", "")) for x in batch],
            [str(x.get("doctrinal_keywords", "")) for x in batch],
            [str(x.get("cross_references", "")) for x in batch],
            [str(x.get("doctrinal_intent", "")) for x in batch]
        ]
        collection.insert(entities)
    collection.flush()

def insert_parent_chunks(collection, parent_chunks, dim=VYAKYARTH_DIM):
    """Inserts parent blocks."""
    n = len(parent_chunks)
    dummy_vec = np.zeros((n, dim), dtype=np.float32).tolist()
    
    entities = [
        [str(x["parent_id"]) for x in parent_chunks],
        dummy_vec,
        [int(x.get("chapter", 0)) for x in parent_chunks],
        [str(x.get('shloka_no', '')) for x in parent_chunks],
        [str(x.get("source_file", "")) for x in parent_chunks],
        [str(x.get("text", "")) for x in parent_chunks]
    ]
    collection.insert(entities)
    collection.flush()
    
def dense_milvus_search(collection, dense_vec, top_k=50, filter_expr=None):
    search_params = {"metric_type": "IP", "params": {"ef": 128}}
    try:
        results = collection.search(
            data=[dense_vec],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["parent_id", "text", "chapter", "shloka_no", "chunk_type", "cross_references", "doctrinal_intent", "ontology_summary"]
        )
        hits = results[0]
        return [{"parent_id": h.entity.get("parent_id"), "score": h.distance, "text": h.entity.get("text"),
                 "chapter": h.entity.get("chapter"), "shloka_no": h.entity.get("shloka_no"),
                 "chunk_type": h.entity.get("chunk_type"), "cross_references": h.entity.get("cross_references"),
                 "doctrinal_intent": h.entity.get("doctrinal_intent"), "ontology_summary": h.entity.get("ontology_summary")} for h in hits]
    except Exception as e:
        print(f"Search Error: {e}")
        return []

def get_parents_by_ids(collection, parent_ids):
    """Retrieves full parent blocks by their unique parent_id."""
    if not parent_ids: return []
    # Ensure IDs are unique and quoted for Milvus expression
    unique_ids = list(set(parent_ids))
    quoted_ids = [f'"{pid}"' for pid in unique_ids]
    expr = f"parent_id in [{','.join(quoted_ids)}]"
    try:
        return collection.query(expr=expr, output_fields=["parent_id", "text", "chapter", "shloka_no"])
    except Exception as e:
        print(f"Query Error in get_parents_by_ids: {e}")
        return []

def get_child_chunks_by_shloka_ids(collection, shloka_ids):
    """Retrieves child chunks for specific Chapter_Shloka IDs (Graph Fallback)."""
    if not shloka_ids: return []
    conditions = []
    for sid in shloka_ids:
        parts = sid.split("_")
        if len(parts) >= 3: # Format: BG_Chapter_Shloka
            conditions.append(f'(chapter == {parts[1]} and shloka_no == "{parts[2]}" and chunk_type == "shloka")')
    
    if not conditions: return []
    expr = " or ".join(conditions)
    try:
        return collection.query(expr=expr, output_fields=["parent_id", "text", "chapter", "shloka_no", "chunk_type", "cross_references", "doctrinal_intent"])
    except Exception as e:
        print(f"Query Error in get_child_chunks_by_shloka_ids: {e}")
        return []