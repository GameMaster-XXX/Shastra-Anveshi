import os
import pickle
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

CACHE_FOLDER = "cache_pipeline_2"
EMB_FILE = os.path.join(CACHE_FOLDER, "gita_embeddings.npy")
CHUNK_FILE = os.path.join(CACHE_FOLDER, "gita_chunks.pkl")
META_FILE = os.path.join(CACHE_FOLDER, "file_metadata.pkl")

os.makedirs(CACHE_FOLDER, exist_ok=True)

def file_hash(path):
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def encode_texts(texts, model_name="muril_emb_sanskrit_finetuned_3", batch_size=64):
    """Encode a list of texts into normalized embeddings."""
    embedder = SentenceTransformer(model_name, cache_folder="models/muril_cache")
    embs = embedder.encode(texts, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=True)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    return embs.astype(np.float32), embedder

def save_embeddings(embeddings, chunked_output, file_metadata):
    """Save embeddings, chunks, and file metadata."""
    np.save(EMB_FILE, embeddings)
    with open(CHUNK_FILE, "wb") as f:
        pickle.dump(chunked_output, f)
    with open(META_FILE, "wb") as f:
        pickle.dump(file_metadata, f)
    print(f"Saved embeddings, chunks, and metadata to {CACHE_FOLDER}")

def load_embeddings():
    """Load cached embeddings, chunks, and file metadata if they exist."""
    if os.path.exists(EMB_FILE) and os.path.exists(CHUNK_FILE) and os.path.exists(META_FILE):
        embeddings = np.load(EMB_FILE)
        with open(CHUNK_FILE, "rb") as f:
            chunks = pickle.load(f)
        with open(META_FILE, "rb") as f:
            file_metadata = pickle.load(f)
        return embeddings, chunks, file_metadata
    return None, None, {}

def compute_chunk_hash(chunk):
    """Compute SHA256 for dedup."""
    content = f"{chunk.get('text', '')}|{chunk.get('chapter', '')}|{chunk.get('shloka_no', '')}|{chunk.get('source_file', '')}"
    return hashlib.sha256(content.encode('utf-8')).hexdigest()
