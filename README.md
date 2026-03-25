# Shashtra Anveshi: Advanced GraphRAG for Sanskrit Scriptures

**Shashtra Anveshi** is a sophisticated Retrieval-Augmented Generation (RAG) pipeline designed specifically for Sanskrit scriptures, currently implemented as a Proof of Concept (POC) for the Bhagavad Gita.

It leverages a dual-database architecture, combining dense vector retrieval (Milvus) with semantic Knowledge Graph traversal (Neo4j). By utilizing specialized language models like `krutrim-ai-labs/Vyakyarth` for embeddings and `Sarvam-M` for generation and listwise reranking, the system answers profound philosophical queries through the persona of a revered Advaita Vedanta Acharya.

## 🏗️ Core Architecture

The system operates on two parallel tracks during retrieval, merging their contexts before final generation:

1. **Semantic Vector Retrieval**: Chunks of Shlokas and their Bhashyas (commentaries) are embedded using the `Vyakyarth` model and stored in **Milvus**. Parent-child document chunking ensures precise semantic matching while maintaining broad contextual windows.
2. **Knowledge Graph Traversal**: Entities and triplets extracted from the scriptures are embedded and stored in **Neo4j**. Queries undergo entity extraction, resolving against the Neo4j vector index to pull allied verses based on cosmological, ontological, or soteriological relationships.

## 🛠️ Prerequisites & Infrastructure

Given the need for reliable containerized operations and vector storage, the project utilizes Docker for infrastructure.

* **Docker & Docker Compose** (For Milvus, MinIO, and etcd)
* **Neo4j** (Desktop or AuraDB instance)
* **Python 3.9+**
* **API Keys**:
* `SARVAM_M_API` (For ontology generation, KG extraction, and inference)
* `NEO4J_URI` & `NEO4J_PASSWORD`



### 1. Start the Infrastructure

Navigate to the project root and spin up the Milvus vector database using the provided SRE-ready configuration:

```bash
docker-compose up -d

```

*This starts `milvus-standalone`, `milvus-etcd`, and `milvus-minio` on your local machine.*

### 2. Environment Setup

Create a virtual environment and install dependencies:

```bash
conda create -n fine_tune python=3.10 -y
conda activate fine_tune
pip install -r requirements.txt

```

Create a `.env` file in the root directory:

```env
SARVAM_M_API=your_sarvam_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=your_neo4j_password
MILVUS_HOST=localhost
MILVUS_PORT=19530

```

---

## 🚀 Data Pipeline (Ingestion)

The ingestion process is divided into textual processing, ontology/graph generation, and database population.

### Step 1: Text Parsing & Ontology Generation

Place your raw chapter files (`.docx`, `.pdf`, or images) into the `Chapters/` directory.

```bash
python ontology_generator.py

```

* **What it does:** Uses `data_ingestion.py` (OCR/PyMuPDF) and `parser.py` (Langchain RecursiveCharacterTextSplitter) to separate Shlokas from Bhashya. It then calls Sarvam-M to generate `gita_refined_ontology.json`, detailing summaries, intents, and keywords for each verse.

### Step 2: Knowledge Graph Construction

Extract structural triplets (Subject, Predicate, Object) from the parsed text.

```bash
python knowledge_graph.py

```

* **What it does:** Generates `gita_knowledge_graph_triplets.csv`.
* Next, sync this data into Neo4j:

```bash
python neo4j_extract.py

```

### Step 3: Vector Database Ingestion

Populate the Milvus database with parent-child chunks and their embeddings.

```bash
python ingestion_pipeline.py

```

* **What it does:** Uses `embedding.py` to cache and generate 768-dimensional dense vectors using `Vyakyarth`. It augments chunks with the previously generated ontology metadata and inserts them into Milvus using `milvus_utils.py`.

---

## 🧠 Inference Pipeline (Querying)

Once the databases are populated, you can query the system. The inference pipeline features a multi-stage retrieval and routing mechanism.

```bash
python pipeline.py

```

### How `pipeline.py` works internally:

1. **Query Processing (`query_processor.py`)**: Detects the user's language, classifies the intent (e.g., Ontological vs. Soteriological), and translates the query to a Sanskrit pivot if necessary.
2. **Dual Retrieval**:
* Pulls precise contextual chunks from Milvus (`milvus_utils.py`).
* Extracts entities from the query and performs a semantic vector search in Neo4j to find linked Shloka IDs (`neo4j_utils.py`).


3. **Reranking & Expansion**: Sarvam-M performs listwise reranking of the retrieved child chunks. The system then fetches the full "Parent" documents to provide maximum context.
4. **Generation (`generator.py`)**: The Acharya persona evaluates the scholarly material. If the exact verse or concept doesn't exist, a localized "Not Found" response is generated. Otherwise, an authoritative, Advaita Vedanta-grounded answer is streamed to the user.

---

## 📁 Repository Structure Overview

| File | Sub-System | Description |
| --- | --- | --- |
| `docker-compose.yml` | **Infra** | Milvus Vector DB, Minio, and Etcd services. |
| `data_ingestion.py` | **Ingestion** | Multi-modal text loaders (Tesseract OCR, PyMuPDF, docx). |
| `parser.py` | **Ingestion** | Regex & Langchain splitters to isolate Shloka and Bhashya logic. |
| `ontology_generator.py` | **Graph** | Calls LLM to build intents and summaries (`gita_refined_ontology.json`). |
| `knowledge_graph.py` | **Graph** | Extracts semantic triplets from chapters (`gita_knowledge_graph_triplets.csv`). |
| `neo4j_extract.py` | **Graph** | Embeds triplets and builds the Neo4j Vector Index. |
| `embedding.py` | **Vector** | Caching and embedding logic using HuggingFace sentence-transformers. |
| `ingestion_pipeline.py` | **Vector** | Merges ontology with text chunks and writes to Milvus collections. |
| `pipeline.py` | **Inference** | Main execution script for the end-to-end user query lifecycle. |
| `query_processor.py` | **Inference** | Intent routing, translation, and LLM-based Listwise Reranking. |
| `neo4j_utils.py` / `milvus_utils.py` | **Inference** | Retrieval operations for Graph and Vector databases respectively. |
| `generator.py` | **Inference** | The 'Acharya' system prompt and final Sarvam-M API call generation. |