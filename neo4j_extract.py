# neo4j_extract.py
import pandas as pd
from neo4j import GraphDatabase
import os, dotenv
from unidecode import unidecode
from sentence_transformers import SentenceTransformer

dotenv.load_dotenv()
# Vyakyarth model for node embeddings
model = SentenceTransformer('krutrim-ai-labs/Vyakyarth', trust_remote_code=True)

class Neo4jExporter:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def reset_and_prepare(self):
        with self.driver.session() as session:
            print("Resetting old indexes and properties...")
            session.run("DROP INDEX concept_embeddings IF EXISTS")
            session.run("MATCH (n:Concept) SET n.embedding = null, n.name_normalized = null")
            
            print("Creating new Vector Index (768 Dim)...")
            session.run("""
                CREATE VECTOR INDEX concept_embeddings IF NOT EXISTS
                FOR (n:Concept) ON (n.embedding)
                OPTIONS {indexConfig: {
                 `vector.dimensions`: 768,
                 `vector.similarity_function`: 'cosine'
                }}
            """)
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE")

    def upload_triplets(self, csv_path):
        df = pd.read_csv(csv_path).fillna("Unknown").astype(str)
        df.columns = df.columns.str.strip()
        
        records = []
        for _, row in df.iterrows():
            name = row['Subject']
            norm = unidecode(name).lower().strip()
            emb = model.encode(norm).tolist()
            records.append({
                "name": name, "norm": norm, "emb": emb,
                "obj": row['Object'], "pred": row['Predicate'], "sid": row['Chapter_Shloka_id']
            })

        query = """
        UNWIND $rows AS row
        MERGE (s:Concept {name: row.name})
        SET s.name_normalized = row.norm, s.embedding = row.emb
        MERGE (o:Concept {name: row.obj})
        SET o.name_normalized = row.norm, o.embedding = row.emb
        WITH s, o, row
        CALL apoc.create.relationship(s, row.pred, {shloka_id: row.sid}, o) YIELD rel
        RETURN count(*)
        """
        with self.driver.session() as session:
            session.run(query, rows=records)
        print("Neo4j Sync with Vyakyarth Complete.")

if __name__ == "__main__":
    URI = "bolt://localhost:7687"
    AUTH = ("neo4j", "gita_password")
    exporter = Neo4jExporter(URI, AUTH)
    exporter.reset_and_prepare()
    exporter.upload_triplets("gita_knowledge_graph_triplets.csv")