# neo4j_utils.py (Updated for Vyakyarth Dense-only implementation)
import os, dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from unidecode import unidecode

dotenv.load_dotenv()

class Neo4jRetriever:
    def __init__(self):
        # Use bolt for Neo4j Desktop or neo4j+s for AuraDB
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.auth = ("neo4j", os.getenv("NEO4J_PASSWORD"))
        self.driver = GraphDatabase.driver(self.uri, auth=self.auth)
        
        # Load Vyakyarth for semantic node resolution
        self.model = SentenceTransformer('krutrim-ai-labs/Vyakyarth', trust_remote_code=True)

    def close(self):
        self.driver.close()

    def get_shlokas_by_entities(self, entities):
        """
        Performs semantic matching on Concept nodes followed by 
        controlled traversal to Shlokas.
        """
        if not entities:
            return []

        all_shloka_ids = set()
        for entity in entities:
            # Normalize and embed the extracted entity
            norm_entity = unidecode(entity).lower().strip()
            entity_emb = self.model.encode(norm_entity).tolist()

            # Cypher query: Semantic Match + Controlled Traversal (Issue 3 Fix)
            query = """
            CALL db.index.vector.queryNodes('concept_embeddings', 5, $emb)
            YIELD node, score
            WHERE score > 0.8
            MATCH (node)-[r]-(m)
            RETURN DISTINCT r.shloka_id AS shloka_id
            """
            
            try:
                with self.driver.session() as session:
                    result = session.run(query, emb=entity_emb)
                    for record in result:
                        if record["shloka_id"]:
                            all_shloka_ids.add(record["shloka_id"])
            except Exception as e:
                print(f"Neo4j Semantic Search Error: {e}")
                
        return list(all_shloka_ids)

# Singleton instance for the pipeline
neo4j_retriever = Neo4jRetriever()