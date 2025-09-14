import os
from neo4j import GraphDatabase

def _get_secret(name: str, *, required: bool = True) -> str:
    """
    Fetch a secret from environment (supports .env via load_dotenv()).
    Raises a clear error if missing and required.
    """
    val = os.getenv(name)
    if required and (val is None or val.strip() == ""):
        raise RuntimeError(
            f"Missing secret: {name}. "
            f"Set it in your secrets/env (e.g., .env, CI secrets)."
        )
    return val

class Neo4JSink:
    """
    Minimal Neo4j writer for concepts + co-occurrence edges.
    Values are pulled strictly from secrets/env:
      - NEO4J_URI
      - NEO4J_USERNAME
      - NEO4J_PASSWORD
      - NEO4J_DATABASE  (optional)
    """
    def __init__(self):
        uri = _get_secret("NEO4J_URI")
        user = _get_secret("NEO4J_USERNAME")
        password = _get_secret("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE")  # optional
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        if self.driver:
            self.driver.close()

    def ensure_schema(self):
        cypher = """
        CREATE CONSTRAINT concept_name_unique IF NOT EXISTS
        FOR (c:Concept) REQUIRE c.name IS UNIQUE
        """
        with self.driver.session(database=self.database) as s:
            s.run(cypher)

    def upsert_document(self, path: str):
        cypher = """
        MERGE (d:Document {path: $path})
        ON CREATE SET d.createdAt = timestamp()
        RETURN id(d) AS id
        """
        with self.driver.session(database=self.database) as s:
            s.run(cypher, {"path": path})

    def upsert_concepts(self, names: list[str]):
        if not names:
            return
        cypher = """
        UNWIND $names AS name
        MERGE (c:Concept {name: name})
        ON CREATE SET c.createdAt = timestamp()
        """
        with self.driver.session(database=self.database) as s:
            s.run(cypher, {"names": names})

    def link_concepts_to_document(self, path: str, keyword_map: dict[str, list[str]]):
        if not keyword_map:
            return 
        payload = [{"name": k, "variants": sorted(set(v))}
                   for k, v in keyword_map.items()]
        # If you don't have APOC, change ON MATCH to a simple overwrite or dedupe in Python first.
        cypher = """
        MERGE (d:Document {path: $path})
        WITH d, $rows AS rows
        UNWIND rows AS row
        MATCH (c:Concept {name: row.name})
        MERGE (c)-[r:MENTIONS]->(d)
        ON CREATE SET r.variants = row.variants, r.count = size(row.variants)
        ON MATCH  SET r.variants = apoc.coll.toSet(coalesce(r.variants, []) + row.variants),
                       r.count    = size(r.variants)
        """
        with self.driver.session(database=self.database) as s:
            s.run(cypher, {"path": path, "rows": payload})

    def upsert_edges(self, edges: list[dict]):
        if not edges:
            return
        cypher = """
        UNWIND $edges AS e
        MATCH (a:Concept {name: e.u})
        MATCH (b:Concept {name: e.v})
        MERGE (a)-[r:CO_OCCURS_WITH]->(b)
        ON CREATE SET r.weight = toInteger(e.w), r.createdAt = timestamp()
        ON MATCH  SET r.weight = coalesce(r.weight,0) + toInteger(e.w)
        """
        with self.driver.session(database=self.database) as s:
            s.run(cypher, {"edges": edges})

    def save_graph(self, *, doc_path: str, keyword_map: dict, adjacency: dict):
        concept_names = sorted(keyword_map.keys())

        dir_edges = []
        for u, nbrs in adjacency.items():
            for v, w in nbrs.items():
                if u == v or w is None:
                    continue
                dir_edges.append({"u": u, "v": v, "w": int(w)})

        self.ensure_schema()
        self.upsert_document(doc_path)
        self.upsert_concepts(concept_names)
        self.link_concepts_to_document(doc_path, keyword_map)

        BATCH = 5000
        for i in range(0, len(dir_edges), BATCH):
            self.upsert_edges(dir_edges[i:i+BATCH])
