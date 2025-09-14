import os
import sys
from typing import Dict, List
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError, ConfigurationError

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
        self.driver = None
        self.database = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize Neo4j connection with proper error handling."""
        try:
            uri = _get_secret("NEO4J_URI")
            user = _get_secret("NEO4J_USERNAME")
            password = _get_secret("NEO4J_PASSWORD")
            self.database = os.getenv("NEO4J_DATABASE")  # optional
            
            print(f"Attempting to connect to Neo4j at: {uri}")
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            
            # Test the connection
            self._test_connection()
            print("✅ Successfully connected to Neo4j!")
            
        except Exception as e:
            self._handle_connection_error(e)
    
    def _test_connection(self):
        """Test the Neo4j connection."""
        try:
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
        except Exception as e:
            raise ServiceUnavailable(f"Connection test failed: {e}")
    
    def _handle_connection_error(self, error):
        """Handle connection errors with helpful messages."""
        error_msg = "Failed to connect to Neo4j. "
        
        if isinstance(error, ServiceUnavailable):
            error_msg += "\n❌ Neo4j service is unavailable. Please check:\n"
            error_msg += "   1. Is Neo4j running? Start it with: neo4j start\n"
            error_msg += "   2. Is the URI correct in your .env file?\n"
            error_msg += "   3. Is Neo4j listening on the correct port (default: 7687)?\n"
            error_msg += "   4. Check firewall/network settings\n"
        elif isinstance(error, AuthError):
            error_msg += "\n❌ Authentication failed. Please check:\n"
            error_msg += "   1. Username and password in your .env file\n"
            error_msg += "   2. Default Neo4j credentials: neo4j/neo4j (change on first login)\n"
        elif isinstance(error, ConfigurationError):
            error_msg += "\n❌ Configuration error. Please check:\n"
            error_msg += "   1. NEO4J_URI format (e.g., bolt://localhost:7687)\n"
            error_msg += "   2. All required environment variables are set\n"
        else:
            error_msg += f"\n❌ Unexpected error: {error}\n"
        
        error_msg += f"\n📝 Please ensure you have:\n"
        error_msg += f"   - A running Neo4j instance\n"
        error_msg += f"   - Correct connection details in your .env file\n"
        error_msg += f"   - Use the template: cp neo4j_config_template.env .env\n"
        
        print(error_msg, file=sys.stderr)
        raise RuntimeError(error_msg) from error

    def close(self):
        if self.driver:
            self.driver.close()

    def _ensure_driver(self):
        """Ensure the driver is available."""
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized. Connection failed during setup.")
    
    def ensure_schema(self):
        self._ensure_driver()
        cypher = """
        CREATE CONSTRAINT concept_name_unique IF NOT EXISTS
        FOR (c:Concept) REQUIRE c.name IS UNIQUE
        """
        with self.driver.session(database=self.database) as s:
            s.run(cypher)

    def upsert_document(self, path: str):
        self._ensure_driver()
        cypher = """
        MERGE (d:Document {path: $path})
        ON CREATE SET d.createdAt = timestamp()
        RETURN id(d) AS id
        """
        with self.driver.session(database=self.database) as s:
            s.run(cypher, {"path": path})

    def upsert_concepts(self, names: List[str]):
        if not names:
            return
        self._ensure_driver()
        cypher = """
        UNWIND $names AS name
        MERGE (c:Concept {name: name})
        ON CREATE SET c.createdAt = timestamp()
        """
        with self.driver.session(database=self.database) as s:
            s.run(cypher, {"names": names})

    def link_concepts_to_document(self, path: str, keyword_map: Dict[str, List[str]]):
        if not keyword_map:
            return 
        self._ensure_driver()
        payload = [{"name": k, "variants": sorted(set(v))}
                   for k, v in keyword_map.items()]
        # Handle deduplication without APOC - use Python to merge variants
        # First, get existing variants to merge with new ones
        existing_cypher = """
        MERGE (d:Document {path: $path})
        WITH d, $rows AS rows
        UNWIND rows AS row
        MATCH (c:Concept {name: row.name})
        OPTIONAL MATCH (c)-[r:MENTIONS]->(d)
        RETURN c.name AS concept_name, coalesce(r.variants, []) AS existing_variants
        """
        
        with self.driver.session(database=self.database) as temp_session:
            existing_result = temp_session.run(existing_cypher, {"path": path, "rows": payload})
            existing_data = {record["concept_name"]: record["existing_variants"] for record in existing_result}
        
        # Merge variants in Python
        for item in payload:
            concept_name = item["name"]
            existing_variants = existing_data.get(concept_name, [])
            # Combine and deduplicate variants
            all_variants = list(set(existing_variants + item["variants"]))
            item["variants"] = sorted(all_variants)
        
        cypher = """
        MERGE (d:Document {path: $path})
        WITH d, $rows AS rows
        UNWIND rows AS row
        MATCH (c:Concept {name: row.name})
        MERGE (c)-[r:MENTIONS]->(d)
        SET r.variants = row.variants, r.count = size(row.variants)
        """
        with self.driver.session(database=self.database) as s:
            s.run(cypher, {"path": path, "rows": payload})

    def upsert_edges(self, edges: List[Dict]):
        if not edges:
            return
        self._ensure_driver()
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

    def save_graph(self, *, doc_path: str, keyword_map: Dict, adjacency: Dict):
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
