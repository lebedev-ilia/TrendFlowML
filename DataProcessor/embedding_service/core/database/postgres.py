"""PostgreSQL database layer with pgvector support"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor
from psycopg2.extensions import register_adapter, AsIs
from psycopg2.pool import ThreadedConnectionPool

from ..errors import EmbeddingServiceError

# Register UUID adapter for psycopg2
# Convert UUID to PostgreSQL UUID type
def adapt_uuid(uuid_obj):
    return AsIs(f"'{uuid_obj}'::uuid")

register_adapter(uuid.UUID, adapt_uuid)


class PostgresEmbeddingStore:
    """PostgreSQL store for embedding metadata"""

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        min_conn: int = 1,
        max_conn: int = 10,
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password

        dsn = f"host={host} port={port} dbname={database} user={user} password={password}"
        self.pool = ThreadedConnectionPool(min_conn, max_conn, dsn=dsn)

        # Initialize schema
        self._init_schema()

    def _get_conn(self):
        """Get connection from pool"""
        return self.pool.getconn()

    def _put_conn(self, conn):
        """Return connection to pool"""
        self.pool.putconn(conn)

    def _init_schema(self):
        """Initialize database schema"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # Create embeddings table (PostgreSQL syntax does not support inline INDEX)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id UUID PRIMARY KEY,
                        category TEXT NOT NULL,
                        name TEXT,
                        embedding_model TEXT NOT NULL,
                        embedding_dim INTEGER NOT NULL,
                        embedding VECTOR,
                        metadata JSONB,
                        image_url TEXT,
                        added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )

                # Create B‑tree indexes for common filters
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_embeddings_category
                    ON embeddings (category);
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_embeddings_embedding_model
                    ON embeddings (embedding_model);
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_embeddings_category_model
                    ON embeddings (category, embedding_model);
                    """
                )

                # Note: ivfflat index requires specific dimension
                # We can't create it without knowing the vector dimension
                # Index will be created per model/dimension later if needed
                # For now, pgvector will use sequential scan which is fine for small datasets
                # To create index manually for specific dimension:
                # CREATE INDEX idx_embedding_vector_512 
                # ON embeddings USING ivfflat (embedding vector_cosine_ops)
                # WITH (lists = 100) 
                # WHERE embedding_dim = 512;

                conn.commit()
        except Exception as e:
            conn.rollback()
            raise EmbeddingServiceError(f"Failed to initialize schema: {e}", error_code="db_init_failed") from e
        finally:
            self._put_conn(conn)

    def add(
        self,
        *,
        object_id: Optional[uuid.UUID],
        category: str,
        name: Optional[str],
        embedding_model: str,
        embedding_dim: int,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        image_url: Optional[str] = None,
    ) -> uuid.UUID:
        """Add an embedding to the database"""
        if object_id is None:
            object_id = uuid.uuid4()

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # Convert embedding to pgvector format string
                embedding_str = "[" + ",".join(str(f) for f in embedding) + "]"

                cur.execute("""
                    INSERT INTO embeddings 
                    (id, category, name, embedding_model, embedding_dim, embedding, metadata, image_url)
                    VALUES (%s::uuid, %s, %s, %s, %s, %s::vector, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        category = EXCLUDED.category,
                        name = EXCLUDED.name,
                        embedding_model = EXCLUDED.embedding_model,
                        embedding_dim = EXCLUDED.embedding_dim,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        image_url = EXCLUDED.image_url,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id;
                """, (
                    str(object_id),
                    category,
                    name,
                    embedding_model,
                    embedding_dim,
                    embedding_str,
                    Json(metadata) if metadata else None,
                    image_url,
                ))
                result = cur.fetchone()
                conn.commit()
                return result[0] if result else object_id
        except Exception as e:
            conn.rollback()
            raise EmbeddingServiceError(f"Failed to add embedding: {e}", error_code="db_add_failed") from e
        finally:
            self._put_conn(conn)

    def get(self, object_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Get an embedding by ID"""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, category, name, embedding_model, embedding_dim,
                           embedding::text as embedding, metadata, image_url, added_at, updated_at
                    FROM embeddings
                    WHERE id = %s::uuid;
                """, (str(object_id),))
                row = cur.fetchone()
                if row:
                    result = dict(row)
                    # Parse embedding vector string
                    if result["embedding"]:
                        embedding_str = result["embedding"].strip("[]")
                        result["embedding"] = [float(x) for x in embedding_str.split(",")]
                    return result
                return None
        except Exception as e:
            raise EmbeddingServiceError(f"Failed to get embedding: {e}", error_code="db_get_failed") from e
        finally:
            self._put_conn(conn)

    def delete(self, object_id: uuid.UUID) -> bool:
        """Delete an embedding by ID"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM embeddings WHERE id = %s::uuid RETURNING id;", (str(object_id),))
                deleted = cur.fetchone() is not None
                conn.commit()
                return deleted
        except Exception as e:
            conn.rollback()
            raise EmbeddingServiceError(f"Failed to delete embedding: {e}", error_code="db_delete_failed") from e
        finally:
            self._put_conn(conn)

    def search(
        self,
        *,
        category: Optional[str],
        embedding_model: str,
        embedding: List[float],
        top_k: int = 10,
        similarity_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings"""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                embedding_str = "[" + ",".join(str(f) for f in embedding) + "]"

                # Build query
                where_clauses = ["embedding_model = %s"]
                params = [embedding_model]

                if category:
                    where_clauses.append("category = %s")
                    params.append(category)

                where_sql = " AND ".join(where_clauses)
                params.append(embedding_str)
                params.append(1.0 - similarity_threshold)  # cosine distance threshold
                params.append(top_k)

                query = f"""
                    SELECT id, category, name, embedding_model, embedding_dim,
                           embedding::text as embedding, metadata, image_url, added_at,
                           1 - (embedding <=> %s::vector) as similarity
                    FROM embeddings
                    WHERE {where_sql}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """

                cur.execute(query, params)
                rows = cur.fetchall()

                results = []
                for row in rows:
                    result = dict(row)
                    # Parse embedding vector string
                    if result["embedding"]:
                        embedding_str = result["embedding"].strip("[]")
                        result["embedding"] = [float(x) for x in embedding_str.split(",")]
                    results.append(result)

                return results
        except Exception as e:
            raise EmbeddingServiceError(f"Failed to search embeddings: {e}", error_code="db_search_failed") from e
        finally:
            self._put_conn(conn)

    def list_categories(self) -> List[str]:
        """List all categories"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT category FROM embeddings ORDER BY category;")
                return [row[0] for row in cur.fetchall()]
        except Exception as e:
            raise EmbeddingServiceError(f"Failed to list categories: {e}", error_code="db_list_failed") from e
        finally:
            self._put_conn(conn)

    def count_by_category(self, category: Optional[str] = None) -> Dict[str, int]:
        """Count embeddings by category"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                if category:
                    cur.execute("SELECT COUNT(*) FROM embeddings WHERE category = %s;", (category,))
                    return {category: cur.fetchone()[0]}
                else:
                    cur.execute("SELECT category, COUNT(*) FROM embeddings GROUP BY category;")
                    return {row[0]: row[1] for row in cur.fetchall()}
        except Exception as e:
            raise EmbeddingServiceError(f"Failed to count embeddings: {e}", error_code="db_count_failed") from e
        finally:
            self._put_conn(conn)

    def get_all_embeddings(
        self,
        category: str,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all embeddings for a category.
        
        Args:
            category: Category name
            embedding_model: Optional model name filter
            
        Returns:
            List of embeddings with keys: id, category, name, embedding_model, embedding_dim,
            embedding (as list), metadata, image_url, added_at
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                where_clauses = ["category = %s"]
                params = [category]
                
                if embedding_model:
                    where_clauses.append("embedding_model = %s")
                    params.append(embedding_model)
                
                where_sql = " AND ".join(where_clauses)
                
                query = f"""
                    SELECT id, category, name, embedding_model, embedding_dim,
                           embedding::text as embedding, metadata, image_url, added_at
                    FROM embeddings
                    WHERE {where_sql}
                    ORDER BY added_at;
                """
                
                cur.execute(query, params)
                rows = cur.fetchall()
                
                results = []
                for row in rows:
                    result = dict(row)
                    # Parse embedding vector string
                    if result["embedding"]:
                        embedding_str = result["embedding"].strip("[]")
                        result["embedding"] = [float(x) for x in embedding_str.split(",")]
                    results.append(result)
                
                return results
        except Exception as e:
            raise EmbeddingServiceError(f"Failed to get all embeddings: {e}", error_code="db_get_all_failed") from e
        finally:
            self._put_conn(conn)

    def close(self):
        """Close connection pool"""
        if hasattr(self, "pool"):
            self.pool.closeall()

