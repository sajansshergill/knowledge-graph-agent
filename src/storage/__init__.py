"""
storage/
--------
Dual-store persistence layer for EKGA.

    AlloyDBLoader  — pgvector semantic + BM25 full-text search
    Neo4jLoader    — knowledge graph (nodes, edges, Cypher traversal)

Schema is defined in schema.sql (AlloyDB) and enforced at runtime
via Neo4jLoader.ensure_schema() (Neo4j constraints + indexes).
"""

from .alloydb_loader import AlloyDBLoader, LoadResult
from .neo4j_loader import Neo4jLoader, GraphLoadResult

__all__ = [
    "AlloyDBLoader",
    "LoadResult",
    "Neo4jLoader",
    "GraphLoadResult",
]