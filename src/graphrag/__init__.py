"""
GraphRAG module for the Smartphone Knowledge Graph.

Two approaches:
1. nl_to_sparql: Transform natural language to SPARQL queries
2. embedding_rag: Use embeddings for semantic search and QA
"""

from .nl_to_sparql import NLToSPARQL, QueryResult

__all__ = ["NLToSPARQL", "QueryResult"]
