"""
pycontext/core/memory/semantic_memory.py

Semantic Memory implementation for PyContext.
"""
from typing import Dict, List, Optional, Any, Tuple, Set, Union
import json
import uuid
import time
from dataclasses import dataclass, field
import asyncio
import logging

from .embedding_provider import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass
class SemanticEntry:
    """An entry in semantic memory representing a piece of knowledge."""
    id: str
    content: Any
    entry_type: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_entries: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "entry_type": self.entry_type,
            "metadata": self.metadata,
            "related_entries": self.related_entries,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            # Only include embedding if it's not too large
            "embedding": self.embedding if (
                    self.embedding is None or len(self.embedding) <= 20
            ) else "Embedding too large to include in dict"
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SemanticEntry':
        """Create an entry from a dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            entry_type=data["entry_type"],
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
            related_entries=data.get("related_entries", []),
            timestamp=data.get("timestamp", time.time()),
            confidence=data.get("confidence", 1.0)
        )


class SimpleVectorStore:
    """
    A simple in-memory vector store for semantic entries.
    For production use, this would be replaced with a proper vector database.
    """

    def __init__(self):
        """Initialize the vector store."""
        self.entries: Dict[str, SemanticEntry] = {}
        self.index: Dict[str, List[str]] = {}  # Type -> list of entry IDs

    async def add(self, entry: SemanticEntry) -> str:
        """Add an entry to the store."""
        self.entries[entry.id] = entry

        # Update index
        if entry.entry_type not in self.index:
            self.index[entry.entry_type] = []
        self.index[entry.entry_type].append(entry.id)

        return entry.id

    async def update(self, entry: SemanticEntry) -> bool:
        """Update an existing entry."""
        if entry.id not in self.entries:
            return False

        self.entries[entry.id] = entry
        return True

    async def get(self, entry_id: str) -> Optional[SemanticEntry]:
        """Get an entry by ID."""
        return self.entries.get(entry_id)

    async def search(
            self,
            embedding: List[float],
            limit: int = 10,
            threshold: float = 0.0,
            entry_type: Optional[str] = None
    ) -> List[Tuple[SemanticEntry, float]]:
        """Search for entries using vector similarity."""
        results = []
        entry_ids = self.index.get(entry_type, []) if entry_type else list(self.entries.keys())

        for entry_id in entry_ids:
            entry = self.entries[entry_id]
            if entry.embedding is None:
                continue

            similarity = self._cosine_similarity(embedding, entry.embedding)
            if similarity >= threshold:
                results.append((entry, similarity))

        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:limit]

    async def get_by_type(
            self,
            entry_type: str,
            limit: Optional[int] = None
    ) -> List[SemanticEntry]:
        """Get entries by type."""
        entry_ids = self.index.get(entry_type, [])

        if limit is not None:
            entry_ids = entry_ids[:limit]

        return [self.entries[entry_id] for entry_id in entry_ids]

    async def remove(self, entry_id: str) -> bool:
        """Remove an entry from the store."""
        if entry_id not in self.entries:
            return False

        entry = self.entries[entry_id]

        # Remove from index
        if entry.entry_type in self.index and entry_id in self.index[entry.entry_type]:
            self.index[entry.entry_type].remove(entry_id)

        # Remove entry
        del self.entries[entry_id]

        return True

    async def clear(self) -> None:
        """Clear all entries from the store."""
        self.entries.clear()
        self.index.clear()

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0

        return dot_product / (magnitude1 * magnitude2)


class KnowledgeGraph:
    """
    A simple graph structure for representing semantic relationships.
    For production use, this would be replaced with a proper graph database.
    """

    def __init__(self):
        """Initialize the knowledge graph."""
        self.nodes: Dict[str, Dict[str, Any]] = {}  # Node ID -> attributes
        self.edges: Dict[str, Set[Tuple[str, str]]] = {}  # Edge type -> set of (source, target) tuples
        self.reverse_edges: Dict[str, Set[Tuple[str, str]]] = {}  # Edge type -> set of (target, source) tuples

    async def add_node(self, node_id: str, attributes: Dict[str, Any] = None) -> str:
        """
        Add a node to the graph.

        Args:
            node_id: Node identifier
            attributes: Node attributes

        Returns:
            Node identifier
        """
        self.nodes[node_id] = attributes or {}
        return node_id

    async def add_edge(
            self,
            source_id: str,
            target_id: str,
            edge_type: str,
            attributes: Dict[str, Any] = None
    ) -> bool:
        """
        Add an edge between nodes.

        Args:
            source_id: Source node identifier
            target_id: Target node identifier
            edge_type: Type of edge
            attributes: Edge attributes

        Returns:
            Whether the edge was added
        """
        # Check if nodes exist
        if source_id not in self.nodes or target_id not in self.nodes:
            return False

        # Create edge sets if needed
        if edge_type not in self.edges:
            self.edges[edge_type] = set()
        if edge_type not in self.reverse_edges:
            self.reverse_edges[edge_type] = set()

        # Add edge
        self.edges[edge_type].add((source_id, target_id))
        self.reverse_edges[edge_type].add((target_id, source_id))

        return True

    async def get_neighbors(
            self,
            node_id: str,
            edge_type: Optional[str] = None,
            direction: str = "outgoing"
    ) -> List[str]:
        """
        Get neighbors of a node.

        Args:
            node_id: Node identifier
            edge_type: Optional edge type filter
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of neighbor node IDs
        """
        if node_id not in self.nodes:
            return []

        neighbors = set()

        if direction in ["outgoing", "both"]:
            # Get outgoing neighbors
            edge_types = [edge_type] if edge_type else list(self.edges.keys())
            for et in edge_types:
                if et in self.edges:
                    neighbors.update(
                        target for source, target in self.edges[et]
                        if source == node_id
                    )

        if direction in ["incoming", "both"]:
            # Get incoming neighbors
            edge_types = [edge_type] if edge_type else list(self.reverse_edges.keys())
            for et in edge_types:
                if et in self.reverse_edges:
                    neighbors.update(
                        source for target, source in self.reverse_edges[et]
                        if target == node_id
                    )

        return list(neighbors)

    async def remove_node(self, node_id: str) -> bool:
        """
        Remove a node and all its edges.

        Args:
            node_id: Node identifier

        Returns:
            Whether the node was removed
        """
        if node_id not in self.nodes:
            return False

        # Remove node
        del self.nodes[node_id]

        # Remove edges
        for edge_type in self.edges:
            self.edges[edge_type] = {
                (source, target) for source, target in self.edges[edge_type]
                if source != node_id and target != node_id
            }

        for edge_type in self.reverse_edges:
            self.reverse_edges[edge_type] = {
                (target, source) for target, source in self.reverse_edges[edge_type]
                if source != node_id and target != node_id
            }

        return True

    async def clear(self) -> None:
        """Clear the graph."""
        self.nodes.clear()
        self.edges.clear()
        self.reverse_edges.clear()


class SemanticMemory:
    """
    Semantic memory for storing and retrieving knowledge.
    Combines vector storage with graph-based relationships.
    """

    def __init__(
            self,
            embedding_provider: Optional[EmbeddingProvider] = None,
            vector_store: Optional[Any] = None,
            knowledge_graph: Optional[Any] = None
    ):
        """
        Initialize the semantic memory.

        Args:
            embedding_provider: Provider for generating embeddings
            vector_store: Store for vector embeddings (optional)
            knowledge_graph: Graph for relationships (optional)
        """
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store or SimpleVectorStore()
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()

    async def add(
            self,
            content: Any,
            entry_type: str,
            related_entries: List[str] = None,
            metadata: Dict[str, Any] = None,
            confidence: float = 1.0
    ) -> str:
        """
        Add a knowledge entry to semantic memory.

        Args:
            content: Content of the entry
            entry_type: Type of entry
            related_entries: Related entry IDs
            metadata: Additional metadata
            confidence: Confidence in the entry (0-1)

        Returns:
            Entry ID
        """
        # Create entry ID
        entry_id = str(uuid.uuid4())

        # Generate embedding if possible
        embedding = None
        if self.embedding_provider:
            try:
                if isinstance(content, str):
                    embedding = await self.embedding_provider.embed(content)
                elif isinstance(content, dict) and "text" in content:
                    embedding = await self.embedding_provider.embed(content["text"])
            except Exception as e:
                logger.warning(f"Error generating embedding: {e}")

        # Create entry
        entry = SemanticEntry(
            id=entry_id,
            content=content,
            entry_type=entry_type,
            embedding=embedding,
            metadata=metadata or {},
            related_entries=related_entries or [],
            confidence=confidence
        )

        # Store in vector store
        await self.vector_store.add(entry)

        # Add to knowledge graph
        await self.knowledge_graph.add_node(
            entry_id,
            {
                "type": entry_type,
                "timestamp": entry.timestamp,
                "content_summary": self._get_content_summary(content)
            }
        )

        # Add relationships
        if related_entries:
            for related_id in related_entries:
                # Check if related entry exists
                related_entry = await self.vector_store.get(related_id)
                if related_entry:
                    await self.knowledge_graph.add_edge(
                        entry_id,
                        related_id,
                        "related_to"
                    )

        return entry_id

    async def get(self, entry_id: str) -> Optional[SemanticEntry]:
        """
        Get an entry by ID.

        Args:
            entry_id: Entry identifier

        Returns:
            Entry if found, None otherwise
        """
        return await self.vector_store.get(entry_id)

    async def search(
            self,
            query: str,
            limit: int = 10,
            threshold: float = 0.0,
            entry_type: Optional[str] = None,
            include_related: bool = True,
            max_related: int = 3
    ) -> List[Tuple[SemanticEntry, float]]:
        """
        Search for entries semantically similar to the query.

        Args:
            query: Search query
            limit: Maximum number of direct results
            threshold: Similarity threshold
            entry_type: Optional entry type filter
            include_related: Whether to include related entries
            max_related: Maximum number of related entries per result

        Returns:
            List of (entry, similarity) tuples
        """
        if not self.embedding_provider:
            return await self._fallback_search(query, limit, entry_type)

        try:
            # Generate query embedding
            query_embedding = await self.embedding_provider.embed(query)

            # Search vector store
            results = await self.vector_store.search(
                query_embedding,
                limit=limit,
                threshold=threshold,
                entry_type=entry_type
            )

            # If requested, include related entries
            if include_related and results:
                related_entries = []
                seen_ids = {result[0].id for result in results}

                for entry, _ in results:
                    # Get related entries from the graph
                    neighbors = await self.knowledge_graph.get_neighbors(
                        entry.id,
                        direction="both"
                    )

                    # Add only a limited number of new related entries
                    for neighbor_id in neighbors[:max_related]:
                        if neighbor_id not in seen_ids:
                            neighbor = await self.vector_store.get(neighbor_id)
                            if neighbor:
                                # Calculate similarity for consistent scoring
                                if neighbor.embedding:
                                    similarity = self._cosine_similarity(
                                        query_embedding,
                                        neighbor.embedding
                                    )
                                else:
                                    similarity = 0.0

                                related_entries.append((neighbor, similarity))
                                seen_ids.add(neighbor_id)

                # Add related entries to results
                results.extend(related_entries)

                # Re-sort by similarity
                results.sort(key=lambda x: x[1], reverse=True)

                # Limit to original limit
                results = results[:limit]

            return results

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return await self._fallback_search(query, limit, entry_type)

    async def add_relationship(
            self,
            source_id: str,
            target_id: str,
            relationship_type: str
    ) -> bool:
        """
        Add a relationship between entries.

        Args:
            source_id: Source entry ID
            target_id: Target entry ID
            relationship_type: Type of relationship

        Returns:
            Whether the relationship was added
        """
        # Check if entries exist
        source = await self.vector_store.get(source_id)
        target = await self.vector_store.get(target_id)

        if not source or not target:
            return False

        # Add edge to graph
        success = await self.knowledge_graph.add_edge(
            source_id,
            target_id,
            relationship_type
        )

        if success:
            # Update related entries in source
            if target_id not in source.related_entries:
                source.related_entries.append(target_id)
                await self.vector_store.update(source)

            # Update related entries in target
            if source_id not in target.related_entries:
                target.related_entries.append(source_id)
                await self.vector_store.update(target)

        return success

    async def get_related(
            self,
            entry_id: str,
            relationship_type: Optional[str] = None,
            limit: Optional[int] = None
    ) -> List[SemanticEntry]:
        """
        Get entries related to the specified entry.

        Args:
            entry_id: Entry identifier
            relationship_type: Optional relationship type filter
            limit: Maximum number of results

        Returns:
            List of related entries
        """
        # Check if entry exists
        entry = await self.vector_store.get(entry_id)
        if not entry:
            return []

        # Get related entry IDs from graph
        neighbor_ids = await self.knowledge_graph.get_neighbors(
            entry_id,
            edge_type=relationship_type,
            direction="both"
        )

        if limit is not None:
            neighbor_ids = neighbor_ids[:limit]

        # Get entries
        related_entries = []
        for neighbor_id in neighbor_ids:
            neighbor = await self.vector_store.get(neighbor_id)
            if neighbor:
                related_entries.append(neighbor)

        return related_entries

    async def remove(self, entry_id: str) -> bool:
        """
        Remove an entry and its relationships.

        Args:
            entry_id: Entry identifier

        Returns:
            Whether the entry was removed
        """
        # Remove from vector store
        success = await self.vector_store.remove(entry_id)

        if success:
            # Remove from knowledge graph
            await self.knowledge_graph.remove_node(entry_id)

        return success

    async def clear(self) -> None:
        """Clear all entries from semantic memory."""
        await self.vector_store.clear()
        await self.knowledge_graph.clear()

    async def _fallback_search(
            self,
            query: str,
            limit: int = 10,
            entry_type: Optional[str] = None
    ) -> List[Tuple[SemanticEntry, float]]:
        """
        Fallback text search when embeddings are not available.

        Args:
            query: Search query
            limit: Maximum number of results
            entry_type: Optional entry type filter

        Returns:
            List of (entry, score) tuples
        """
        query = query.lower()
        results = []

        # Get entries of the specified type, or all entries
        if entry_type:
            entries = await self.vector_store.get_by_type(entry_type)
        else:
            # This is inefficient but necessary for the fallback
            entries = [
                await self.vector_store.get(entry_id)
                for entry_id in self.knowledge_graph.nodes.keys()
            ]
            entries = [e for e in entries if e is not None]

        # Simple text matching
        for entry in entries:
            score = 0.0

            if isinstance(entry.content, str):
                if query in entry.content.lower():
                    # Crude score based on query length vs content length
                    score = len(query) / len(entry.content) if len(entry.content) > 0 else 0
            elif isinstance(entry.content, dict) and "text" in entry.content:
                if query in entry.content["text"].lower():
                    score = len(query) / len(entry.content["text"]) if len(entry.content["text"]) > 0 else 0

            if score > 0:
                results.append((entry, score))

        # Sort by score (highest first)
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:limit]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0

        return dot_product / (magnitude1 * magnitude2)

    def _get_content_summary(self, content: Any) -> str:
        """Get a summary of content for graph nodes."""
        if isinstance(content, str):
            # Truncate long strings
            if len(content) > 100:
                return content[:97] + "..."
            return content
        elif isinstance(content, dict) and "text" in content:
            text = content["text"]
            if len(text) > 100:
                return text[:97] + "..."
            return text
        elif isinstance(content, dict) and "title" in content:
            return content["title"]
        else:
            # For other types, use a generic summary
            return f"{type(content).__name__} object"
