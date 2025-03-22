"""
tests/unit/core/memory/test_semantic_memory.py

Tests for the Semantic Memory module.
"""
import unittest
import asyncio
from unittest.mock import MagicMock, patch
import time

from pycontext.core.memory.semantic_memory import (
    SemanticEntry,
    SemanticMemory,
    SimpleVectorStore,
    KnowledgeGraph
)
from pycontext.core.memory.embedding_provider import SimpleEmbeddingProvider


class TestSemanticEntry(unittest.TestCase):
    """Test the SemanticEntry class."""

    def test_creation(self):
        """Test entry creation."""
        entry = SemanticEntry(
            id="test-id",
            content="Test content",
            entry_type="test_type",
            embedding=[0.1, 0.2, 0.3],
            metadata={"key": "value"},
            related_entries=["rel1", "rel2"],
            confidence=0.8
        )

        self.assertEqual(entry.id, "test-id")
        self.assertEqual(entry.content, "Test content")
        self.assertEqual(entry.entry_type, "test_type")
        self.assertEqual(entry.embedding, [0.1, 0.2, 0.3])
        self.assertEqual(entry.metadata["key"], "value")
        self.assertEqual(entry.related_entries, ["rel1", "rel2"])
        self.assertEqual(entry.confidence, 0.8)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        entry = SemanticEntry(
            id="test-id",
            content="Test content",
            entry_type="test_type",
            embedding=[0.1, 0.2, 0.3],
            metadata={"key": "value"},
            related_entries=["rel1", "rel2"],
            timestamp=1609459200
        )

        entry_dict = entry.to_dict()

        self.assertEqual(entry_dict["id"], "test-id")
        self.assertEqual(entry_dict["content"], "Test content")
        self.assertEqual(entry_dict["entry_type"], "test_type")
        self.assertEqual(entry_dict["embedding"], [0.1, 0.2, 0.3])
        self.assertEqual(entry_dict["metadata"]["key"], "value")
        self.assertEqual(entry_dict["related_entries"], ["rel1", "rel2"])
        self.assertEqual(entry_dict["timestamp"], 1609459200)

    def test_from_dict(self):
        """Test creation from dictionary."""
        entry_dict = {
            "id": "test-id",
            "content": "Test content",
            "entry_type": "test_type",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {"key": "value"},
            "related_entries": ["rel1", "rel2"],
            "timestamp": 1609459200,
            "confidence": 0.8
        }

        entry = SemanticEntry.from_dict(entry_dict)

        self.assertEqual(entry.id, "test-id")
        self.assertEqual(entry.content, "Test content")
        self.assertEqual(entry.entry_type, "test_type")
        self.assertEqual(entry.embedding, [0.1, 0.2, 0.3])
        self.assertEqual(entry.metadata["key"], "value")
        self.assertEqual(entry.related_entries, ["rel1", "rel2"])
        self.assertEqual(entry.timestamp, 1609459200)
        self.assertEqual(entry.confidence, 0.8)

    def test_long_embedding_handling(self):
        """Test handling of long embeddings in to_dict."""
        # Create an entry with a long embedding
        long_embedding = [0.1] * 100
        entry = SemanticEntry(
            id="test-id",
            content="Test content",
            entry_type="test_type",
            embedding=long_embedding
        )

        entry_dict = entry.to_dict()

        # Embedding should be summarized
        self.assertEqual(entry_dict["embedding"], "Embedding too large to include in dict")


class TestSimpleVectorStore(unittest.TestCase):
    """Test the SimpleVectorStore class."""

    def setUp(self):
        """Set up the test environment."""
        self.store = SimpleVectorStore()

        # Add some test entries
        self.entries = []
        for i in range(3):
            self.entries.append(SemanticEntry(
                id=f"entry-{i}",
                content=f"Content {i}",
                entry_type="test" if i < 2 else "other",
                embedding=[0.1 * i, 0.2 * i, 0.3 * i]
            ))

    async def add_test_entries(self):
        """Add test entries to the store."""
        for entry in self.entries:
            await self.store.add(entry)

    def test_add_and_get(self):
        """Test adding and retrieving entries."""
        asyncio.run(self._test_add_and_get())

    async def _test_add_and_get(self):
        # Add entries
        for entry in self.entries:
            entry_id = await self.store.add(entry)
            self.assertEqual(entry_id, entry.id)

        # Get entries
        for entry in self.entries:
            stored_entry = await self.store.get(entry.id)
            self.assertEqual(stored_entry, entry)

    def test_update(self):
        """Test updating entries."""
        asyncio.run(self._test_update())

    async def _test_update(self):
        # Add entries
        await self.add_test_entries()

        # Update an entry
        updated_entry = SemanticEntry(
            id=self.entries[0].id,
            content="Updated content",
            entry_type=self.entries[0].entry_type,
            embedding=[0.5, 0.5, 0.5]
        )

        success = await self.store.update(updated_entry)
        self.assertTrue(success)

        # Verify update
        stored_entry = await self.store.get(updated_entry.id)
        self.assertEqual(stored_entry.content, "Updated content")
        self.assertEqual(stored_entry.embedding, [0.5, 0.5, 0.5])

        # Try to update non-existent entry
        non_existent = SemanticEntry(
            id="non-existent",
            content="Non-existent",
            entry_type="test"
        )

        success = await self.store.update(non_existent)
        self.assertFalse(success)

    def test_search(self):
        """Test searching by vector similarity."""
        asyncio.run(self._test_search())

    async def _test_search(self):
        # Add entries
        await self.add_test_entries()

        # Search with a query vector
        query = [0.1, 0.2, 0.3]
        results = await self.store.search(query, limit=2)

        # Check results
        self.assertEqual(len(results), 2)

        # First result should be most similar
        self.assertEqual(results[0][0].id, "entry-1")  # Most similar to query

        # With threshold
        results = await self.store.search(query, threshold=0.9)
        self.assertTrue(len(results) < 3)  # Some entries should be filtered out

        # With entry type filter
        results = await self.store.search(query, entry_type="other")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].id, "entry-2")

    def test_get_by_type(self):
        """Test retrieving entries by type."""
        asyncio.run(self._test_get_by_type())

    async def _test_get_by_type(self):
        # Add entries
        await self.add_test_entries()

        # Get entries by type
        test_entries = await self.store.get_by_type("test")
        other_entries = await self.store.get_by_type("other")

        # Check results
        self.assertEqual(len(test_entries), 2)
        self.assertEqual(len(other_entries), 1)

        entry_ids = [e.id for e in test_entries]
        self.assertIn("entry-0", entry_ids)
        self.assertIn("entry-1", entry_ids)

        # Test with limit
        limited_entries = await self.store.get_by_type("test", limit=1)
        self.assertEqual(len(limited_entries), 1)

    def test_remove(self):
        """Test removing entries."""
        asyncio.run(self._test_remove())

    async def _test_remove(self):
        # Add entries
        await self.add_test_entries()

        # Remove an entry
        success = await self.store.remove("entry-1")
        self.assertTrue(success)

        # Verify removal
        entry = await self.store.get("entry-1")
        self.assertIsNone(entry)

        # Check entry type index
        test_entries = await self.store.get_by_type("test")
        self.assertEqual(len(test_entries), 1)
        self.assertEqual(test_entries[0].id, "entry-0")

        # Try to remove non-existent entry
        success = await self.store.remove("non-existent")
        self.assertFalse(success)

    def test_clear(self):
        """Test clearing all entries."""
        asyncio.run(self._test_clear())

    async def _test_clear(self):
        # Add entries
        await self.add_test_entries()

        # Clear store
        await self.store.clear()

        # Verify clearing
        for entry in self.entries:
            stored_entry = await self.store.get(entry.id)
            self.assertIsNone(stored_entry)

        # Check index
        test_entries = await self.store.get_by_type("test")
        self.assertEqual(len(test_entries), 0)


class TestKnowledgeGraph(unittest.TestCase):
    """Test the KnowledgeGraph class."""

    def setUp(self):
        """Set up the test environment."""
        self.graph = KnowledgeGraph()

    def test_add_and_get_nodes(self):
        """Test adding and retrieving nodes."""
        asyncio.run(self._test_add_and_get_nodes())

    async def _test_add_and_get_nodes(self):
        # Add nodes
        node1_id = await self.graph.add_node("node1", {"type": "person"})
        node2_id = await self.graph.add_node("node2", {"type": "document"})

        # Check IDs
        self.assertEqual(node1_id, "node1")
        self.assertEqual(node2_id, "node2")

        # Check nodes exist
        self.assertIn("node1", self.graph.nodes)
        self.assertIn("node2", self.graph.nodes)

        # Check attributes
        self.assertEqual(self.graph.nodes["node1"]["type"], "person")
        self.assertEqual(self.graph.nodes["node2"]["type"], "document")

    def test_add_and_get_edges(self):
        """Test adding and retrieving edges."""
        asyncio.run(self._test_add_and_get_edges())

    async def _test_add_and_get_edges(self):
        # Add nodes
        await self.graph.add_node("node1", {"type": "person"})
        await self.graph.add_node("node2", {"type": "document"})
        await self.graph.add_node("node3", {"type": "topic"})

        # Add edges
        success1 = await self.graph.add_edge("node1", "node2", "authored")
        success2 = await self.graph.add_edge("node2", "node3", "relates_to")
        success3 = await self.graph.add_edge("node1", "node3", "interested_in")

        # Check success
        self.assertTrue(success1)
        self.assertTrue(success2)
        self.assertTrue(success3)

        # Try to add edge with non-existent node
        success4 = await self.graph.add_edge("node1", "node4", "knows")
        self.assertFalse(success4)

        # Check edges
        self.assertIn(("node1", "node2"), self.graph.edges["authored"])
        self.assertIn(("node2", "node3"), self.graph.edges["relates_to"])
        self.assertIn(("node1", "node3"), self.graph.edges["interested_in"])

        # Check reverse edges
        self.assertIn(("node2", "node1"), self.graph.reverse_edges["authored"])
        self.assertIn(("node3", "node2"), self.graph.reverse_edges["relates_to"])
        self.assertIn(("node3", "node1"), self.graph.reverse_edges["interested_in"])

    def test_get_neighbors(self):
        """Test getting node neighbors."""
        asyncio.run(self._test_get_neighbors())

    async def _test_get_neighbors(self):
        # Add nodes
        await self.graph.add_node("node1", {"type": "person"})
        await self.graph.add_node("node2", {"type": "document"})
        await self.graph.add_node("node3", {"type": "topic"})
        await self.graph.add_node("node4", {"type": "person"})

        # Add edges
        await self.graph.add_edge("node1", "node2", "authored")
        await self.graph.add_edge("node2", "node3", "relates_to")
        await self.graph.add_edge("node4", "node1", "knows")
        await self.graph.add_edge("node4", "node2", "read")

        # Get outgoing neighbors
        outgoing = await self.graph.get_neighbors("node1", direction="outgoing")
        self.assertEqual(len(outgoing), 1)
        self.assertIn("node2", outgoing)

        # Get incoming neighbors
        incoming = await self.graph.get_neighbors("node1", direction="incoming")
        self.assertEqual(len(incoming), 1)
        self.assertIn("node4", incoming)

        # Get all neighbors
        all_neighbors = await self.graph.get_neighbors("node1", direction="both")
        self.assertEqual(len(all_neighbors), 2)
        self.assertIn("node2", all_neighbors)
        self.assertIn("node4", all_neighbors)

        # Filter by edge type
        authored_neighbors = await self.graph.get_neighbors("node1", edge_type="authored")
        self.assertEqual(len(authored_neighbors), 1)
        self.assertIn("node2", authored_neighbors)

        # Node with multiple connections
        node4_neighbors = await self.graph.get_neighbors("node4", direction="outgoing")
        self.assertEqual(len(node4_neighbors), 2)
        self.assertIn("node1", node4_neighbors)
        self.assertIn("node2", node4_neighbors)

    def test_remove_node(self):
        """Test removing nodes and their edges."""
        asyncio.run(self._test_remove_node())

    async def _test_remove_node(self):
        # Add nodes
        await self.graph.add_node("node1", {"type": "person"})
        await self.graph.add_node("node2", {"type": "document"})
        await self.graph.add_node("node3", {"type": "topic"})

        # Add edges
        await self.graph.add_edge("node1", "node2", "authored")
        await self.graph.add_edge("node2", "node3", "relates_to")

        # Remove a node
        success = await self.graph.remove_node("node2")
        self.assertTrue(success)

        # Verify node removal
        self.assertNotIn("node2", self.graph.nodes)

        # Verify edge removal
        self.assertNotIn(("node1", "node2"), self.graph.edges.get("authored", set()))
        self.assertNotIn(("node2", "node3"), self.graph.edges.get("relates_to", set()))

        # Try to remove non-existent node
        success = await self.graph.remove_node("non-existent")
        self.assertFalse(success)

    def test_clear(self):
        """Test clearing the graph."""
        asyncio.run(self._test_clear())

    async def _test_clear(self):
        # Add nodes and edges
        await self.graph.add_node("node1")
        await self.graph.add_node("node2")
        await self.graph.add_edge("node1", "node2", "related")

        # Clear graph
        await self.graph.clear()

        # Verify clearing
        self.assertEqual(len(self.graph.nodes), 0)
        self.assertEqual(len(self.graph.edges), 0)
        self.assertEqual(len(self.graph.reverse_edges), 0)


class TestSemanticMemory(unittest.TestCase):
    """Test the SemanticMemory class."""

    def setUp(self):
        """Set up the test environment."""
        self.embedding_provider = SimpleEmbeddingProvider(dimension=3)
        self.memory = SemanticMemory(embedding_provider=self.embedding_provider)

    def test_add_and_get(self):
        """Test adding and retrieving entries."""
        asyncio.run(self._test_add_and_get())

    async def _test_add_and_get(self):
        # Add an entry
        entry_id = await self.memory.add(
            content="This is a test entry about artificial intelligence.",
            entry_type="knowledge",
            metadata={"source": "test"}
        )

        # Get the entry
        entry = await self.memory.get(entry_id)

        # Check entry
        self.assertIsNotNone(entry)
        self.assertEqual(entry.content, "This is a test entry about artificial intelligence.")
        self.assertEqual(entry.entry_type, "knowledge")
        self.assertEqual(entry.metadata["source"], "test")
        self.assertIsNotNone(entry.embedding)  # Should have generated an embedding

    def test_add_with_relationships(self):
        """Test adding entries with relationships."""
        asyncio.run(self._test_add_with_relationships())

    async def _test_add_with_relationships(self):
        # Add first entry
        entry1_id = await self.memory.add(
            content="This is about machine learning.",
            entry_type="knowledge"
        )

        # Add second entry with relationship to first
        entry2_id = await self.memory.add(
            content="Deep learning is a subset of machine learning.",
            entry_type="knowledge",
            related_entries=[entry1_id]
        )

        # Get entries
        entry1 = await self.memory.get(entry1_id)
        entry2 = await self.memory.get(entry2_id)

        # Check relationship in graph
        neighbors = await self.memory.knowledge_graph.get_neighbors(entry2_id)
        self.assertIn(entry1_id, neighbors)

        # Check related_entries field
        self.assertIn(entry1_id, entry2.related_entries)

    @patch.object(SimpleEmbeddingProvider, 'embed')
    def test_search(self, mock_embed):
        """Test semantic search."""
        # Mock embedding generation to return predictable vectors
        mock_embed.side_effect = lambda text: [0.1, 0.2, 0.3]

        asyncio.run(self._test_search())

    async def _test_search(self):
        # Add some entries
        await self.memory.add(
            content="Python is a programming language.",
            entry_type="knowledge"
        )
        await self.memory.add(
            content="Machine learning is a subset of artificial intelligence.",
            entry_type="knowledge"
        )
        await self.memory.add(
            content="Natural language processing involves computers understanding human language.",
            entry_type="knowledge"
        )

        # Search for entries
        results = await self.memory.search("programming language Python")

        # Check results
        self.assertGreater(len(results), 0)

        # First result should match the query
        self.assertTrue("python" in results[0][0].content.lower())

        # With entry type filter
        results = await self.memory.search(
            "artificial intelligence",
            entry_type="knowledge"
        )
        self.assertGreater(len(results), 0)

        # Test threshold
        results = await self.memory.search(
            "unrelated query",
            threshold=0.9  # High threshold should filter out weak matches
        )
        self.assertEqual(len(results), 0)

    def test_add_relationship(self):
        """Test adding relationships between entries."""
        asyncio.run(self._test_add_relationship())

    async def _test_add_relationship(self):
        # Add entries
        entry1_id = await self.memory.add(
            content="Python basics",
            entry_type="course"
        )
        entry2_id = await self.memory.add(
            content="Advanced Python",
            entry_type="course"
        )

        # Add relationship
        success = await self.memory.add_relationship(
            entry1_id,
            entry2_id,
            "prerequisite_for"
        )

        # Check success
        self.assertTrue(success)

        # Check relationship in graph
        neighbors = await self.memory.knowledge_graph.get_neighbors(
            entry1_id,
            edge_type="prerequisite_for"
        )
        self.assertIn(entry2_id, neighbors)

        # Check related entries fields
        entry1 = await self.memory.get(entry1_id)
        entry2 = await self.memory.get(entry2_id)

        self.assertIn(entry2_id, entry1.related_entries)
        self.assertIn(entry1_id, entry2.related_entries)

        # Try adding relationship with non-existent entry
        success = await self.memory.add_relationship(
            entry1_id,
            "non-existent",
            "knows"
        )
        self.assertFalse(success)

    def test_get_related(self):
        """Test getting related entries."""
        asyncio.run(self._test_get_related())

    async def _test_get_related(self):
        # Add entries with relationships
        entry1_id = await self.memory.add(
            content="Biology basics",
            entry_type="course"
        )
        entry2_id = await self.memory.add(
            content="Human anatomy",
            entry_type="course"
        )
        entry3_id = await self.memory.add(
            content="Cell biology",
            entry_type="course"
        )

        # Add relationships
        await self.memory.add_relationship(entry1_id, entry2_id, "related_to")
        await self.memory.add_relationship(entry1_id, entry3_id, "includes")
        await self.memory.add_relationship(entry3_id, entry2_id, "related_to")

        # Get all related entries
        related = await self.memory.get_related(entry1_id)
        self.assertEqual(len(related), 2)
        related_ids = [e.id for e in related]
        self.assertIn(entry2_id, related_ids)
        self.assertIn(entry3_id, related_ids)

        # Filter by relationship type
        includes_related = await self.memory.get_related(entry1_id, relationship_type="includes")
        self.assertEqual(len(includes_related), 1)
        self.assertEqual(includes_related[0].id, entry3_id)

        # With limit
        limited_related = await self.memory.get_related(entry1_id, limit=1)
        self.assertEqual(len(limited_related), 1)

    def test_remove(self):
        """Test removing entries."""
        asyncio.run(self._test_remove())

    async def _test_remove(self):
        # Add entries with relationships
        entry1_id = await self.memory.add(
            content="Entry 1",
            entry_type="test"
        )
        entry2_id = await self.memory.add(
            content="Entry 2",
            entry_type="test",
            related_entries=[entry1_id]
        )

        # Remove an entry
        success = await self.memory.remove(entry1_id)
        self.assertTrue(success)

        # Verify removal
        entry = await self.memory.get(entry1_id)
        self.assertIsNone(entry)

        # Check relationship removal
        neighbors = await self.memory.knowledge_graph.get_neighbors(entry2_id)
        self.assertNotIn(entry1_id, neighbors)

        # Try to remove non-existent entry
        success = await self.memory.remove("non-existent")
        self.assertFalse(success)

    def test_clear(self):
        """Test clearing all entries."""
        asyncio.run(self._test_clear())

    async def _test_clear(self):
        # Add some entries
        await self.memory.add(
            content="Entry 1",
            entry_type="test"
        )
        await self.memory.add(
            content="Entry 2",
            entry_type="test"
        )

        # Clear memory
        await self.memory.clear()

        # Verify vector store is cleared
        self.assertEqual(len(self.memory.vector_store.entries), 0)
        self.assertEqual(len(self.memory.vector_store.index), 0)

        # Verify graph is cleared
        self.assertEqual(len(self.memory.knowledge_graph.nodes), 0)
        self.assertEqual(len(self.memory.knowledge_graph.edges), 0)

    def test_fallback_search(self):
        """Test fallback search when embeddings are not available."""
        asyncio.run(self._test_fallback_search())

    async def _test_fallback_search(self):
        # Create memory without embedding provider
        memory = SemanticMemory(embedding_provider=None)

        # Add some entries
        await memory.add(
            content="Python programming for beginners",
            entry_type="article"
        )
        await memory.add(
            content="Advanced Java techniques",
            entry_type="article"
        )
        await memory.add(
            content="Python vs Java performance comparison",
            entry_type="article"
        )

        # Search
        results = await memory.search("python")

        # Check results
        self.assertGreater(len(results), 0)
        self.assertTrue(all("python" in r[0].content.lower() for r in results))

        # With entry type
        results = await memory.search(
            "java",
            entry_type="article"
        )
        self.assertGreater(len(results), 0)
        self.assertTrue(all("java" in r[0].content.lower() for r in results))

    def test_content_summary(self):
        """Test the content summary generation."""
        memory = SemanticMemory()

        # Test with string
        summary1 = memory._get_content_summary("Short text")
        self.assertEqual(summary1, "Short text")

        # Test with long string
        long_text = "This is a very long text that should be truncated" + " very long" * 20
        summary2 = memory._get_content_summary(long_text)
        self.assertTrue(len(summary2) <= 100)
        self.assertTrue(summary2.endswith("..."))

        # Test with dict containing text
        dict_content = {"text": "Text in dictionary"}
        summary3 = memory._get_content_summary(dict_content)
        self.assertEqual(summary3, "Text in dictionary")

        # Test with dict containing title
        dict_with_title = {"title": "Title of content"}
        summary4 = memory._get_content_summary(dict_with_title)
        self.assertEqual(summary4, "Title of content")

        # Test with other type
        list_content = [1, 2, 3]
        summary5 = memory._get_content_summary(list_content)
        self.assertEqual(summary5, "list object")


class TestVectorStoreIntegration(unittest.TestCase):
    """Test integration with a custom vector store."""

    class MockVectorStore:
        """Mock vector store for testing."""

        def __init__(self):
            self.entries = {}
            self.index = {}

        async def add(self, entry):
            self.entries[entry.id] = entry
            if entry.entry_type not in self.index:
                self.index[entry.entry_type] = []
            self.index[entry.entry_type].append(entry.id)
            return entry.id

        async def get(self, entry_id):
            return self.entries.get(entry_id)

        async def search(self, embedding, limit=10, threshold=0, entry_type=None):
            # Return some mock results
            results = []
            for entry_id, entry in self.entries.items():
                if entry_type and entry.entry_type != entry_type:
                    continue
                results.append((entry, 0.8))  # Mock similarity score
            return results[:limit]

        async def get_by_type(self, entry_type, limit=None):
            results = []
            if entry_type in self.index:
                ids = self.index[entry_type]
                if limit:
                    ids = ids[:limit]
                for entry_id in ids:
                    if entry_id in self.entries:
                        results.append(self.entries[entry_id])
            return results

        async def remove(self, entry_id):
            if entry_id in self.entries:
                entry = self.entries[entry_id]
                if entry.entry_type in self.index:
                    if entry_id in self.index[entry.entry_type]:
                        self.index[entry.entry_type].remove(entry_id)
                del self.entries[entry_id]
                return True
            return False

        async def clear(self):
            self.entries.clear()
            self.index.clear()

    def setUp(self):
        """Set up the test environment."""
        self.vector_store = self.MockVectorStore()
        self.memory = SemanticMemory(vector_store=self.vector_store)

    def test_custom_vector_store(self):
        """Test using a custom vector store."""
        asyncio.run(self._test_custom_vector_store())

    async def _test_custom_vector_store(self):
        # Add some entries
        entry_id = await self.memory.add(
            content="Test content",
            entry_type="test"
        )

        # Check storage
        entry = await self.vector_store.get(entry_id)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.content, "Test content")

        # Test search
        results = await self.memory.search("test")
        self.assertGreater(len(results), 0)

        # Test removal
        success = await self.memory.remove(entry_id)
        self.assertTrue(success)
        entry = await self.vector_store.get(entry_id)
        self.assertIsNone(entry)


class TestSemanticMemoryIntegration(unittest.TestCase):
    """Integration tests for the SemanticMemory class."""

    def setUp(self):
        """Set up the test environment."""
        self.embedding_provider = SimpleEmbeddingProvider(dimension=3)
        self.vector_store = SimpleVectorStore()
        self.knowledge_graph = KnowledgeGraph()
        self.memory = SemanticMemory(
            embedding_provider=self.embedding_provider,
            vector_store=self.vector_store,
            knowledge_graph=self.knowledge_graph
        )

    def test_full_workflow(self):
        """Test a complete workflow with multiple operations."""
        asyncio.run(self._test_full_workflow())

    async def _test_full_workflow(self):
        # 1. Add some entries
        entries = []
        for i in range(5):
            entry_id = await self.memory.add(
                content=f"Entry {i + 1} content",
                entry_type="test",
                metadata={"index": i}
            )
            entries.append(entry_id)

        # 2. Add some relationships
        for i in range(len(entries) - 1):
            await self.memory.add_relationship(
                entries[i],
                entries[i + 1],
                "next"
            )

        # 3. Perform a search
        results = await self.memory.search("content")
        self.assertEqual(len(results), 5)

        # 4. Get related entries
        related = await self.memory.get_related(entries[0])
        self.assertEqual(len(related), 1)
        self.assertEqual(related[0].id, entries[1])

        # 5. Remove an entry
        await self.memory.remove(entries[2])

        # 6. Check that entry is gone
        entry = await self.memory.get(entries[2])
        self.assertIsNone(entry)

        # 7. Check that relationships are updated
        related_to_1 = await self.memory.get_related(entries[1])
        self.assertEqual(len(related_to_1), 0)  # Connection to entry 2 was removed

        # 8. Clear everything
        await self.memory.clear()

        # 9. Verify everything is cleared
        for entry_id in entries:
            entry = await self.memory.get(entry_id)
            self.assertIsNone(entry)


if __name__ == "__main__":
    unittest.main()
    