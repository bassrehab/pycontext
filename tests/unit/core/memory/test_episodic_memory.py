"""
tests/unit/core/memory/test_episodic_memory.py

Tests for the Episodic Memory module.
"""
import unittest
import asyncio
import time
import tempfile
import os
from unittest.mock import MagicMock, patch

from pycontext.core.memory.episodic_memory import EpisodicMemory, Episode
from pycontext.core.memory.embedding_provider import SimpleEmbeddingProvider


class TestEpisodicMemory(unittest.TestCase):
    """Test the Episodic Memory implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.memory = EpisodicMemory(max_episodes=5)

    def test_add_and_get(self):
        """Test adding and retrieving episodes."""
        # Add an episode
        episode_id = asyncio.run(self.memory.add(
            content="Test episode content",
            episode_type="test_type",
            metadata={"key": "value"}
        ))

        # Get the episode
        episode = self.memory.get(episode_id)

        # Check
        self.assertIsNotNone(episode)
        self.assertEqual(episode.content, "Test episode content")
        self.assertEqual(episode.episode_type, "test_type")
        self.assertEqual(episode.metadata["key"], "value")

    def test_get_by_type(self):
        """Test retrieving episodes by type."""
        # Add episodes of different types
        asyncio.run(self.memory.add(
            content="Episode 1",
            episode_type="type_a"
        ))
        asyncio.run(self.memory.add(
            content="Episode 2",
            episode_type="type_a"
        ))
        asyncio.run(self.memory.add(
            content="Episode 3",
            episode_type="type_b"
        ))

        # Get episodes by type
        type_a_episodes = self.memory.get_by_type("type_a")
        type_b_episodes = self.memory.get_by_type("type_b")
        type_c_episodes = self.memory.get_by_type("type_c")  # Non-existent type

        # Check
        self.assertEqual(len(type_a_episodes), 2)
        self.assertEqual(len(type_b_episodes), 1)
        self.assertEqual(len(type_c_episodes), 0)

        # Check content
        type_a_contents = [e.content for e in type_a_episodes]
        self.assertIn("Episode 1", type_a_contents)
        self.assertIn("Episode 2", type_a_contents)

        type_b_contents = [e.content for e in type_b_episodes]
        self.assertIn("Episode 3", type_b_contents)

    def test_get_by_reference(self):
        """Test retrieving episodes by reference."""
        # Add an episode
        episode_id = asyncio.run(self.memory.add(
            content="Original episode",
            episode_type="original"
        ))

        # Add episodes that reference the original
        asyncio.run(self.memory.add(
            content="Reference 1",
            episode_type="reference",
            references=[episode_id]
        ))
        asyncio.run(self.memory.add(
            content="Reference 2",
            episode_type="reference",
            references=[episode_id]
        ))
        asyncio.run(self.memory.add(
            content="No reference",
            episode_type="reference"
        ))

        # Get episodes by reference
        referencing_episodes = self.memory.get_by_reference(episode_id)

        # Check
        self.assertEqual(len(referencing_episodes), 2)
        reference_contents = [e.content for e in referencing_episodes]
        self.assertIn("Reference 1", reference_contents)
        self.assertIn("Reference 2", reference_contents)
        self.assertNotIn("No reference", reference_contents)

    def test_get_by_metadata(self):
        """Test retrieving episodes by metadata."""
        # Add episodes with different metadata
        asyncio.run(self.memory.add(
            content="Episode A1",
            episode_type="test",
            metadata={"category": "A", "priority": "high"}
        ))
        asyncio.run(self.memory.add(
            content="Episode A2",
            episode_type="test",
            metadata={"category": "A", "priority": "low"}
        ))
        asyncio.run(self.memory.add(
            content="Episode B",
            episode_type="test",
            metadata={"category": "B", "priority": "high"}
        ))

        # Get episodes by metadata
        category_a = self.memory.get_by_metadata("category", "A")
        high_priority = self.memory.get_by_metadata("priority", "high")

        # Check
        self.assertEqual(len(category_a), 2)
        self.assertEqual(len(high_priority), 2)

        category_a_contents = [e.content for e in category_a]
        self.assertIn("Episode A1", category_a_contents)
        self.assertIn("Episode A2", category_a_contents)

        high_priority_contents = [e.content for e in high_priority]
        self.assertIn("Episode A1", high_priority_contents)
        self.assertIn("Episode B", high_priority_contents)

    def test_get_recent(self):
        """Test getting recent episodes."""
        # Add episodes with slight delay between them
        asyncio.run(self.memory.add(
            content="Episode 1",
            episode_type="test"
        ))
        time.sleep(0.01)

        asyncio.run(self.memory.add(
            content="Episode 2",
            episode_type="test"
        ))
        time.sleep(0.01)

        asyncio.run(self.memory.add(
            content="Episode 3",
            episode_type="test"
        ))

        # Get recent episodes
        recent = self.memory.get_recent(limit=2)

        # Check
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0].content, "Episode 3")
        self.assertEqual(recent[1].content, "Episode 2")

    def test_search_with_text(self):
        """Test searching episodes with text."""
        # Add episodes with different content
        asyncio.run(self.memory.add(
            content="This is about programming in Python",
            episode_type="test"
        ))
        asyncio.run(self.memory.add(
            content="Java is another programming language",
            episode_type="test"
        ))
        asyncio.run(self.memory.add(
            content="Cats and dogs are animals",
            episode_type="test"
        ))

        # Search for episodes
        programming_results = self.memory.search("programming")
        python_results = self.memory.search("python")
        animal_results = self.memory.search("animal")

        # Check
        self.assertEqual(len(programming_results), 2)
        self.assertEqual(len(python_results), 1)
        self.assertEqual(len(animal_results), 1)

        # Check first result content
        self.assertTrue("programming" in programming_results[0][0].content.lower())
        self.assertTrue("python" in python_results[0][0].content.lower())
        self.assertTrue("animal" in animal_results[0][0].content.lower())

    def test_remove(self):
        """Test removing episodes."""
        # Add an episode
        episode_id = asyncio.run(self.memory.add(
            content="Episode to remove",
            episode_type="test"
        ))

        # Verify it exists
        self.assertIsNotNone(self.memory.get(episode_id))

        # Remove it
        success = self.memory.remove(episode_id)

        # Check
        self.assertTrue(success)
        self.assertIsNone(self.memory.get(episode_id))

        # Try to remove non-existent episode
        success = self.memory.remove("non-existent-id")
        self.assertFalse(success)

    def test_clear(self):
        """Test clearing all episodes."""
        # Add some episodes
        for i in range(3):
            asyncio.run(self.memory.add(
                content=f"Episode {i + 1}",
                episode_type="test"
            ))

        # Verify they exist
        self.assertEqual(len(self.memory.episodes), 3)

        # Clear memory
        self.memory.clear()

        # Check
        self.assertEqual(len(self.memory.episodes), 0)
        self.assertEqual(len(self.memory.episode_types), 0)

    def test_max_episodes(self):
        """Test max episodes limit."""
        # Add episodes up to the limit (5) and beyond
        for i in range(7):
            asyncio.run(self.memory.add(
                content=f"Episode {i + 1}",
                episode_type="test"
            ))

        # Check only max_episodes are kept
        self.assertEqual(len(self.memory.episodes), 5)

        # Check the oldest episodes were removed
        episodes = self.memory.get_recent(limit=10)
        contents = [e.content for e in episodes]

        self.assertIn("Episode 7", contents)
        self.assertIn("Episode 6", contents)
        self.assertNotIn("Episode 1", contents)
        self.assertNotIn("Episode 2", contents)

    def test_save_and_load(self):
        """Test saving and loading episodes from a file."""
        # Add some episodes
        for i in range(3):
            asyncio.run(self.memory.add(
                content=f"Episode {i + 1}",
                episode_type="test",
                metadata={"index": i}
            ))

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            filename = temp.name
            self.memory.save_to_file(filename)

        # Create a new memory instance
        new_memory = EpisodicMemory()

        # Load from the file
        new_memory.load_from_file(filename)

        # Clean up
        os.unlink(filename)

        # Check if episodes were loaded correctly
        self.assertEqual(len(new_memory.episodes), 3)

        # Verify contents
        episodes = new_memory.get_by_type("test")
        contents = set(e.content for e in episodes)
        expected = {"Episode 1", "Episode 2", "Episode 3"}
        self.assertEqual(contents, expected)

        # Verify metadata was preserved
        for episode in episodes:
            self.assertIn("index", episode.metadata)

    @patch('pycontext.core.memory.episodic_memory.EpisodicMemory._generate_embedding')
    async def test_with_embeddings(self, mock_generate_embedding):
        """Test episodic memory with embeddings."""
        # Mock the embedding generation
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3, 0.4]

        # Create memory with embedding support
        embedding_provider = SimpleEmbeddingProvider(dimension=4)
        memory = EpisodicMemory(use_embeddings=True, embedding_provider=embedding_provider)

        # Add an episode
        episode_id = await memory.add(
            content="Test episode with embedding",
            episode_type="test"
        )

        # Check that embedding was generated
        mock_generate_embedding.assert_called_once()

        # Get the episode
        episode = memory.get(episode_id)

        # Check embedding
        self.assertIsNotNone(episode.embedding)
        self.assertEqual(len(episode.embedding), 4)
        self.assertEqual(episode.embedding, [0.1, 0.2, 0.3, 0.4])


class TestEpisode(unittest.TestCase):
    """Test the Episode class."""

    def test_creation(self):
        """Test episode creation."""
        episode = Episode(
            id="test-id",
            content="Test content",
            episode_type="test_type",
            timestamp=time.time(),
            metadata={"key": "value"},
            references=["ref1", "ref2"]
        )

        self.assertEqual(episode.id, "test-id")
        self.assertEqual(episode.content, "Test content")
        self.assertEqual(episode.episode_type, "test_type")
        self.assertEqual(episode.metadata["key"], "value")
        self.assertEqual(episode.references, ["ref1", "ref2"])

    def test_age_property(self):
        """Test age property."""
        # Create an episode with a timestamp 1 hour ago
        timestamp = time.time() - 3600
        episode = Episode(
            id="test-id",
            content="Test content",
            episode_type="test_type",
            timestamp=timestamp
        )

        # Check age is approximately 1 hour
        self.assertAlmostEqual(episode.age, 3600, delta=10)

    def test_formatted_date(self):
        """Test formatted date property."""
        # Create an episode with a specific timestamp
        timestamp = 1609459200  # 2021-01-01 00:00:00 UTC
        episode = Episode(
            id="test-id",
            content="Test content",
            episode_type="test_type",
            timestamp=timestamp
        )

        # Check formatted date
        self.assertTrue(len(episode.formatted_date) > 0)
        # We can't check the exact string as it depends on local timezone
        self.assertTrue("2021" in episode.formatted_date)

    def test_serialization(self):
        """Test episode serialization to dictionary."""
        episode = Episode(
            id="test-id",
            content="Test content",
            episode_type="test_type",
            timestamp=1609459200,
            metadata={"key": "value"},
            references=["ref1", "ref2"]
        )

        # Convert to dict
        episode_dict = episode.to_dict()

        # Check dict contents
        self.assertEqual(episode_dict["id"], "test-id")
        self.assertEqual(episode_dict["content"], "Test content")
        self.assertEqual(episode_dict["episode_type"], "test_type")
        self.assertEqual(episode_dict["timestamp"], 1609459200)
        self.assertEqual(episode_dict["metadata"]["key"], "value")
        self.assertEqual(episode_dict["references"], ["ref1", "ref2"])
        self.assertTrue("formatted_date" in episode_dict)

    def test_deserialization(self):
        """Test episode deserialization from dictionary."""
        episode_dict = {
            "id": "test-id",
            "content": "Test content",
            "episode_type": "test_type",
            "timestamp": 1609459200,
            "metadata": {"key": "value"},
            "references": ["ref1", "ref2"]
        }

        # Create from dict
        episode = Episode.from_dict(episode_dict)

        # Check episode properties
        self.assertEqual(episode.id, "test-id")
        self.assertEqual(episode.content, "Test content")
        self.assertEqual(episode.episode_type, "test_type")
        self.assertEqual(episode.timestamp, 1609459200)
        self.assertEqual(episode.metadata["key"], "value")
        self.assertEqual(episode.references, ["ref1", "ref2"])
        self.assertIsNone(episode.embedding)


if __name__ == "__main__":
    unittest.main()
