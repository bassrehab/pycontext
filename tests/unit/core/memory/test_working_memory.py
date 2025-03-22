"""
tests/unit/core/memory/test_working_memory.py

Tests for the Working Memory module.
"""
import unittest
import time
from pycontext.core.memory.working_memory import WorkingMemory, MemoryItem


class TestWorkingMemory(unittest.TestCase):
    """Test the Working Memory implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.memory = WorkingMemory(capacity=5, default_ttl=3600)

    def test_add_and_get(self):
        """Test adding and retrieving items."""
        # Add an item
        item_id = self.memory.add(
            content="Test content",
            memory_type="test_type",
            metadata={"key": "value"}
        )

        # Get the item
        content = self.memory.get(item_id)

        # Check
        self.assertEqual(content, "Test content")
        self.assertEqual(len(self.memory.items), 1)

    def test_get_by_type(self):
        """Test retrieving items by type."""
        # Add items of different types
        self.memory.add(
            content="Item 1",
            memory_type="type_a"
        )
        self.memory.add(
            content="Item 2",
            memory_type="type_a"
        )
        self.memory.add(
            content="Item 3",
            memory_type="type_b"
        )

        # Get items by type
        type_a_items = self.memory.get_by_type("type_a")
        type_b_items = self.memory.get_by_type("type_b")

        # Check
        self.assertEqual(len(type_a_items), 2)
        self.assertEqual(len(type_b_items), 1)
        self.assertIn("Item 1", type_a_items)
        self.assertIn("Item 2", type_a_items)
        self.assertIn("Item 3", type_b_items)

    def test_get_by_metadata(self):
        """Test retrieving items by metadata."""
        # Add items with different metadata
        self.memory.add(
            content="Item 1",
            memory_type="test_type",
            metadata={"user_id": "123", "category": "A"}
        )
        self.memory.add(
            content="Item 2",
            memory_type="test_type",
            metadata={"user_id": "123", "category": "B"}
        )
        self.memory.add(
            content="Item 3",
            memory_type="test_type",
            metadata={"user_id": "456", "category": "A"}
        )

        # Get items by metadata
        user_123_items = self.memory.get_by_metadata("user_id", "123")
        category_a_items = self.memory.get_by_metadata("category", "A")

        # Check
        self.assertEqual(len(user_123_items), 2)
        self.assertEqual(len(category_a_items), 2)
        self.assertIn("Item 1", user_123_items)
        self.assertIn("Item 2", user_123_items)
        self.assertIn("Item 1", category_a_items)
        self.assertIn("Item 3", category_a_items)

    def test_update(self):
        """Test updating items."""
        # Add an item
        item_id = self.memory.add(
            content="Original content",
            memory_type="test_type",
            metadata={"key": "value"}
        )

        # Update the item
        success = self.memory.update(
            item_id=item_id,
            content="Updated content",
            metadata={"new_key": "new_value"}
        )

        # Check
        self.assertTrue(success)
        content = self.memory.get(item_id)
        self.assertEqual(content, "Updated content")

        # Check metadata was updated (not replaced)
        item = self.memory.items[item_id]
        self.assertEqual(item.metadata["key"], "value")
        self.assertEqual(item.metadata["new_key"], "new_value")

    def test_remove(self):
        """Test removing items."""
        # Add an item
        item_id = self.memory.add(
            content="Test content",
            memory_type="test_type"
        )

        # Remove the item
        success = self.memory.remove(item_id)

        # Check
        self.assertTrue(success)
        self.assertIsNone(self.memory.get(item_id))
        self.assertEqual(len(self.memory.items), 0)

    def test_expiry(self):
        """Test item expiry."""
        # Add an item with a short TTL
        item_id = self.memory.add(
            content="Test content",
            memory_type="test_type",
            ttl=0.1  # 100ms TTL
        )

        # Check it exists
        self.assertIsNotNone(self.memory.get(item_id))

        # Wait for it to expire
        time.sleep(0.2)

        # Check it's gone
        self.assertIsNone(self.memory.get(item_id))

    def test_capacity(self):
        """Test capacity management."""
        # Add items up to capacity
        ids = []
        for i in range(5):  # Capacity is 5
            item_id = self.memory.add(
                content=f"Item {i + 1}",
                memory_type="test_type"
            )
            ids.append(item_id)

        # Check all items exist
        for item_id in ids:
            self.assertIsNotNone(self.memory.get(item_id))

        # Add one more item (should evict the oldest)
        new_id = self.memory.add(
            content="New item",
            memory_type="test_type"
        )

        # Check new item exists
        self.assertIsNotNone(self.memory.get(new_id))

        # Check oldest item was evicted
        self.assertIsNone(self.memory.get(ids[0]))

        # Check remaining items still exist
        for item_id in ids[1:]:
            self.assertIsNotNone(self.memory.get(item_id))

    def test_get_recent(self):
        """Test getting recent items."""
        # Add items with different timestamps
        for i in range(10):
            self.memory.add(
                content=f"Item {i + 1}",
                memory_type="test_type"
            )
            time.sleep(0.01)  # Small delay to ensure different timestamps

        # Get recent items
        recent = self.memory.get_recent(limit=3)

        # Check
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[0], "Item 10")
        self.assertEqual(recent[1], "Item 9")
        self.assertEqual(recent[2], "Item 8")

    def test_search(self):
        """Test searching through items."""
        # Add items with searchable content
        self.memory.add(
            content="This is about cats",
            memory_type="test_type",
            metadata={"category": "animals"}
        )
        self.memory.add(
            content="Dogs are great pets",
            memory_type="test_type",
            metadata={"category": "animals"}
        )
        self.memory.add(
            content="Python is a programming language",
            memory_type="test_type",
            metadata={"category": "tech"}
        )

        # Search without filter
        cat_results = self.memory.search("cat")

        # Search with filter
        animal_python_results = self.memory.search(
            "python",
            metadata_filter={"category": "animals"}
        )
        tech_python_results = self.memory.search(
            "python",
            metadata_filter={"category": "tech"}
        )

        # Check
        self.assertEqual(len(cat_results), 1)
        self.assertEqual(cat_results[0], "This is about cats")

        self.assertEqual(len(animal_python_results), 0)
        self.assertEqual(len(tech_python_results), 1)
        self.assertEqual(tech_python_results[0], "Python is a programming language")


if __name__ == "__main__":
    unittest.main()
