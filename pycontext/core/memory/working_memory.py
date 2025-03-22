"""
pycontext/core/memory/working_memory.py

Working Memory implementation for PyContext.
"""
from typing import Dict, List, Optional, Any
import time
from dataclasses import dataclass
import uuid


@dataclass
class MemoryItem:
    """A single item in working memory."""
    id: str
    content: Any
    memory_type: str
    timestamp: float
    ttl: Optional[float] = None  # Time-to-live in seconds
    metadata: Dict = None

    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_expired(self) -> bool:
        """Check if the memory item has expired."""
        if self.ttl is None:
            return False
        return time.time() > (self.timestamp + self.ttl)


class WorkingMemory:
    """
    Working memory implementation for short-term storage during agent interactions.
    """

    def __init__(self, capacity: int = 100, default_ttl: Optional[float] = 3600):
        """
        Initialize working memory.

        Args:
            capacity: Maximum number of items in memory
            default_ttl: Default time-to-live for items in seconds (None for no expiry)
        """
        self.items: Dict[str, MemoryItem] = {}
        self.capacity = capacity
        self.default_ttl = default_ttl

    def add(
            self,
            content: Any,
            memory_type: str,
            ttl: Optional[float] = None,
            metadata: Dict = None
    ) -> str:
        """
        Add an item to working memory.

        Args:
            content: Content to store
            memory_type: Type of memory item
            ttl: Time-to-live in seconds (None for default)
            metadata: Additional metadata

        Returns:
            ID of the memory item
        """
        # Clean expired items first
        self._clean_expired()

        # Check if we need to evict items
        if len(self.items) >= self.capacity:
            self._evict()

        # Use provided TTL or default
        if ttl is None:
            ttl = self.default_ttl

        # Create new memory item
        item_id = str(uuid.uuid4())
        item = MemoryItem(
            id=item_id,
            content=content,
            memory_type=memory_type,
            timestamp=time.time(),
            ttl=ttl,
            metadata=metadata or {}
        )

        # Add to memory
        self.items[item_id] = item

        return item_id

    def get(self, item_id: str) -> Optional[Any]:
        """
        Get an item from working memory.

        Args:
            item_id: ID of the memory item

        Returns:
            Content of the memory item, or None if not found or expired
        """
        item = self.items.get(item_id)

        if item is None or item.is_expired:
            # Remove if expired
            if item is not None and item.is_expired:
                del self.items[item_id]
            return None

        return item.content

    def get_by_type(self, memory_type: str) -> List[Any]:
        """
        Get all items of a specific type.

        Args:
            memory_type: Type of memory items to retrieve

        Returns:
            List of content of memory items
        """
        # Clean expired items first
        self._clean_expired()

        return [
            item.content for item in self.items.values()
            if item.memory_type == memory_type
        ]

    def get_by_metadata(self, key: str, value: Any) -> List[Any]:
        """
        Get all items with a specific metadata key-value pair.

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            List of content of matching memory items
        """
        # Clean expired items first
        self._clean_expired()

        return [
            item.content for item in self.items.values()
            if key in item.metadata and item.metadata[key] == value
        ]

    def update(
            self,
            item_id: str,
            content: Any = None,
            metadata: Dict = None,
            reset_ttl: bool = False
    ) -> bool:
        """
        Update an existing memory item.

        Args:
            item_id: ID of the memory item
            content: New content (None to keep existing)
            metadata: Metadata to update or add
            reset_ttl: Whether to reset the TTL

        Returns:
            Whether the update was successful
        """
        if item_id not in self.items or self.items[item_id].is_expired:
            # Remove if expired
            if item_id in self.items and self.items[item_id].is_expired:
                del self.items[item_id]
            return False

        item = self.items[item_id]

        # Update content if provided
        if content is not None:
            item.content = content

        # Update metadata if provided
        if metadata is not None:
            item.metadata.update(metadata)

        # Reset TTL if requested
        if reset_ttl and item.ttl is not None:
            item.timestamp = time.time()

        return True

    def remove(self, item_id: str) -> bool:
        """
        Remove an item from working memory.

        Args:
            item_id: ID of the memory item

        Returns:
            Whether the removal was successful
        """
        if item_id in self.items:
            del self.items[item_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all items from working memory."""
        self.items.clear()

    def _clean_expired(self) -> None:
        """Clean expired items from memory."""
        expired_ids = [
            item_id for item_id, item in self.items.items()
            if item.is_expired
        ]

        for item_id in expired_ids:
            del self.items[item_id]

    def _evict(self) -> None:
        """Evict items to make space when capacity is reached."""
        # Strategy: Remove oldest items first
        if not self.items:
            return

        # Sort by timestamp (oldest first)
        sorted_items = sorted(
            self.items.items(),
            key=lambda x: x[1].timestamp
        )

        # Remove oldest item
        oldest_id = sorted_items[0][0]
        del self.items[oldest_id]

    def get_recent(self, limit: int = 5) -> List[Any]:
        """
        Get the most recent memory items.

        Args:
            limit: Maximum number of items to return

        Returns:
            List of content of most recent memory items
        """
        # Clean expired items first
        self._clean_expired()

        # Sort by timestamp (newest first)
        sorted_items = sorted(
            self.items.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )

        # Return content of most recent items
        return [item.content for item in sorted_items[:limit]]

    def get_all_metadata(self) -> Dict[str, Dict]:
        """
        Get metadata for all items.

        Returns:
            Dictionary mapping item IDs to their metadata
        """
        return {
            item_id: item.metadata
            for item_id, item in self.items.items()
            if not item.is_expired
        }

    def search(self, query: str, metadata_filter: Dict = None) -> List[Any]:
        """
        Simple search through memory items.
        Note: This is a basic implementation - for production use,
        consider using a vector database or search engine.

        Args:
            query: Search query
            metadata_filter: Optional metadata filter

        Returns:
            List of content of matching memory items
        """
        # Clean expired items first
        self._clean_expired()

        query = query.lower()
        results = []

        for item in self.items.values():
            # Skip if content is not a string
            if not isinstance(item.content, str):
                continue

            # Check if query is in content
            if query in item.content.lower():
                # Check metadata filter if provided
                if metadata_filter:
                    match = True
                    for key, value in metadata_filter.items():
                        if key not in item.metadata or item.metadata[key] != value:
                            match = False
                            break

                    if not match:
                        continue

                results.append(item.content)

        return results