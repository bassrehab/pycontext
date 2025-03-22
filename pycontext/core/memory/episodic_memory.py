"""
pycontext/core/memory/episodic_memory.py

Episodic Memory implementation for PyContext.
"""
from typing import Dict, List, Optional, Any, Tuple
import time
import json
import uuid
from dataclasses import dataclass, field
import datetime


@dataclass
class Episode:
    """
    An episode in episodic memory, representing a complete interaction or event.
    """
    id: str
    content: Any
    episode_type: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None

    @property
    def age(self) -> float:
        """Get the age of the episode in seconds."""
        return time.time() - self.timestamp

    @property
    def formatted_date(self) -> str:
        """Get the formatted date of the episode."""
        dt = datetime.datetime.fromtimestamp(self.timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "episode_type": self.episode_type,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "references": self.references,
            "formatted_date": self.formatted_date
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Episode':
        """Create an episode from a dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            episode_type=data["episode_type"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
            references=data.get("references", []),
            embedding=data.get("embedding")
        )


class EpisodicMemory:
    """
    Episodic memory for storing and retrieving complete episodes or interactions.
    Unlike working memory, episodic memory is designed for longer-term storage
    and retrieval of complete interactions.
    """

    def __init__(
        self,
        use_embeddings: bool = False,
        embedding_provider: Any = None,
        max_episodes: int = 1000
    ):
        """
        Initialize the episodic memory.

        Args:
            use_embeddings: Whether to use embeddings for similarity search
            embedding_provider: Provider for generating embeddings
            max_episodes: Maximum number of episodes to store
        """
        self.episodes: Dict[str, Episode] = {}
        self.use_embeddings = use_embeddings
        self.embedding_provider = embedding_provider
        self.max_episodes = max_episodes
        self.episode_types: Dict[str, List[str]] = {}  # Type -> list of episode IDs

    async def add(
        self,
        content: Any,
        episode_type: str,
        metadata: Dict[str, Any] = None,
        references: List[str] = None
    ) -> str:
        """
        Add an episode to memory.

        Args:
            content: Content of the episode (can be any serializable type)
            episode_type: Type of episode
            metadata: Additional metadata
            references: References to other episodes

        Returns:
            ID of the episode
        """
        # Check if we need to make room
        if len(self.episodes) >= self.max_episodes:
            self._prune_oldest()

        episode_id = str(uuid.uuid4())

        # Create embedding if needed
        embedding = None
        if self.use_embeddings and self.embedding_provider:
            if isinstance(content, str):
                try:
                    embedding = await self._generate_embedding(content)
                except Exception as e:
                    print(f"Error generating embedding: {e}")
            elif isinstance(content, dict) and "text" in content:
                try:
                    embedding = await self._generate_embedding(content["text"])
                except Exception as e:
                    print(f"Error generating embedding: {e}")

        # Create episode
        episode = Episode(
            id=episode_id,
            content=content,
            episode_type=episode_type,
            timestamp=time.time(),
            metadata=metadata or {},
            references=references or [],
            embedding=embedding
        )

        # Store episode
        self.episodes[episode_id] = episode

        # Update episode type index
        if episode_type not in self.episode_types:
            self.episode_types[episode_type] = []
        self.episode_types[episode_type].append(episode_id)

        return episode_id

    def get(self, episode_id: str) -> Optional[Episode]:
        """
        Get an episode by ID.

        Args:
            episode_id: Episode ID

        Returns:
            Episode if found, None otherwise
        """
        return self.episodes.get(episode_id)

    def get_by_type(self, episode_type: str, limit: int = None) -> List[Episode]:
        """
        Get episodes by type.

        Args:
            episode_type: Type of episodes to retrieve
            limit: Maximum number of episodes to return

        Returns:
            List of episodes
        """
        if episode_type not in self.episode_types:
            return []

        episode_ids = self.episode_types[episode_type]
        if limit is not None:
            episode_ids = episode_ids[-limit:]  # Get most recent episodes

        return [self.episodes[episode_id] for episode_id in episode_ids]

    def get_by_reference(self, reference_id: str) -> List[Episode]:
        """
        Get episodes that reference a specific episode.

        Args:
            reference_id: ID of the episode to find references to

        Returns:
            List of episodes
        """
        return [
            episode for episode in self.episodes.values()
            if reference_id in episode.references
        ]

    def get_by_metadata(self, key: str, value: Any) -> List[Episode]:
        """
        Get episodes by metadata key-value pair.

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            List of episodes
        """
        return [
            episode for episode in self.episodes.values()
            if key in episode.metadata and episode.metadata[key] == value
        ]

    def get_recent(self, limit: int = 10) -> List[Episode]:
        """
        Get the most recent episodes.

        Args:
            limit: Maximum number of episodes to return

        Returns:
            List of episodes
        """
        sorted_episodes = sorted(
            self.episodes.values(),
            key=lambda e: e.timestamp,
            reverse=True
        )

        return sorted_episodes[:limit]

    def search(self, query: str, limit: int = 5) -> List[Tuple[Episode, float]]:
        """
        Search for episodes similar to the query.

        Args:
            query: Search query
            limit: Maximum number of episodes to return

        Returns:
            List of (episode, similarity_score) tuples
        """
        if self.use_embeddings and self.embedding_provider:
            return self._search_with_embeddings(query, limit)
        else:
            return self._search_with_text(query, limit)

    def remove(self, episode_id: str) -> bool:
        """
        Remove an episode from memory.

        Args:
            episode_id: Episode ID

        Returns:
            Whether the episode was removed
        """
        if episode_id not in self.episodes:
            return False

        episode = self.episodes[episode_id]

        # Remove from type index
        if episode.episode_type in self.episode_types:
            if episode_id in self.episode_types[episode.episode_type]:
                self.episode_types[episode.episode_type].remove(episode_id)

        # Remove episode
        del self.episodes[episode_id]

        return True

    def clear(self) -> None:
        """Clear all episodes from memory."""
        self.episodes.clear()
        self.episode_types.clear()

    def save_to_file(self, filename: str) -> None:
        """
        Save episodes to a file.

        Args:
            filename: Path to the file
        """
        # Convert episodes to dictionaries
        data = {
            episode_id: episode.to_dict()
            for episode_id, episode in self.episodes.items()
        }

        # Save to file
        with open(filename, 'w') as file:
            json.dump(data, file)

    def load_from_file(self, filename: str) -> None:
        """
        Load episodes from a file.

        Args:
            filename: Path to the file
        """
        # Clear current episodes
        self.clear()

        # Load from file
        with open(filename, 'r') as file:
            data = json.load(file)

        # Convert dictionaries to episodes
        for episode_id, episode_dict in data.items():
            episode = Episode.from_dict(episode_dict)
            self.episodes[episode_id] = episode

            # Update type index
            if episode.episode_type not in self.episode_types:
                self.episode_types[episode.episode_type] = []
            self.episode_types[episode.episode_type].append(episode_id)

    def _prune_oldest(self) -> None:
        """Remove the oldest episodes to make room for new ones."""
        # Sort episodes by timestamp (oldest first)
        sorted_episodes = sorted(
            self.episodes.items(),
            key=lambda item: item[1].timestamp
        )

        # Determine how many to remove (10% of max_episodes or at least 1)
        num_to_remove = max(1, int(self.max_episodes * 0.1))

        # Remove oldest episodes
        for i in range(min(num_to_remove, len(sorted_episodes))):
            episode_id, episode = sorted_episodes[i]
            self.remove(episode_id)

    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if not self.embedding_provider:
            raise ValueError("No embedding provider available")

        return await self.embedding_provider.embed(text)

    def _search_with_embeddings(self, query: str, limit: int = 5) -> List[Tuple[Episode, float]]:
        """
        Search for episodes using embeddings.

        Args:
            query: Search query
            limit: Maximum number of episodes to return

        Returns:
            List of (episode, similarity_score) tuples
        """
        try:
            # Generate query embedding
            query_embedding = asyncio.run(self._generate_embedding(query))

            # Calculate similarity with all episodes
            similarities = []

            for episode in self.episodes.values():
                if episode.embedding is None:
                    continue

                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, episode.embedding)
                similarities.append((episode, similarity))

            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)

            return similarities[:limit]

        except Exception as e:
            print(f"Error in embedding search: {e}")
            # Fall back to text search
            return self._search_with_text(query, limit)

    def _search_with_text(self, query: str, limit: int = 5) -> List[Tuple[Episode, float]]:
        """
        Search for episodes using simple text matching.

        Args:
            query: Search query
            limit: Maximum number of episodes to return

        Returns:
            List of (episode, similarity_score) tuples
        """
        query = query.lower()
        results = []

        for episode in self.episodes.values():
            # Skip if content is not a string or dict
            if isinstance(episode.content, str):
                content = episode.content.lower()
                if query in content:
                    # Crude similarity score based on query length vs content length
                    score = len(query) / len(content) if len(content) > 0 else 0
                    results.append((episode, score))
            elif isinstance(episode.content, dict) and "text" in episode.content:
                content = episode.content["text"].lower()
                if query in content:
                    score = len(query) / len(content) if len(content) > 0 else 0
                    results.append((episode, score))

        # Sort by score (highest first)
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:limit]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0

        return dot_product / (magnitude1 * magnitude2)