"""
pycontext/core/memory/embedding_provider.py

Embedding provider interface and implementations for PyContext memory systems.
"""
from typing import List, Dict, Any, Optional
import abc
import numpy as np


class EmbeddingProvider(abc.ABC):
    """Abstract base class for embedding providers."""

    @abc.abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        pass


class SimpleEmbeddingProvider(EmbeddingProvider):
    """
    A simple embedding provider that uses basic TF-IDF for embeddings.
    This is a fallback when no external embedding provider is available.
    Not suitable for production use - provides only very basic similarity matching.
    """

    def __init__(self, dimension: int = 100):
        """
        Initialize the simple embedding provider.

        Args:
            dimension: Dimension of the embedding vectors
        """
        self.dimension = dimension
        self.vocab: Dict[str, int] = {}  # Word -> index
        self.idf: Dict[str, float] = {}  # Word -> IDF
        self.documents: List[List[str]] = []  # List of tokenized documents

    async def embed(self, text: str) -> List[float]:
        """
        Generate a simple TF-IDF embedding for the text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Tokenize
        tokens = self._tokenize(text)

        # Update vocabulary and documents
        self._update_vocab(tokens)
        self.documents.append(tokens)

        # Update IDF values
        self._update_idf()

        # Create TF-IDF vector
        vector = self._create_tfidf_vector(tokens)

        # Reduce dimension if needed
        if len(vector) > self.dimension:
            vector = self._reduce_dimension(vector)
        elif len(vector) < self.dimension:
            # Pad with zeros
            vector = vector + [0.0] * (self.dimension - len(vector))

        return vector

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Very simple tokenization by lowercase and split
        text = text.lower()
        # Remove punctuation
        for char in ",.!?;:()[]{}\"'":
            text = text.replace(char, " ")
        return text.split()

    def _update_vocab(self, tokens: List[str]) -> None:
        """
        Update vocabulary with new tokens.

        Args:
            tokens: New tokens
        """
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def _update_idf(self) -> None:
        """Update IDF values for all words in vocabulary."""
        num_docs = len(self.documents)

        if num_docs == 0:
            return

        for word in self.vocab:
            # Count documents containing this word
            doc_count = sum(1 for doc in self.documents if word in doc)
            # Calculate IDF
            self.idf[word] = np.log(num_docs / (1 + doc_count))

    def _create_tfidf_vector(self, tokens: List[str]) -> List[float]:
        """
        Create TF-IDF vector for tokens.

        Args:
            tokens: Tokens to vectorize

        Returns:
            TF-IDF vector
        """
        # Count token frequencies
        tf = {}
        for token in tokens:
            if token not in tf:
                tf[token] = 0
            tf[token] += 1

        # Normalize TF
        doc_length = len(tokens)
        if doc_length > 0:
            for token in tf:
                tf[token] /= doc_length

        # Create vector
        vector = [0.0] * len(self.vocab)

        for token, freq in tf.items():
            if token in self.vocab and token in self.idf:
                index = self.vocab[token]
                vector[index] = freq * self.idf[token]

        return vector

    def _reduce_dimension(self, vector: List[float]) -> List[float]:
        """
        Reduce vector dimension using a simple averaging technique.

        Args:
            vector: Vector to reduce

        Returns:
            Reduced vector
        """
        if len(vector) <= self.dimension:
            return vector

        # Convert to numpy array for easier manipulation
        arr = np.array(vector)

        # Split into chunks
        chunks = np.array_split(arr, self.dimension)

        # Average each chunk
        reduced = [float(chunk.mean()) for chunk in chunks]

        return reduced


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider that uses OpenAI's embedding API."""

    def __init__(
            self,
            api_key: str,
            model: str = "text-embedding-ada-002"
    ):
        """
        Initialize the OpenAI embedding provider.

        Args:
            api_key: OpenAI API key
            model: Model to use for embeddings
        """
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install it with: pip install openai"
            )

        self.model = model

    async def embed(self, text: str) -> List[float]:
        """
        Generate an embedding using OpenAI's API.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        response = await self.client.embeddings.create(
            model=self.model,
            input=text
        )

        return response.data[0].embedding