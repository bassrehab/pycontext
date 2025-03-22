"""
pycontext/core/mcp/protocol.py

Core Model Context Protocol (MCP) definitions.
"""
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass
import time
import uuid

from .tokenization import count_tokens


class ContextType(Enum):
    """Types of context blocks in the MCP."""
    SYSTEM = 0  # System prompts and instructions
    USER = 1  # User inputs
    AGENT = 2  # Agent outputs/responses
    MEMORY = 3  # Retrieved memories
    KNOWLEDGE = 4  # Knowledge base information
    TOOL = 5  # Tool usage outputs


@dataclass
class ContextBlock:
    """Represents a single block of context in the MCP."""
    id: str
    content: str
    relevance_score: float
    type: ContextType
    metadata: Dict[str, str] = None
    timestamp: int = None
    token_count: int = None
    references: List[str] = None

    def __post_init__(self):
        """Initialize default values after initialization."""
        if self.timestamp is None:
            self.timestamp = int(time.time() * 1000)
        if self.metadata is None:
            self.metadata = {}
        if self.references is None:
            self.references = []

        # Use tokenization utilities to count tokens
        model_name = self.metadata.get("model") if self.metadata else None
        if self.token_count is None:
            self.token_count = count_tokens(self.content, model_name)


@dataclass
class ContextMetrics:
    """Metrics about a context package."""
    total_tokens: int
    context_saturation: float
    type_distribution: Dict[str, float]


@dataclass
class ContextPackage:
    """A complete package of context blocks with metadata."""
    session_id: str
    agent_id: str
    blocks: List[ContextBlock]
    metrics: ContextMetrics
    version: int = 1
    trace_id: str = None
    max_tokens: int = 8192  # Default context window size

    def __post_init__(self):
        """Initialize default values after initialization."""
        if self.trace_id is None:
            self.trace_id = str(uuid.uuid4())

    def calculate_metrics(self) -> None:
        """Calculate metrics based on current context blocks."""
        total_tokens = sum(block.token_count for block in self.blocks)

        # Calculate context saturation based on max_tokens
        context_saturation = min(1.0, total_tokens / self.max_tokens)

        # Calculate distribution of context types
        type_counts = {}
        for block in self.blocks:
            type_name = block.type.name
            if type_name not in type_counts:
                type_counts[type_name] = 0
            type_counts[type_name] += block.token_count

        type_distribution = {
            k: v / total_tokens if total_tokens > 0 else 0
            for k, v in type_counts.items()
        }

        self.metrics = ContextMetrics(
            total_tokens=total_tokens,
            context_saturation=context_saturation,
            type_distribution=type_distribution
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary representation for serialization."""
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "blocks": [
                {
                    "id": b.id,
                    "content": b.content,
                    "relevance_score": b.relevance_score,
                    "type": b.type.name,
                    "metadata": b.metadata,
                    "timestamp": b.timestamp,
                    "token_count": b.token_count,
                    "references": b.references
                }
                for b in self.blocks
            ],
            "metrics": {
                "total_tokens": self.metrics.total_tokens,
                "context_saturation": self.metrics.context_saturation,
                "type_distribution": self.metrics.type_distribution
            },
            "version": self.version,
            "trace_id": self.trace_id,
            "max_tokens": self.max_tokens
        }


def create_context_block(
        content: str,
        context_type: ContextType,
        relevance_score: float = 1.0,
        metadata: Dict[str, str] = None,
        model_name: Optional[str] = None
) -> ContextBlock:
    """
    Helper function to create a context block with accurate token counting.

    Args:
        content: The text content
        context_type: Type of context
        relevance_score: Relevance score (0-1)
        metadata: Additional metadata
        model_name: Optional model name for tokenization

    Returns:
        A new ContextBlock instance
    """
    if metadata is None:
        metadata = {}

    if model_name:
        metadata["model"] = model_name

    # Count tokens using the appropriate tokenizer
    token_count = count_tokens(content, model_name)

    return ContextBlock(
        id=str(uuid.uuid4()),
        content=content,
        relevance_score=relevance_score,
        type=context_type,
        metadata=metadata,
        token_count=token_count
    )


def compress_context_block(
        block: ContextBlock,
        compression_ratio: float = 0.7,
        model_name: Optional[str] = None
) -> ContextBlock:
    """
    Create a compressed version of a context block.

    Args:
        block: Original block to compress
        compression_ratio: Target compression ratio (0-1)
        model_name: Optional model name for tokenization

    Returns:
        A new compressed ContextBlock
    """
    # For now, implement a simple truncation-based compression
    # In a full implementation, this could use summarization techniques

    if block.token_count <= 10:  # Don't compress very small blocks
        return block

    target_tokens = int(block.token_count * compression_ratio)

    # Simple compression by truncation and adding an indicator
    tokens = count_tokens(block.content, model_name)
    if len(tokens) <= target_tokens:
        return block

    # Keep first 2/3 and last 1/3 of the target tokens
    first_part = int(target_tokens * 0.67)
    last_part = target_tokens - first_part

    if last_part > 0:
        content = (
                " ".join(tokens[:first_part]) +
                " [...content compressed...] " +
                " ".join(tokens[-last_part:])
        )
    else:
        content = " ".join(tokens[:target_tokens]) + " [...content truncated...]"

    # Create new metadata noting compression
    metadata = block.metadata.copy() if block.metadata else {}
    metadata["compressed"] = "true"
    metadata["original_tokens"] = str(block.token_count)

    return ContextBlock(
        id=block.id,
        content=content,
        relevance_score=block.relevance_score,
        type=block.type,
        metadata=metadata,
        timestamp=block.timestamp,
        references=block.references.copy() if block.references else []
    )