"""
Core Model Context Protocol (MCP) definitions.
"""
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass
import time
import uuid


class ContextType(Enum):
    """Types of context blocks in the MCP."""
    SYSTEM = 0  # System prompts and instructions
    USER = 1    # User inputs
    AGENT = 2   # Agent outputs/responses
    MEMORY = 3  # Retrieved memories
    KNOWLEDGE = 4  # Knowledge base information
    TOOL = 5    # Tool usage outputs


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
        if self.token_count is None:
            # Approximate token count based on whitespace splitting
            # In production, use a proper tokenizer
            self.token_count = len(self.content.split())


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
    
    def __post_init__(self):
        """Initialize default values after initialization."""
        if self.trace_id is None:
            self.trace_id = str(uuid.uuid4())
    
    def calculate_metrics(self) -> None:
        """Calculate metrics based on current context blocks."""
        total_tokens = sum(block.token_count for block in self.blocks)
        # Assuming 8K context window
        context_saturation = min(1.0, total_tokens / 8192)
        
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
            "trace_id": self.trace_id
        }


def create_context_block(
    content: str,
    context_type: ContextType,
    relevance_score: float = 1.0,
    metadata: Dict[str, str] = None
) -> ContextBlock:
    """Helper function to create a context block."""
    return ContextBlock(
        id=str(uuid.uuid4()),
        content=content,
        relevance_score=relevance_score,
        type=context_type,
        metadata=metadata or {},
    )
