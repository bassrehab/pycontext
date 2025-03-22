"""
Context Manager implementation for the PyContext framework.
"""
import uuid
from typing import Dict, List, Optional, Callable, Any

from ..mcp.protocol import (
    ContextType,
    ContextBlock,
    ContextPackage,
    ContextMetrics,
    create_context_block,
)


class ContextManager:
    """Manages context operations using Model Context Protocol."""
    
    def __init__(
        self,
        max_tokens: int = 8192,
        relevance_threshold: float = 0.2
    ):
        """Initialize the context manager.
        
        Args:
            max_tokens: Maximum tokens in context window
            relevance_threshold: Minimum relevance score for inclusion
        """
        self.max_tokens = max_tokens
        self.relevance_threshold = relevance_threshold
        self.sessions: Dict[str, ContextPackage] = {}
    
    def create_session(self, agent_id: str) -> str:
        """Create a new context session.
        
        Args:
            agent_id: Identifier for the agent
            
        Returns:
            session_id: Unique identifier for the session
        """
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = ContextPackage(
            session_id=session_id,
            agent_id=agent_id,
            blocks=[],
            metrics=ContextMetrics(
                total_tokens=0,
                context_saturation=0.0,
                type_distribution={}
            )
        )
        return session_id
    
    def add_context(
        self,
        session_id: str,
        content: str,
        context_type: ContextType,
        relevance_score: float = 1.0,
        metadata: Dict[str, str] = None
    ) -> str:
        """Add context to an existing session.
        
        Args:
            session_id: Session identifier
            content: Content text
            context_type: Type of context
            relevance_score: Relevance score (0-1)
            metadata: Additional metadata
            
        Returns:
            block_id: Identifier for the added block
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} does not exist")
        
        block_id = str(uuid.uuid4())
        block = ContextBlock(
            id=block_id,
            content=content,
            relevance_score=relevance_score,
            type=context_type,
            metadata=metadata or {}
        )
        
        self.sessions[session_id].blocks.append(block)
        self.sessions[session_id].calculate_metrics()
        
        # If we've exceeded context window, perform context pruning
        if self.sessions[session_id].metrics.context_saturation >= 0.9:
            self._prune_context(session_id)
        
        return block_id
    
    def _prune_context(self, session_id: str) -> None:
        """Prune least relevant context to fit within token limits.
        
        Args:
            session_id: Session identifier
        """
        session = self.sessions[session_id]
        
        # Don't prune SYSTEM context
        system_blocks = [b for b in session.blocks if b.type == ContextType.SYSTEM]
        other_blocks = [b for b in session.blocks if b.type != ContextType.SYSTEM]
        
        # Sort by relevance score (ascending)
        other_blocks.sort(key=lambda x: x.relevance_score)
        
        # Keep removing blocks until we're under target
        system_tokens = sum(b.token_count for b in system_blocks)
        target_tokens = int(self.max_tokens * 0.8) - system_tokens  # Target 80% usage
        current_tokens = sum(b.token_count for b in other_blocks)
        
        while current_tokens > target_tokens and other_blocks:
            removed_block = other_blocks.pop(0)  # Remove least relevant
            current_tokens -= removed_block.token_count
        
        # Reconstitute the blocks list
        session.blocks = system_blocks + other_blocks
        session.calculate_metrics()
    
    def get_formatted_context(
        self,
        session_id: str,
        formatter: Callable = None
    ) -> str:
        """Get formatted context for model input.
        
        Args:
            session_id: Session identifier
            formatter: Optional custom formatter function
            
        Returns:
            Formatted context string
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} does not exist")
        
        session = self.sessions[session_id]
        
        # Default formatter concatenates content with block type as separator
        if formatter is None:
            result = []
            for block in session.blocks:
                if block.relevance_score >= self.relevance_threshold:
                    result.append(f"[{block.type.name}]\n{block.content}")
            return "\n\n".join(result)
        
        return formatter(session)
    
    def export_session(self, session_id: str) -> Dict:
        """Export session as serializable dict.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary representation of the session
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} does not exist")
        
        return self.sessions[session_id].to_dict()
    
    def import_session(self, session_data: Dict) -> str:
        """Import a session from serialized data.
        
        Args:
            session_data: Dictionary representation of a session
            
        Returns:
            session_id: Session identifier
        """
        session_id = session_data["session_id"]
        
        blocks = []
        for block_data in session_data["blocks"]:
            blocks.append(ContextBlock(
                id=block_data["id"],
                content=block_data["content"],
                relevance_score=block_data["relevance_score"],
                type=ContextType[block_data["type"]],
                metadata=block_data["metadata"],
                timestamp=block_data["timestamp"],
                token_count=block_data["token_count"],
                references=block_data["references"]
            ))
        
        metrics = ContextMetrics(
            total_tokens=session_data["metrics"]["total_tokens"],
            context_saturation=session_data["metrics"]["context_saturation"],
            type_distribution=session_data["metrics"]["type_distribution"]
        )
        
        self.sessions[session_id] = ContextPackage(
            session_id=session_id,
            agent_id=session_data["agent_id"],
            blocks=blocks,
            metrics=metrics,
            version=session_data["version"],
            trace_id=session_data["trace_id"]
        )
        
        return session_id
