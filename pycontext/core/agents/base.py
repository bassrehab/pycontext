"""
Base Agent implementation for the PyContext framework.
"""
from typing import Dict, List, Optional, Any
import asyncio
import uuid
from abc import ABC, abstractmethod

from ..mcp.protocol import ContextType
from ..context.manager import ContextManager


class BaseAgent(ABC):
    """Base class for agents that use Model Context Protocol."""
    
    def __init__(
        self,
        agent_id: str = None,
        agent_role: str = "base_agent",
        context_manager: ContextManager = None
    ):
        """Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for the agent (optional)
            agent_role: Role of the agent
            context_manager: Context manager instance (optional)
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.role = agent_role
        self.context_manager = context_manager or ContextManager()
        self.session_id = None
    
    async def initialize_session(self) -> str:
        """Initialize a new context session.
        
        Returns:
            session_id: Unique identifier for the session
        """
        self.session_id = self.context_manager.create_session(self.agent_id)
        
        # Add system prompt as SYSTEM context
        system_prompt = await self._load_role_prompt()
        self.context_manager.add_context(
            session_id=self.session_id,
            content=system_prompt,
            context_type=ContextType.SYSTEM,
            relevance_score=1.0,  # System prompts always max relevance
            metadata={"type": "system_prompt", "role": self.role}
        )
        
        return self.session_id
    
    @abstractmethod
    async def _load_role_prompt(self) -> str:
        """Load role-specific prompt - implement in subclasses."""
        pass
    
    async def add_user_context(
        self,
        content: str,
        metadata: Dict = None
    ) -> str:
        """Add user input to context.
        
        Args:
            content: User input text
            metadata: Additional metadata
            
        Returns:
            block_id: Identifier for the added block
        """
        if self.session_id is None:
            await self.initialize_session()
        
        return self.context_manager.add_context(
            session_id=self.session_id,
            content=content,
            context_type=ContextType.USER,
            relevance_score=0.9,  # User context starts with high relevance
            metadata=metadata or {}
        )
    
    async def add_memory_context(
        self,
        content: str,
        relevance_score: float,
        metadata: Dict = None
    ) -> str:
        """Add memory (from episodic or semantic memory) to context.
        
        Args:
            content: Memory content
            relevance_score: Relevance score (0-1)
            metadata: Additional metadata
            
        Returns:
            block_id: Identifier for the added block
        """
        if self.session_id is None:
            await self.initialize_session()
        
        return self.context_manager.add_context(
            session_id=self.session_id,
            content=content,
            context_type=ContextType.MEMORY,
            relevance_score=relevance_score,
            metadata=metadata or {}
        )
    
    async def add_tool_context(
        self,
        content: str,
        tool_name: str,
        metadata: Dict = None
    ) -> str:
        """Add tool usage results to context.
        
        Args:
            content: Tool output content
            tool_name: Name of the tool
            metadata: Additional metadata
            
        Returns:
            block_id: Identifier for the added block
        """
        if self.session_id is None:
            await self.initialize_session()
        
        if metadata is None:
            metadata = {}
        metadata["tool_name"] = tool_name
        
        return self.context_manager.add_context(
            session_id=self.session_id,
            content=content,
            context_type=ContextType.TOOL,
            relevance_score=0.8,  # Tool outputs generally have high relevance
            metadata=metadata
        )
    
    @abstractmethod
    async def process(self, input_text: str) -> str:
        """Process input and generate a response - implement in subclasses."""
        pass
    
    def export_context(self) -> Dict:
        """Export current context for transfer to another agent.
        
        Returns:
            Dictionary representation of the current context
        """
        if self.session_id is None:
            raise ValueError("No active session")
        
        return self.context_manager.export_session(self.session_id)
    
    def import_context(self, context_data: Dict) -> None:
        """Import context from another agent.
        
        Args:
            context_data: Dictionary representation of a context
        """
        self.session_id = self.context_manager.import_session(context_data)
