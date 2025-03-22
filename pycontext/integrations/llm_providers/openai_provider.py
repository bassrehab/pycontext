"""
pycontext/integrations/llm_providers/openai_provider.py

OpenAI integration for PyContext.
"""
from typing import Dict, List, Optional, Any, Union
import asyncio
import json

from ...core.agents.base import BaseAgent
from ...core.mcp.protocol import ContextType


class OpenAIProvider:
    """
    OpenAI API integration for PyContext.
    """

    def __init__(
            self,
            api_key: str,
            model: str = "gpt-4",
            temperature: float = 0.7,
            max_tokens: int = 1000
    ):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install it with: pip install openai"
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate(
            self,
            messages: List[Dict[str, str]],
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text using the OpenAI API.

        Args:
            messages: List of message dictionaries
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Returns:
            Generated text
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens
        )

        return response.choices[0].message.content

    def format_context(self, context_str: str) -> List[Dict[str, str]]:
        """
        Format context string into OpenAI message format.

        Args:
            context_str: Context string

        Returns:
            Formatted messages for OpenAI API
        """
        # Split by context type markers
        sections = context_str.split("\n\n")
        messages = []

        for section in sections:
            if not section.strip():
                continue

            # Extract type and content
            parts = section.split("\n", 1)
            if len(parts) != 2:
                continue

            type_marker = parts[0].strip("[]")
            content = parts[1].strip()

            if type_marker == "SYSTEM":
                messages.append({"role": "system", "content": content})
            elif type_marker == "USER":
                messages.append({"role": "user", "content": content})
            elif type_marker == "AGENT":
                messages.append({"role": "assistant", "content": content})
            elif type_marker in ["MEMORY", "KNOWLEDGE", "TOOL"]:
                # Add as system message with type prefix
                messages.append({
                    "role": "system",
                    "content": f"[{type_marker}]\n{content}"
                })

        return messages


class OpenAIAgent(BaseAgent):
    """
    Agent implementation using OpenAI.
    """

    def __init__(
            self,
            agent_id: str = None,
            agent_role: str = "openai_agent",
            openai_provider: OpenAIProvider = None,
            api_key: str = None,
            model: str = "gpt-4"
    ):
        """
        Initialize the OpenAI agent.

        Args:
            agent_id: Optional unique identifier
            agent_role: Role of the agent
            openai_provider: OpenAI provider instance
            api_key: OpenAI API key (if provider not provided)
            model: OpenAI model to use (if provider not provided)
        """
        super().__init__(
            agent_id=agent_id,
            agent_role=agent_role
        )

        if openai_provider is None and api_key is not None:
            openai_provider = OpenAIProvider(
                api_key=api_key,
                model=model
            )

        if openai_provider is None:
            raise ValueError("Either openai_provider or api_key must be provided")

        self.provider = openai_provider

    async def _load_role_prompt(self) -> str:
        """Default system prompt for OpenAI agent."""
        return f"You are an AI assistant named {self.agent_id or 'Assistant'}. " \
               f"Your role is {self.role}. " \
               f"Provide helpful, accurate, and concise responses."

    async def process(self, input_text: str) -> str:
        """
        Process user input and generate a response.

        Args:
            input_text: User's message

        Returns:
            Generated response
        """
        # Add user input to context
        await self.add_user_context(input_text)

        # Get formatted context
        context_str = self.context_manager.get_formatted_context(self.session_id)

        # Format for OpenAI
        messages = self.provider.format_context(context_str)

        # Generate response
        response = await self.provider.generate(messages)

        # Add response to context
        self.context_manager.add_context(
            session_id=self.session_id,
            content=response,
            context_type=ContextType.AGENT,
            relevance_score=0.9,
            metadata={"type": "agent_response", "model": self.provider.model}
        )

        return response
    