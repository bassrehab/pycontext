"""
pycontext/integrations/llm_providers/anthropic_provider.py

Anthropic Claude integration for PyContext.
"""
from typing import Dict, List, Optional, Any, Union
import asyncio
import json

from ...core.agents.base import BaseAgent
from ...core.mcp.protocol import ContextType


class AnthropicProvider:
    """
    Anthropic Claude API integration for PyContext.
    """

    def __init__(
            self,
            api_key: str,
            model: str = "claude-3-opus-20240229",
            temperature: float = 0.7,
            max_tokens: int = 1000
    ):
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key
            model: Model to use (e.g., "claude-3-opus-20240229")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Install it with: pip install anthropic"
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate(
            self,
            messages: List[Dict[str, str]],
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            system: Optional[str] = None
    ) -> str:
        """
        Generate text using the Anthropic Claude API.

        Args:
            messages: List of message dictionaries
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            system: Optional system prompt

        Returns:
            Generated text
        """
        # Convert messages to Anthropic format
        anthropic_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                anthropic_messages.append({
                    "role": "user",
                    "content": content
                })
            elif role == "assistant":
                anthropic_messages.append({
                    "role": "assistant",
                    "content": content
                })

        # Create the completion
        response = await self.client.messages.create(
            model=self.model,
            messages=anthropic_messages,
            system=system,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens
        )

        return response.content[0].text

    def format_context(self, context_str: str) -> Dict[str, Any]:
        """
        Format context string into Anthropic message format and system prompt.

        Args:
            context_str: Context string

        Returns:
            Dictionary with 'messages' and 'system' fields
        """
        # Split by context type markers
        sections = context_str.split("\n\n")
        messages = []
        system_prompts = []

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
                # Add to system prompts
                system_prompts.append(content)
            elif type_marker == "USER":
                messages.append({"role": "user", "content": content})
            elif type_marker == "AGENT":
                messages.append({"role": "assistant", "content": content})
            elif type_marker in ["MEMORY", "KNOWLEDGE", "TOOL"]:
                # Add as context in the most recent user message or create a new one
                prefixed_content = f"[{type_marker}]\n{content}"
                if messages and messages[-1]["role"] == "user":
                    messages[-1]["content"] += f"\n\n{prefixed_content}"
                else:
                    messages.append({"role": "user", "content": prefixed_content})

        # Combine system prompts
        system = "\n\n".join(system_prompts) if system_prompts else None

        return {
            "messages": messages,
            "system": system
        }


class AnthropicAgent(BaseAgent):
    """
    Agent implementation using Anthropic Claude.
    """

    def __init__(
            self,
            agent_id: str = None,
            agent_role: str = "anthropic_agent",
            anthropic_provider: AnthropicProvider = None,
            api_key: str = None,
            model: str = "claude-3-opus-20240229"
    ):
        """
        Initialize the Anthropic Claude agent.

        Args:
            agent_id: Optional unique identifier
            agent_role: Role of the agent
            anthropic_provider: Anthropic provider instance
            api_key: Anthropic API key (if provider not provided)
            model: Anthropic model to use (if provider not provided)
        """
        super().__init__(
            agent_id=agent_id,
            agent_role=agent_role
        )

        if anthropic_provider is None and api_key is not None:
            anthropic_provider = AnthropicProvider(
                api_key=api_key,
                model=model
            )

        if anthropic_provider is None:
            raise ValueError("Either anthropic_provider or api_key must be provided")

        self.provider = anthropic_provider

    async def _load_role_prompt(self) -> str:
        """Default system prompt for Anthropic agent."""
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

        # Format for Anthropic
        context_data = self.provider.format_context(context_str)

        # Generate response
        response = await self.provider.generate(
            messages=context_data["messages"],
            system=context_data["system"]
        )

        # Add response to context
        self.context_manager.add_context(
            session_id=self.session_id,
            content=response,
            context_type=ContextType.AGENT,
            relevance_score=0.9,
            metadata={"type": "agent_response", "model": self.provider.model}
        )

        return response
