"""
Basic chatbot example using PyContext.
"""
import asyncio
import json
from typing import Dict, List

from pycontext.core.agents.base import BaseAgent
from pycontext.core.context.manager import ContextManager
from pycontext.core.mcp.protocol import ContextType


class SimpleChatBot(BaseAgent):
    """A simple chatbot implementation using PyContext."""
    
    def __init__(self, name: str = "ChatBot"):
        """Initialize the chatbot.
        
        Args:
            name: Name of the chatbot
        """
        super().__init__(agent_role=f"chat_agent:{name}")
        self.name = name
    
    async def _load_role_prompt(self) -> str:
        """Load the chatbot's system prompt."""
        return f"""You are {self.name}, a helpful assistant.
Your role is to provide friendly, concise, and accurate responses to user queries.
Always maintain a conversational tone and prioritize clarity in your responses."""
    
    async def process(self, input_text: str) -> str:
        """Process user input and generate a response.
        
        Args:
            input_text: User's message
            
        Returns:
            Chatbot's response
        """
        # Add user input to context
        await self.add_user_context(input_text)
        
        # This is where you would call an LLM in a real implementation
        # For this example, we'll simulate a response
        response = f"This is a simulated response from {self.name} to: {input_text}"
        
        # Add agent response to context
        self.context_manager.add_context(
            session_id=self.session_id,
            content=response,
            context_type=ContextType.AGENT,
            relevance_score=0.9,
            metadata={"type": "agent_response", "agent_name": self.name}
        )
        
        return response


async def main():
    """Run the chatbot example."""
    chatbot = SimpleChatBot(name="PyContext Assistant")
    
    # Initialize session
    await chatbot.initialize_session()
    
    # Process a few messages
    messages = [
        "Hello, how are you?",
        "What can you help me with?",
        "Tell me about PyContext framework."
    ]
    
    for msg in messages:
        print(f"\nUser: {msg}")
        response = await chatbot.process(msg)
        print(f"{chatbot.name}: {response}")
    
    # Export context and print it
    context_data = chatbot.export_context()
    print("\nContext data:")
    print(json.dumps(context_data, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
