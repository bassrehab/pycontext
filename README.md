# PyContext

A comprehensive Python framework for building autonomous multi-agent systems with standardized context management.

## Overview

PyContext implements the Model Context Protocol (MCP) for efficient context handling between agents, providing a robust foundation for building complex multi-agent systems.

Key features:
- **Model Context Protocol (MCP)**: Standardized context exchange between agents
- **Token Optimization**: Accurate token counting and context window management
- **Agent Framework**: Extensible agent architecture with specialized roles
- **Memory Systems**: Hierarchical memory with different retention patterns
- **LLM Integrations**: Ready-to-use integrations with popular LLM providers

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pycontext.git
cd pycontext

# Install the package in development mode
pip install -e .

# Install extra dependencies
pip install -e ".[dev,openai]"
```

## Quick Start

Here's a simple example of using PyContext to create a conversational agent:

```python
import asyncio
import os
from pycontext.core.context.manager import ContextManager
from pycontext.integrations.llm_providers.openai_provider import OpenAIProvider, OpenAIAgent

async def main():
    # Create an OpenAI provider
    provider = OpenAIProvider(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-3.5-turbo"
    )
    
    # Create an agent
    agent = OpenAIAgent(
        agent_id="assistant",
        agent_role="helpful assistant",
        openai_provider=provider
    )
    
    # Initialize the agent
    await agent.initialize_session()
    
    # Process a query
    response = await agent.process("Hello, what can you help me with today?")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

## Core Components

### Model Context Protocol (MCP)

MCP provides a standardized way to manage and exchange context between agents:

```python
from pycontext.core.mcp.protocol import create_context_block, ContextType

# Create a context block
block = create_context_block(
    content="This is important information",
    context_type=ContextType.KNOWLEDGE,
    relevance_score=0.9,
    model_name="gpt-4"
)
```

### Context Manager

The Context Manager handles context operations including pruning, formatting, and window management:

```python
from pycontext.core.context.manager import ContextManager

# Create a context manager
manager = ContextManager(max_tokens=4096)

# Create a session
session_id = manager.create_session("my_agent")

# Add context
manager.add_context(
    session_id=session_id,
    content="Hello, how can I help you?",
    context_type=ContextType.USER,
    relevance_score=0.9
)

# Get formatted context
formatted = manager.get_formatted_context(session_id)
```

### Agents

PyContext provides various agent implementations:

```python
from pycontext.core.agents.intent_agent import IntentAgent

# Create an intent analysis agent
intent_agent = IntentAgent(llm_client=provider)

# Process a query
result = await intent_agent.process("I need help with my internet connection")
```

### Working Memory

The Working Memory system provides short-term storage during agent interactions:

```python
from pycontext.core.memory.working_memory import WorkingMemory

# Create working memory
memory = WorkingMemory(capacity=100)

# Add items
item_id = memory.add(
    content="Customer info",
    memory_type="customer_data",
    metadata={"customer_id": "12345"}
)

# Retrieve items
data = memory.get(item_id)
```

## Examples

Check out the examples directory for complete examples:

- `examples/simple_chatbot/`: Basic conversational agent
- `examples/customer_service/`: Customer service scenario with intent recognition

## Development Status

PyContext is currently in alpha stage. The API may change before the first stable release.

## License

MIT