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
- **Procedural Workflows**: Step-by-step procedures with dependency management and error handling
- **Agent Coordination**: Sophisticated multi-agent collaboration

## Installation

```bash
# Clone the repository
git clone https://github.com/bassrehab/pycontext.git
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

PyContext provides various specialized agent implementations:

- **BaseAgent**: Core agent functionality
- **IntentAgent**: Analyzes user intent
- **KnowledgeAgent**: Retrieves and synthesizes knowledge
- **TechnicalAgent**: Diagnoses and solves technical issues
- **DialogAgent**: Manages conversational flow
- **ProceduralAgent**: Executes multi-step procedures

```python
from pycontext.core.agents.procedural_agent import ProceduralAgent

# Create a procedural agent
procedural_agent = ProceduralAgent(
    agent_id="procedure_executor",
    llm_client=provider,
    procedural_memory=memory
)

# Execute a procedure
result = await procedural_agent.execute_procedure(
    procedure_id,
    inputs={"parameter1": "value1"}
)
```

### Memory Systems

The PyContext memory system includes different types of memory:

- **Working Memory**: Short-term storage during agent interactions
- **Episodic Memory**: Long-term storage of experiences and interactions
- **Semantic Memory**: Knowledge network with semantic relationships
- **Procedural Memory**: Step-by-step procedures with dependency management

```python
from pycontext.core.memory.procedural_memory import ProceduralMemory

# Create procedural memory
memory = ProceduralMemory()

# Define a procedure using the builder pattern
builder = memory.create_procedure_builder()
procedure_id = builder\
    .set_name("Example Procedure")\
    .set_description("A simple example procedure")\
    .add_step(
        name="First Step",
        description="The first step",
        action={"type": "simple_action"}
    )\
    .add_step(
        name="Second Step",
        description="Depends on the first step",
        action={"type": "another_action"},
        dependencies=["step1"]
    )\
    .build()
```

### Agent Coordination

PyContext includes components for coordinating multiple agents:

- **Orchestrator**: Manages task distribution and execution
- **Planner**: Breaks down complex tasks into subtasks
- **Router**: Directs tasks to appropriate agents

```python
from pycontext.core.coordination.orchestrator import AgentOrchestrator

# Create an orchestrator
orchestrator = AgentOrchestrator(
    agents={"intent": intent_agent, "knowledge": knowledge_agent},
    context_manager=context_manager
)

# Create a task
task_id = await orchestrator.create_task(
    agent_type="knowledge",
    input_data={"query": "Tell me about climate change"},
    priority=TaskPriority.NORMAL
)

# Wait for the task to complete
result = await orchestrator.wait_for_task(task_id)
```

## Examples

Check out the examples directory for complete examples:

- `examples/simple_chatbot/`: Basic conversational agent
- `examples/customer_service/`: Customer service scenario with intent recognition
- `examples/procedural_agent/`: Procedural workflows with dependencies and error handling
- `examples/coordination/`: Multi-agent coordination with various agent types

## Documentation

- [Core Concepts](docs/core_concepts.md)
- [Memory Systems](docs/memory_systems.md)
- [Agent Types](docs/agent_types.md)
- [Procedural Agent](docs/procedural_agent.md)
- [Coordination](docs/coordination.md)
- [LLM Integration](docs/llm_integration.md)

## Development Status

PyContext is currently in alpha stage. The API may change before the first stable release.

## License

MIT