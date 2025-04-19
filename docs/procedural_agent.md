# Procedural Agent and Memory

The Procedural Agent and Procedural Memory system in PyContext provide a powerful framework for defining, executing, and managing step-by-step procedures with dependencies, error handling, and coordination with other agents.

## Overview

The Procedural system consists of two main components:

1. **Procedural Memory**: Stores and manages procedure definitions, their execution state, and history.
2. **Procedural Agent**: Interfaces with the Procedural Memory to create, execute, and manage procedures based on user requests.

This system enables:
- Creating procedures programmatically or through natural language
- Executing multi-step procedures with proper dependency management
- Handling errors and retries automatically
- Coordinating with other specialized agents
- Tracking execution state and results

## Procedural Memory

Procedural Memory stores the knowledge of how to perform specific procedures, including the steps, their dependencies, and execution order.

### Key Components

- **Procedure**: A complete workflow with steps, dependencies, and execution state.
- **ProcedureStep**: An individual action in a procedure with its own state and results.
- **ProcedureExecutor**: Handles the execution of procedures, respecting dependencies.
- **ProcedureBuilder**: Helper for creating procedures with a fluent API.

### Creating Procedures

Procedures can be created programmatically using the builder pattern:

```python
from pycontext.core.memory.procedural_memory import ProceduralMemory

# Create memory
memory = ProceduralMemory()

# Create a procedure builder
builder = memory.create_procedure_builder()

# Define the procedure
procedure_id = builder\
    .set_name("Data Processing Workflow")\
    .set_description("A workflow to process data")\
    .add_tag("data")\
    .add_tag("processing")\
    .add_input("source", "default.csv")\
    
    # Define steps with dependencies
    .add_step(
        name="Fetch Data",
        description="Fetch data from the source",
        action={"type": "fetch_data", "source": "{source}"}
    )\
    .add_step(
        name="Process Data",
        description="Process the fetched data",
        action={"type": "process_data"},
        dependencies=["step1"]  # Depends on first step
    )\
    .build()
```

### Action Handlers

Procedural Memory uses action handlers to execute the specific actions in procedure steps:

```python
# Register action handlers
memory.register_action_handler("fetch_data", fetch_data_handler)
memory.register_action_handler("process_data", process_data_handler)

# Example handler function
async def fetch_data_handler(action, inputs):
    source = action.get("source", "default.csv")
    # Fetch data logic...
    return {
        "status": "success",
        "outputs": {
            "data": [...]
        }
    }
```

### Executing Procedures

Procedures can be executed directly through Procedural Memory:

```python
# Execute a procedure
result = await memory.execute_procedure(
    procedure_id,
    inputs={"source": "custom.csv"}
)

# Check result
if result.status == ProcedureStatus.COMPLETED:
    print("Procedure completed successfully!")
    print(f"Outputs: {result.outputs}")
else:
    print(f"Procedure failed: {result.error}")
```

## Procedural Agent

The Procedural Agent provides a higher-level interface for working with procedures, including:

- Converting natural language instructions into procedures
- Managing procedure execution based on user requests
- Providing descriptive feedback during execution
- Coordinating with other agents during procedure steps

### Initialization

```python
from pycontext.core.agents.procedural_agent import ProceduralAgent
from pycontext.core.memory.procedural_memory import ProceduralMemory

# Create memory and agent
memory = ProceduralMemory()
agent = ProceduralAgent(
    agent_id="procedure_executor",
    llm_client=llm_provider,  # LLM provider for natural language processing
    procedural_memory=memory
)

# Initialize the agent
await agent.initialize_session()
```

### Creating Procedures with Natural Language

```python
# Create a procedure using natural language
result = await agent.process(
    "Create a procedure for customer onboarding with these steps: "
    "1. Collect customer information "
    "2. Verify email address "
    "3. Set up account "
    "4. Send welcome email"
)

if result.get("success", False):
    procedure_id = result["procedure_id"]
    print(f"Created procedure with ID: {procedure_id}")
```

### Executing Procedures

```python
# Execute a procedure via the agent
result = await agent.process(f"Execute the Customer Onboarding procedure with name=John Smith, email=john@example.com")

# Or directly
result = await agent.execute_procedure(
    procedure_id,
    inputs={
        "customer_name": "John Smith",
        "customer_email": "john@example.com"
    }
)
```

### Built-in Action Handlers

The Procedural Agent comes with several built-in action handlers:

1. **llm_query**: Sends queries to the LLM
2. **input_validation**: Validates inputs against rules
3. **conditional**: Implements branching logic
4. **wait**: Introduces delays in execution
5. **output_transformation**: Transforms outputs between steps

## Error Handling and Recovery

The Procedural system includes robust error handling:

- Step-level timeouts
- Procedure-level timeouts
- Automatic retries with configurable delay
- Detailed error reporting

Example with retry logic:

```python
builder.add_step(
    name="Retry Example",
    description="A step with retry logic",
    action={"type": "api_call", "endpoint": "/users"},
    max_retries=3,
    retry_delay=1.0  # 1 second delay between retries
)
```

## Multi-Agent Integration

Procedural Agent can coordinate with other specialized agents:

```python
# Register handlers for agent coordination
memory.register_action_handler("consult_knowledge_agent", handle_knowledge_agent)
memory.register_action_handler("consult_technical_agent", handle_technical_agent)

# Create a procedure that uses multiple agents
builder.add_step(
    name="Get Technical Diagnosis",
    description="Use technical agent to diagnose an issue",
    action={
        "type": "consult_technical_agent",
        "issue": "{issue_description}"
    }
)
```

## Example: Data Processing Workflow

Here's a complete example of a data processing workflow:

```python
# Create procedure
builder = memory.create_procedure_builder()
procedure_id = builder\
    .set_name("Data Processing")\
    .set_description("Process and analyze data")\
    
    # Fetch data
    .add_step(
        name="Fetch Data",
        action={"type": "fetch_data", "source": "{source}"}
    )\
    
    # Validate data
    .add_step(
        name="Validate Data",
        action={
            "type": "input_validation",
            "validations": [
                {
                    "field": "data",
                    "rules": [{"type": "required"}, {"type": "array_not_empty"}]
                }
            ]
        },
        dependencies=["step1"]
    )\
    
    # Process data
    .add_step(
        name="Process Data",
        action={"type": "process_data", "mode": "advanced"},
        dependencies=["step2"]
    )\
    
    # Generate report using LLM
    .add_step(
        name="Generate Report",
        action={
            "type": "llm_query",
            "prompt": "Generate a summary report of this data: {processed_data}"
        },
        dependencies=["step3"]
    )\
    .build()

# Execute procedure
result = await memory.execute_procedure(
    procedure_id,
    inputs={"source": "data.csv"}
)
```

## Conclusion

The Procedural Agent and Memory system provides a powerful framework for defining and executing complex workflows in your applications. By combining procedural logic with other specialized agents, you can create sophisticated autonomous systems that can handle a wide range of tasks with proper dependency management, error handling, and coordination.