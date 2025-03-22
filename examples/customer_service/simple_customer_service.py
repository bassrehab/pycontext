"""
examples/customer_service/simple_customer_service.py

Simple customer service example using PyContext.
"""
import asyncio
import json
import os
from typing import Dict, List, Any

from pycontext.core.agents.base import BaseAgent
from pycontext.core.agents.intent_agent import IntentAgent
from pycontext.core.context.manager import ContextManager
from pycontext.core.mcp.protocol import ContextType
from pycontext.core.memory.working_memory import WorkingMemory
from pycontext.integrations.llm_providers.openai_provider import OpenAIProvider, OpenAIAgent


class SimpleCustomerServiceSystem:
    """
    Simple customer service system using PyContext.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the customer service system.

        Args:
            api_key: OpenAI API key (optional)
        """
        self.context_manager = ContextManager()
        self.working_memory = WorkingMemory()

        # Initialize OpenAI provider if API key is provided
        self.openai_provider = None
        if api_key:
            self.openai_provider = OpenAIProvider(
                api_key=api_key,
                model="gpt-3.5-turbo"  # Use cheaper model for examples
            )

        # Create intent agent
        self.intent_agent = IntentAgent(
            agent_id="intent_analyzer",
            llm_client=self.openai_provider
        )

        # Create response agent if OpenAI is available
        self.response_agent = None
        if self.openai_provider:
            self.response_agent = OpenAIAgent(
                agent_id="response_agent",
                agent_role="customer_service",
                openai_provider=self.openai_provider
            )

    async def process_query(self, customer_id: str, query: str) -> Dict:
        """
        Process a customer query.

        Args:
            customer_id: Customer identifier
            query: Customer's query

        Returns:
            Response and analysis
        """
        # Store query in working memory
        self.working_memory.add(
            content=query,
            memory_type="customer_query",
            metadata={"customer_id": customer_id}
        )

        # Step 1: Analyze intent
        print(f"Analyzing intent: '{query}'")
        intent_result = await self.intent_agent.process(query)

        print(f"Intent analysis: {json.dumps(intent_result, indent=2)}")

        # Store intent in working memory
        self.working_memory.add(
            content=intent_result,
            memory_type="intent_analysis",
            metadata={"customer_id": customer_id}
        )

        # Step 2: Generate response if OpenAI is available
        response = "No response agent available. Please provide an OpenAI API key to enable responses."

        if self.response_agent:
            print("Generating response...")

            # Initialize session if needed
            if not self.response_agent.session_id:
                await self.response_agent.initialize_session()

            # Add intent analysis to context
            await self.response_agent.add_memory_context(
                content=json.dumps(intent_result),
                relevance_score=0.8,
                metadata={"type": "intent_analysis"}
            )

            # Process query
            response = await self.response_agent.process(query)

        # Store response in working memory
        self.working_memory.add(
            content=response,
            memory_type="agent_response",
            metadata={"customer_id": customer_id}
        )

        # Return results
        return {
            "query": query,
            "intent_analysis": intent_result,
            "response": response
        }

    def get_customer_history(self, customer_id: str) -> List[Dict]:
        """
        Get customer interaction history.

        Args:
            customer_id: Customer identifier

        Returns:
            List of interactions
        """
        # Get all memory items for this customer
        query_items = self.working_memory.get_by_metadata("customer_id", customer_id)

        # Convert to history format
        history = []

        for item in self.working_memory.items.values():
            if item.metadata.get("customer_id") == customer_id:
                history.append({
                    "timestamp": item.timestamp,
                    "type": item.memory_type,
                    "content": item.content
                })

        # Sort by timestamp
        history.sort(key=lambda x: x["timestamp"])

        return history


async def run_demo():
    """Run a simple demonstration."""
    # Get API key from environment or use None
    api_key = os.environ.get("OPENAI_API_KEY")

    # Create customer service system
    system = SimpleCustomerServiceSystem(api_key=api_key)

    # Example queries
    queries = [
        "I'm having trouble connecting to my WiFi. It was working yesterday but now it keeps disconnecting.",
        "I want to upgrade my current plan to include more data.",
        "How much does your premium package cost?",
        "Can you explain the charges on my last bill?",
        "I need help setting up my new router."
    ]

    # Process each query
    customer_id = "customer_123"

    for i, query in enumerate(queries):
        print(f"\n=== Query {i + 1} ===")
        result = await system.process_query(customer_id, query)

        print("\nResponse:")
        print(result["response"])

        print("\n" + "=" * 50)

    # Get customer history
    print("\n=== Customer History ===")
    history = system.get_customer_history(customer_id)

    for entry in history:
        if entry["type"] == "customer_query":
            print(f"\nCustomer at {entry['timestamp']}:")
            print(entry["content"])
        elif entry["type"] == "agent_response":
            print(f"\nAgent at {entry['timestamp']}:")
            print(entry["content"])

    print("\nDemo completed!")


if __name__ == "__main__":
    asyncio.run(run_demo())
