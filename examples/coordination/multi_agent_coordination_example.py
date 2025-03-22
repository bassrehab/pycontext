"""
examples/coordination/multi_agent_coordination_example.py

Example demonstrating the agent coordination system.
"""
import asyncio
import json
import os
import logging
from typing import Dict, List, Any, Optional

from pycontext.core.agents.base import BaseAgent
from pycontext.core.agents.intent_agent import IntentAgent
from pycontext.core.agents.technical_agent import TechnicalAgent
from pycontext.core.agents.knowledge_agent import KnowledgeAgent
from pycontext.core.context.manager import ContextManager
from pycontext.core.memory.working_memory import WorkingMemory
from pycontext.core.memory.semantic_memory import SemanticMemory
from pycontext.core.memory.embedding_provider import SimpleEmbeddingProvider
from pycontext.core.coordination.orchestrator import AgentOrchestrator, TaskPriority
from pycontext.core.coordination.planner import TaskPlanner, DEFAULT_TASK_TEMPLATES
from pycontext.core.coordination.router import AgentRouter, MultiAgentRouter
from pycontext.integrations.llm_providers.openai_provider import OpenAIProvider, OpenAIAgent
from pycontext.integrations.llm_providers.anthropic_provider import AnthropicProvider, AnthropicAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CoordinationExample:
    """
    Example demonstrating the agent coordination system.
    Shows how multiple specialized agents can work together to solve complex tasks.
    """

    def __init__(
            self,
            openai_key: Optional[str] = None,
            anthropic_key: Optional[str] = None
    ):
        """
        Initialize the coordination example.

        Args:
            openai_key: Optional OpenAI API key
            anthropic_key: Optional Anthropic API key
        """
        # Set up providers
        self.openai_provider = None
        if openai_key:
            self.openai_provider = OpenAIProvider(
                api_key=openai_key,
                model="gpt-3.5-turbo"  # Use cheaper model for examples
            )

        self.anthropic_provider = None
        if anthropic_key:
            self.anthropic_provider = AnthropicProvider(
                api_key=anthropic_key,
                model="claude-3-sonnet-20240229"
            )

        # Select primary provider
        self.primary_provider = self.anthropic_provider or self.openai_provider

        # Set up memory systems
        self.context_manager = ContextManager()
        self.working_memory = WorkingMemory()
        self.embedding_provider = SimpleEmbeddingProvider()
        self.semantic_memory = SemanticMemory(embedding_provider=self.embedding_provider)

        # Set up agents
        self.agents = {}
        self._setup_agents()

        # Set up coordination components
        self.orchestrator = AgentOrchestrator(
            agents=self.agents,
            context_manager=self.context_manager
        )

        self.planner = TaskPlanner(
            available_agents=list(self.agents.keys()),
            llm_client=self.primary_provider,
            context_manager=self.context_manager,
            task_templates=DEFAULT_TASK_TEMPLATES
        )

        self.router = AgentRouter(
            agents=self.agents,
            default_agent="knowledge_agent" if "knowledge_agent" in self.agents else list(self.agents.keys())[0],
            intent_agent="intent_agent" if "intent_agent" in self.agents else None,
            context_manager=self.context_manager
        )

        self.multi_router = MultiAgentRouter(
            agents=self.agents,
            default_sequence=["intent_agent", "knowledge_agent",
                              "technical_agent"] if "intent_agent" in self.agents else list(self.agents.keys()),
            intent_agent="intent_agent" if "intent_agent" in self.agents else None,
            context_manager=self.context_manager
        )

        # Set up technical tools for the technical agent
        if "technical_agent" in self.agents:
            self.agents["technical_agent"].technical_tools = {
                "network_diagnostic": self._mock_network_diagnostic,
                "hardware_diagnostic": self._mock_hardware_diagnostic,
                "system_check": self._mock_system_check
            }

    def _setup_agents(self):
        """Set up agent instances."""
        # Create intent agent
        if self.primary_provider:
            self.agents["intent_agent"] = IntentAgent(
                agent_id="intent_analyzer",
                llm_client=self.primary_provider
            )

        # Create knowledge agent
        if self.primary_provider:
            self.agents["knowledge_agent"] = KnowledgeAgent(
                agent_id="knowledge_provider",
                llm_client=self.primary_provider,
                semantic_memory=self.semantic_memory
            )

        # Create technical agent
        if self.primary_provider:
            self.agents["technical_agent"] = TechnicalAgent(
                agent_id="technical_support",
                llm_client=self.primary_provider
            )

        # Create response agents with different providers for comparison
        if self.openai_provider:
            self.agents["openai_assistant"] = OpenAIAgent(
                agent_id="openai_assistant",
                agent_role="assistant",
                openai_provider=self.openai_provider
            )

        if self.anthropic_provider:
            self.agents["anthropic_assistant"] = AnthropicAgent(
                agent_id="anthropic_assistant",
                agent_role="assistant",
                anthropic_provider=self.anthropic_provider
            )

    async def seed_knowledge(self):
        """Seed the semantic memory with some example knowledge."""
        if "knowledge_agent" not in self.agents:
            logger.warning("No knowledge agent available for seeding knowledge")
            return

        knowledge_agent = self.agents["knowledge_agent"]

        # Add some technical knowledge
        await knowledge_agent.add_knowledge(
            content="WiFi connections can be disrupted by interference from other devices, "
                    "walls, or distance from the router. Common solutions include resetting "
                    "the router, changing the WiFi channel, or moving closer to the router.",
            entry_type="technical_knowledge",
            metadata={"topic": "networking", "subtopic": "wifi"}
        )

        await knowledge_agent.add_knowledge(
            content="If your computer is running slowly, it could be due to too many background "
                    "processes, insufficient RAM, or malware. Try closing unnecessary programs, "
                    "scanning for malware, or upgrading your RAM.",
            entry_type="technical_knowledge",
            metadata={"topic": "hardware", "subtopic": "performance"}
        )

        # Add some product knowledge
        await knowledge_agent.add_knowledge(
            content="Our Premium Plan includes 100GB of data per month, unlimited calling, "
                    "and access to our streaming service. It costs $50 per month with a "
                    "12-month contract, or $60 month-to-month.",
            entry_type="product_knowledge",
            metadata={"topic": "plans", "name": "Premium Plan"}
        )

        await knowledge_agent.add_knowledge(
            content="To upgrade your plan, log into your account on our website, go to "
                    "'Account Settings', then 'Plan Options', and select the plan you want "
                    "to upgrade to. Changes typically take effect on your next billing cycle.",
            entry_type="support_knowledge",
            metadata={"topic": "account_management", "action": "upgrade"}
        )

        # Add some policy knowledge
        await knowledge_agent.add_knowledge(
            content="Our refund policy allows for full refunds within 14 days of purchase "
                    "for hardware products, and prorated refunds for service cancellations. "
                    "Refunds typically take 3-5 business days to process.",
            entry_type="policy_knowledge",
            metadata={"topic": "refunds"}
        )

        logger.info("Seeded knowledge base with example entries")

    async def demonstrate_orchestrator(self):
        """Demonstrate the orchestrator with a complex task."""
        print("\n--- ORCHESTRATOR DEMONSTRATION ---")
        print("Creating and executing a complex task with multiple subtasks")

        # Start the orchestrator
        await self.orchestrator.start()

        # Create a composite task
        parent_task_id = await self.orchestrator.create_composite_task(
            subtasks=[
                {
                    "agent_type": "intent_agent",
                    "input_data": {
                        "query": "I'm having trouble connecting to WiFi and I want to upgrade my plan."
                    },
                    "priority": TaskPriority.HIGH.value
                },
                {
                    "agent_type": "technical_agent",
                    "input_data": {
                        "query": "I'm having trouble connecting to WiFi",
                        "diagnose": True
                    },
                    "priority": TaskPriority.NORMAL.value
                },
                {
                    "agent_type": "knowledge_agent",
                    "input_data": {
                        "query": "How do I upgrade my plan?",
                        "retrieve_knowledge": {"limit": 2}
                    },
                    "priority": TaskPriority.NORMAL.value
                }
            ],
            aggregate_results=True,
            sequence=True  # Execute in sequence
        )

        print(f"Created composite task with ID: {parent_task_id}")
        print("Waiting for task completion...")

        # Wait for the task to complete
        result = await self.orchestrator.wait_for_task(parent_task_id)

        print("\nTask completed!")
        print(f"Total duration: {result.get('duration', 0):.2f} seconds")

        # Display results
        if 'result' in result and 'aggregated_result' in result['result']:
            for key, values in result['result']['aggregated_result'].items():
                print(f"\n{key.upper()}:")
                for value in values:
                    if isinstance(value, dict):
                        print(json.dumps(value, indent=2))
                    else:
                        print(value)

        # Stop the orchestrator
        await self.orchestrator.stop()

    async def demonstrate_planner(self):
        """Demonstrate the task planner."""
        print("\n--- PLANNER DEMONSTRATION ---")

        # Example queries to plan
        queries = [
            "My internet keeps dropping every hour or so, especially during video calls.",
            "Can you tell me about the different plans you offer?",
            "I'm trying to figure out how to upgrade my current subscription."
        ]

        for query in queries:
            print(f"\nQuery: '{query}'")
            print("Creating plan...")

            # Create plan
            plan = await self.planner.create_plan(
                goal=query,
                context={"customer_id": "example_customer"}
            )

            # Display plan
            print(f"Plan created with {len(plan.subtasks)} subtasks:")
            for i, subtask in enumerate(plan.subtasks):
                print(f"  {i + 1}. {subtask['agent_type']}: {subtask['input_data']}")

        # Try with a template
        print("\nUsing a template for technical support:")
        plan = await self.planner.create_plan(
            goal="My laptop keeps freezing after I updated my operating system.",
            template_name="technical_support"
        )

        print(f"Template plan created with {len(plan.subtasks)} subtasks:")
        for i, subtask in enumerate(plan.subtasks):
            print(f"  {i + 1}. {subtask['agent_type']}: {subtask['input_data']}")

    async def demonstrate_router(self):
        """Demonstrate the agent router."""
        print("\n--- ROUTER DEMONSTRATION ---")

        # Example queries to route
        queries = [
            "My WiFi connection keeps dropping.",
            "What are the features of your Premium Plan?",
            "Hello, how are you today?",
            "Can you help me troubleshoot my printer?"
        ]

        for query in queries:
            print(f"\nQuery: '{query}'")

            # Simple routing
            agent_type, confidence = await self.router.route(query)
            print(f"Routed to: {agent_type} (confidence: {confidence:.2f})")

            # Process with router
            result = await self.router.process_message(query)
            print(f"Processed with: {result.get('agent_type', 'unknown')}")

            # Multi-agent routing
            sequence = await self.multi_router.route(query)
            print(f"Multi-agent sequence: {sequence}")

    async def run_conversation_example(self):
        """Run a full conversation example with the multi-agent router."""
        print("\n--- CONVERSATION EXAMPLE ---")

        # Create a session for the conversation
        session_id = self.context_manager.create_session("conversation")

        # Example conversation
        conversation = [
            "Hi, I need some help with my internet connection and plan.",
            "My WiFi keeps dropping, especially when I'm on video calls.",
            "I'm also interested in upgrading my current plan.",
            "What options do I have for plans with more data?",
            "Thank you for your help!"
        ]

        for message in conversation:
            print(f"\nUser: {message}")

            # Process with multi-agent router
            result = await self.multi_router.process_message(
                message,
                session_id=session_id,
                max_agents=2  # Limit to 2 agents per message
            )

            # Display agent sequence
            print(f"Agents: {', '.join(result['agent_sequence'])}")

            # Display final result
            if result.get('final_result'):
                if isinstance(result['final_result'], dict):
                    if 'answer' in result['final_result']:
                        print(f"Assistant: {result['final_result']['answer']}")
                    else:
                        print(f"Assistant: {json.dumps(result['final_result'], indent=2)}")
                else:
                    print(f"Assistant: {result['final_result']}")
            else:
                print("Assistant: I'm not sure how to respond to that.")

    async def _mock_network_diagnostic(
            self,
            issue_description: str,
            system_info: Dict = None
    ) -> Dict:
        """Mock network diagnostic tool."""
        # Simple mock implementation
        return {
            "tool": "network_diagnostic",
            "status": "completed",
            "results": {
                "connectivity": "unstable",
                "speed_test": {
                    "download": 25.4,  # Mbps (slow)
                    "upload": 5.2,  # Mbps (slow)
                    "latency": 120  # ms (high)
                },
                "packet_loss": 15.5,  # percentage (high)
                "dns_resolution": "functional"
            }
        }

    async def _mock_hardware_diagnostic(
            self,
            issue_description: str,
            system_info: Dict = None
    ) -> Dict:
        """Mock hardware diagnostic tool."""
        return {
            "tool": "hardware_diagnostic",
            "status": "completed",
            "results": {
                "device_health": "good",
                "cpu_usage": 45,  # percentage
                "memory_usage": 60,  # percentage
                "disk_space": 75,  # percentage used
                "temperature": "normal"
            }
        }

    async def _mock_system_check(
            self,
            issue_description: str,
            system_info: Dict = None
    ) -> Dict:
        """Mock system check tool."""
        return {
            "tool": "system_check",
            "status": "completed",
            "results": {
                "os_status": "healthy",
                "driver_status": "up-to-date",
                "firmware_version": "1.2.3",
                "required_updates": []
            }
        }


async def run_coordination_demo():
    """Run the full coordination demonstration."""
    # Get API keys from environment
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not openai_key and not anthropic_key:
        print("WARNING: No API keys provided. Most demonstrations will produce mock responses.")

    # Create the example
    example = CoordinationExample(
        openai_key=openai_key,
        anthropic_key=anthropic_key
    )

    # Seed knowledge base
    await example.seed_knowledge()

    # Run demonstrations
    await example.demonstrate_planner()
    await example.demonstrate_router()
    await example.demonstrate_orchestrator()
    await example.run_conversation_example()

    print("\n--- DEMONSTRATION COMPLETE ---")



async def run_coordination_demo():
    """Run the full coordination demonstration."""
    # Get API keys from environment
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not openai_key and not anthropic_key:
        print("WARNING: No API keys provided. Most demonstrations will produce mock responses.")

    # Create the example
    example = CoordinationExample(
        openai_key=openai_key,
        anthropic_key=anthropic_key
    )

    # Seed knowledge base
    await example.seed_knowledge()

    # Run demonstrations
    await example.demonstrate_planner()
    await example.demonstrate_router()
    await example.demonstrate_orchestrator()
    await example.run_conversation_example()

    print("\n--- DEMONSTRATION COMPLETE ---")


if __name__ == "__main__":
    asyncio.run(run_coordination_demo())