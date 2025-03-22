"""
examples/customer_service/comprehensive_customer_service.py

Comprehensive customer service example demonstrating PyContext's capabilities.
"""
import asyncio
import json
import os
import tempfile
from typing import Dict, List, Any, Optional
from datetime import datetime

from pycontext.core.agents.base import BaseAgent
from pycontext.core.agents.intent_agent import IntentAgent
from pycontext.core.agents.technical_agent import TechnicalAgent
from pycontext.core.context.manager import ContextManager
from pycontext.core.mcp.protocol import ContextType
from pycontext.core.memory.working_memory import WorkingMemory
from pycontext.core.memory.episodic_memory import EpisodicMemory, Episode
from pycontext.core.memory.embedding_provider import SimpleEmbeddingProvider
from pycontext.integrations.llm_providers.openai_provider import OpenAIProvider, OpenAIAgent
from pycontext.integrations.llm_providers.anthropic_provider import AnthropicProvider, AnthropicAgent
from pycontext.specialized.stores.redis_context_store import RedisContextStore


class MockRedisClient:
    """A simple mock Redis client for demonstration purposes."""

    def __init__(self):
        self.data = {}
        self.sets = {}

    async def hmset(self, key, mapping):
        if key not in self.data:
            self.data[key] = {}
        self.data[key].update(mapping)
        return True

    async def hgetall(self, key):
        return self.data.get(key, {})

    async def exists(self, key):
        return key in self.data

    async def sadd(self, key, *values):
        if key not in self.sets:
            self.sets[key] = set()
        self.sets[key].update(values)
        return len(values)

    async def smembers(self, key):
        return self.sets.get(key, set())

    async def set(self, key, value):
        self.data[key] = value
        return True

    async def get(self, key):
        return self.data.get(key)

    async def delete(self, key):
        if key in self.data:
            del self.data[key]
            return 1
        return 0

    async def expire(self, key, seconds):
        return 1

    async def keys(self, pattern):
        import fnmatch
        return [k for k in self.data.keys() if fnmatch.fnmatch(k, pattern)]

    def pipeline(self):
        return MockRedisPipeline(self)


class MockRedisPipeline:
    """A simple mock Redis pipeline."""

    def __init__(self, client):
        self.client = client
        self.commands = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def execute(self):
        results = []
        for cmd, args, kwargs in self.commands:
            method = getattr(self.client, cmd)
            if asyncio.iscoroutinefunction(method):
                result = await method(*args, **kwargs)
            else:
                result = method(*args, **kwargs)
            results.append(result)
        self.commands = []
        return results

    def __getattr__(self, name):
        async def wrapper(*args, **kwargs):
            self.commands.append((name, args, kwargs))
            return self

        return wrapper


class ComprehensiveCustomerService:
    """
    Comprehensive customer service system showcasing PyContext's capabilities.
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            anthropic_key: Optional[str] = None,
            use_redis: bool = False
    ):
        """
        Initialize the customer service system.

        Args:
            api_key: OpenAI API key (optional)
            anthropic_key: Anthropic API key (optional)
            use_redis: Whether to use Redis for context storage
        """
        # Initialize memory systems
        self.working_memory = WorkingMemory()
        self.embedding_provider = SimpleEmbeddingProvider()
        self.episodic_memory = EpisodicMemory(
            use_embeddings=True,
            embedding_provider=self.embedding_provider
        )

        # Initialize context management
        self.context_manager = ContextManager()

        # Configure Redis if requested
        self.redis_store = None
        if use_redis:
            # Use mock Redis for demonstration
            redis_client = MockRedisClient()
            self.redis_store = RedisContextStore(redis_client)

        # Initialize LLM providers
        self.openai_provider = None
        if api_key:
            self.openai_provider = OpenAIProvider(
                api_key=api_key,
                model="gpt-3.5-turbo"  # Use cheaper model for examples
            )

        self.anthropic_provider = None
        if anthropic_key:
            self.anthropic_provider = AnthropicProvider(
                api_key=anthropic_key,
                model="claude-3-sonnet-20240229"
            )

        # Select the primary provider based on availability
        self.primary_provider = self.anthropic_provider or self.openai_provider

        # Initialize agent system
        self.agents = {}
        self._setup_agents()

        # Technical tools for the technical agent
        self.technical_tools = {
            "network_diagnostic": self._mock_network_diagnostic,
            "hardware_diagnostic": self._mock_hardware_diagnostic,
            "system_check": self._mock_system_check
        }

        # Customer database (mock)
        self.customers = {}

    def _setup_agents(self):
        """Set up the agent system."""
        # Intent analysis agent
        if self.primary_provider:
            self.agents["intent"] = IntentAgent(
                agent_id="intent_analyzer",
                llm_client=self.primary_provider
            )

        # Technical support agent
        if self.primary_provider:
            self.agents["technical"] = TechnicalAgent(
                agent_id="tech_support",
                llm_client=self.primary_provider,
                technical_tools=self.technical_tools
            )

        # Response agents with different providers for comparison
        if self.openai_provider:
            self.agents["openai_response"] = OpenAIAgent(
                agent_id="openai_agent",
                agent_role="customer_service",
                openai_provider=self.openai_provider
            )

        if self.anthropic_provider:
            self.agents["anthropic_response"] = AnthropicAgent(
                agent_id="anthropic_agent",
                agent_role="customer_service",
                anthropic_provider=self.anthropic_provider
            )

        # Use one of the response agents as the primary
        if "anthropic_response" in self.agents:
            self.agents["response"] = self.agents["anthropic_response"]
        elif "openai_response" in self.agents:
            self.agents["response"] = self.agents["openai_response"]

    async def register_customer(
            self,
            customer_id: str,
            name: str,
            email: str,
            plan: str
    ) -> Dict:
        """
        Register a customer in the system.

        Args:
            customer_id: Unique customer identifier
            name: Customer name
            email: Customer email
            plan: Customer subscription plan

        Returns:
            Customer information
        """
        customer_info = {
            "id": customer_id,
            "name": name,
            "email": email,
            "plan": plan,
            "registered_at": datetime.now().isoformat(),
            "interactions": []
        }

        self.customers[customer_id] = customer_info

        # Store in episodic memory
        await self.episodic_memory.add(
            content=customer_info,
            episode_type="customer_registration",
            metadata={"customer_id": customer_id}
        )

        return customer_info

    async def process_query(
            self,
            customer_id: str,
            query: str,
            use_agent: str = "response"
    ) -> Dict:
        """
        Process a customer query.

        Args:
            customer_id: Customer identifier
            query: Customer's query
            use_agent: Agent to use for response generation

        Returns:
            Response and analysis
        """
        # Check if customer exists
        if customer_id not in self.customers:
            return {
                "error": "Customer not found",
                "message": "Please register the customer first."
            }

        # Store query in working memory
        query_id = self.working_memory.add(
            content=query,
            memory_type="customer_query",
            metadata={"customer_id": customer_id, "timestamp": datetime.now().isoformat()}
        )

        # Store in episodic memory
        episode_id = await self.episodic_memory.add(
            content={"query": query, "timestamp": datetime.now().isoformat()},
            episode_type="customer_query",
            metadata={"customer_id": customer_id}
        )

        # Step 1: Analyze intent
        print(f"Analyzing intent: '{query}'")
        intent_result = None

        if "intent" in self.agents:
            intent_agent = self.agents["intent"]
            intent_result = await intent_agent.process(query)

            print(f"Intent analysis: {json.dumps(intent_result, indent=2)}")

            # Store intent in working memory
            self.working_memory.add(
                content=intent_result,
                memory_type="intent_analysis",
                metadata={"customer_id": customer_id, "query_id": query_id}
            )

            # Store in episodic memory
            await self.episodic_memory.add(
                content=intent_result,
                episode_type="intent_analysis",
                metadata={"customer_id": customer_id},
                references=[episode_id]
            )

        # Step 2: Process based on intent
        response = None
        diagnosis = None

        if intent_result and self._is_technical_issue(intent_result):
            # Use technical agent for technical issues
            if "technical" in self.agents:
                tech_agent = self.agents["technical"]
                diagnosis = await tech_agent.diagnose_issue(query, customer_id)

                print(f"Technical diagnosis: {json.dumps(diagnosis, indent=2)}")

                # Store diagnosis in working memory
                diag_id = self.working_memory.add(
                    content=diagnosis,
                    memory_type="technical_diagnosis",
                    metadata={"customer_id": customer_id, "query_id": query_id}
                )

                # Store in episodic memory
                await self.episodic_memory.add(
                    content=diagnosis,
                    episode_type="technical_diagnosis",
                    metadata={"customer_id": customer_id},
                    references=[episode_id]
                )

                # Generate final response using the specified agent
                response = await self._generate_response(use_agent, query, intent_result, diagnosis)
            else:
                response = "Technical agent not available. Please provide valid LLM credentials."
        else:
            # For non-technical issues, use the specified response agent
            response = await self._generate_response(use_agent, query, intent_result)

        # If no response was generated, provide a fallback
        if not response:
            response = "I couldn't generate a response at this time. Please try again later."

        # Store response in working memory
        resp_id = self.working_memory.add(
            content=response,
            memory_type="agent_response",
            metadata={"customer_id": customer_id, "query_id": query_id}
        )

        # Store in episodic memory
        await self.episodic_memory.add(
            content={"response": response, "timestamp": datetime.now().isoformat()},
            episode_type="agent_response",
            metadata={"customer_id": customer_id},
            references=[episode_id]
        )

        # Update customer interactions
        self.customers[customer_id]["interactions"].append({
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "intent": intent_result,
            "diagnosis": diagnosis
        })

        # Return results
        return {
            "query": query,
            "intent_analysis": intent_result,
            "technical_diagnosis": diagnosis,
            "response": response,
            "customer_id": customer_id
        }

    async def search_customer_history(
            self,
            customer_id: str,
            query: str = None,
            limit: int = 5
    ) -> List[Dict]:
        """
        Search customer interaction history.

        Args:
            customer_id: Customer identifier
            query: Optional search query
            limit: Maximum number of results

        Returns:
            List of matching episodes
        """
        if not query:
            # Get recent episodes
            episodes = self.episodic_memory.get_by_metadata("customer_id", customer_id)
            episodes.sort(key=lambda e: e.timestamp, reverse=True)
            episodes = episodes[:limit]
            return [e.to_dict() for e in episodes]
        else:
            # Search episodes
            results = self.episodic_memory.search(
                query,
                limit=limit
            )

            # Filter by customer
            customer_results = []
            for episode, score in results:
                if episode.metadata.get("customer_id") == customer_id:
                    result_dict = episode.to_dict()
                    result_dict["similarity_score"] = score
                    customer_results.append(result_dict)

            return customer_results

    def get_customer_history(self, customer_id: str) -> Dict:
        """
        Get complete customer history.

        Args:
            customer_id: Customer identifier

        Returns:
            Customer information with interactions
        """
        if customer_id not in self.customers:
            return {"error": "Customer not found"}

        return self.customers[customer_id]

    async def save_context(self, agent_id: str, filename: str) -> bool:
        """
        Save agent context to a file.

        Args:
            agent_id: Agent identifier
            filename: Path to the file

        Returns:
            Whether the save was successful
        """
        if agent_id not in self.agents:
            return False

        agent = self.agents[agent_id]

        if not agent.session_id:
            return False

        context_data = agent.export_context()

        with open(filename, 'w') as file:
            json.dump(context_data, file, indent=2)

        return True

    async def load_context(self, agent_id: str, filename: str) -> bool:
        """
        Load agent context from a file.

        Args:
            agent_id: Agent identifier
            filename: Path to the file

        Returns:
            Whether the load was successful
        """
        if agent_id not in self.agents:
            return False

        agent = self.agents[agent_id]

        try:
            with open(filename, 'r') as file:
                context_data = json.load(file)

            agent.import_context(context_data)
            return True
        except Exception as e:
            print(f"Error loading context: {e}")
            return False

    async def _generate_response(
            self,
            agent_id: str,
            query: str,
            intent_result: Dict = None,
            diagnosis: Dict = None
    ) -> str:
        """
        Generate a response using the specified agent.

        Args:
            agent_id: Agent to use
            query: Customer's query
            intent_result: Intent analysis result
            diagnosis: Technical diagnosis

        Returns:
            Generated response
        """
        if agent_id not in self.agents:
            return "Specified agent not available."

        agent = self.agents[agent_id]

        # Initialize session if needed
        if not agent.session_id:
            await agent.initialize_session()

        # Add intent analysis to context if available
        if intent_result:
            await agent.add_memory_context(
                content=json.dumps(intent_result),
                relevance_score=0.8,
                metadata={"type": "intent_analysis"}
            )

        # Add diagnosis to context if available
        if diagnosis:
            await agent.add_memory_context(
                content=json.dumps(diagnosis),
                relevance_score=0.8,
                metadata={"type": "technical_diagnosis"}
            )

        # Process query
        response = await agent.process(query)

        # Store context if Redis is available
        if self.redis_store and agent.session_id:
            context_data = agent.export_context()
            await self.redis_store.store_session_async(context_data)

        return response

    def _is_technical_issue(self, intent_result: Dict) -> bool:
        """
        Determine if an intent represents a technical issue.

        Args:
            intent_result: Intent analysis result

        Returns:
            Whether the intent is a technical issue
        """
        if not intent_result:
            return False

        primary_intent = intent_result.get("primary_intent", "").lower()

        # Check for technical terms
        technical_terms = [
            "problem", "issue", "error", "broken", "not working",
            "technical", "fix", "repair", "troubleshoot"
        ]

        return any(term in primary_intent for term in technical_terms)

    async def _mock_network_diagnostic(
            self,
            issue_description: str,
            system_info: Dict = None
    ) -> Dict:
        """Mock network diagnostic tool."""
        # Very simple mock implementation
        return {
            "tool": "network_diagnostic",
            "status": "completed",
            "results": {
                "connectivity": "stable",
                "speed_test": {
                    "download": 95.4,  # Mbps
                    "upload": 15.2,  # Mbps
                    "latency": 28  # ms
                },
                "packet_loss": 0.5,  # percentage
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


async def run_comprehensive_demo():
    """Run a comprehensive demonstration."""
    # Get API keys from environment or use None
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    print("Starting comprehensive customer service demo...")

    # Check if any API keys are available
    if not openai_key and not anthropic_key:
        print("WARNING: No API keys provided. Using mock responses.")

    # Create customer service system
    service = ComprehensiveCustomerService(
        api_key=openai_key,
        anthropic_key=anthropic_key,
        use_redis=True  # Use mock Redis for demo
    )

    # Register a customer
    customer = await service.register_customer(
        customer_id="cust_123",
        name="Alex Johnson",
        email="alex@example.com",
        plan="Premium"
    )

    print(f"\nRegistered customer: {customer['name']} ({customer['id']})")

    # Example queries
    queries = [
        "I'd like to upgrade my current plan to include more data.",
        "My internet keeps disconnecting every hour or so, especially during video calls.",
        "Can you tell me when my next bill is due?",
        "The router you provided keeps blinking red on the power light. Is that normal?",
        "I've been a customer for 3 years now, do you have any loyalty discounts?"
    ]

    # Process each query
    for i, query in enumerate(queries):
        print(f"\n\n--- Query {i + 1} ---")
        print(f"Customer: {query}")

        # Process query
        result = await service.process_query("cust_123", query)

        # Print response
        print(f"\nAgent: {result['response']}")

        # Print intent if available
        if result.get("intent_analysis"):
            print(f"\nIntent: {result['intent_analysis'].get('primary_intent', 'Unknown')}")

        # Wait a bit between queries
        if i < len(queries) - 1:
            await asyncio.sleep(1)

    # Search customer history
    print("\n\n--- Customer History Search ---")

    search_results = await service.search_customer_history(
        customer_id="cust_123",
        query="internet disconnect"
    )

    print(f"Found {len(search_results)} relevant interactions:")
    for result in search_results:
        if isinstance(result.get("content"), dict) and "query" in result["content"]:
            print(f"\n- Query: {result['content']['query']}")
            if "response" in result["content"]:
                print(f"- Response summary: {result['content']['response'][:100]}...")

        if "similarity_score" in result:
            print(f"- Relevance: {result['similarity_score']:.2f}")

    # Save and load context
    print("\n\n--- Context Persistence Demo ---")

    # Save context to file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    filename = temp_file.name
    temp_file.close()

    if "response" in service.agents:
        success = await service.save_context("response", filename)
        if success:
            print(f"Successfully saved context to {filename}")

            # Let's load it into another agent if available
            if "technical" in service.agents:
                success = await service.load_context("technical", filename)
                if success:
                    print("Successfully loaded context into technical agent")

    # Clean up
    try:
        os.unlink(filename)
    except:
        pass

    print("\nDemo completed!")


if __name__ == "__main__":
    asyncio.run(run_comprehensive_demo())
