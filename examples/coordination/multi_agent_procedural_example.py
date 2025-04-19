"""
examples/coordination/multi_agent_procedural_example.py

Example demonstrating integration of the Procedural Agent with other agent types 
in a multi-agent system.
"""
import asyncio
import json
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import time
import uuid

from pycontext.core.agents.base import BaseAgent
from pycontext.core.agents.intent_agent import IntentAgent
from pycontext.core.agents.technical_agent import TechnicalAgent
from pycontext.core.agents.knowledge_agent import KnowledgeAgent
from pycontext.core.agents.dialog_agent import DialogAgent, DialogState
from pycontext.core.agents.procedural_agent import ProceduralAgent
from pycontext.core.context.manager import ContextManager
from pycontext.core.memory.working_memory import WorkingMemory
from pycontext.core.memory.semantic_memory import SemanticMemory
from pycontext.core.memory.procedural_memory import ProceduralMemory, ProcedureStatus
from pycontext.core.memory.embedding_provider import SimpleEmbeddingProvider
from pycontext.core.coordination.orchestrator import AgentOrchestrator, TaskPriority
from pycontext.core.coordination.planner import TaskPlanner
from pycontext.core.coordination.router import AgentRouter, MultiAgentRouter
from pycontext.integrations.llm_providers.openai_provider import OpenAIProvider
from pycontext.integrations.llm_providers.anthropic_provider import AnthropicProvider

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiAgentProceduralSystem:
    """
    A multi-agent system that integrates the Procedural Agent with other agent types.
    Demonstrates coordination between agents for complex workflows.
    """

    def __init__(
            self,
            openai_key: Optional[str] = None,
            anthropic_key: Optional[str] = None
    ):
        """
        Initialize the multi-agent system.

        Args:
            openai_key: Optional OpenAI API key
            anthropic_key: Optional Anthropic API key
        """
        # Set up LLM providers
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

        # Initialize shared memory systems
        self.context_manager = ContextManager()
        self.working_memory = WorkingMemory()
        self.embedding_provider = SimpleEmbeddingProvider()
        self.semantic_memory = SemanticMemory(embedding_provider=self.embedding_provider)
        self.procedural_memory = ProceduralMemory()

        # Initialize agents
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
            context_manager=self.context_manager
        )

        self.router = MultiAgentRouter(
            agents=self.agents,
            default_sequence=[
                "intent_agent", 
                "dialog_agent", 
                "knowledge_agent"
            ],
            intent_agent="intent_agent" if "intent_agent" in self.agents else None,
            context_manager=self.context_manager
        )

        # Set up technical tools
        self._register_technical_tools()
        
        # Set up procedural tool handlers
        self._register_procedural_tools()
        
        # Initialize session
        self.session_id = self.context_manager.create_session("multi_agent_system")
        
        # Track created procedures
        self.procedures = {}

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
        
        # Create dialog agent
        if self.primary_provider:
            self.agents["dialog_agent"] = DialogAgent(
                agent_id="dialog_manager",
                llm_client=self.primary_provider,
                working_memory=self.working_memory
            )
        
        # Create procedural agent
        if self.primary_provider:
            self.agents["procedural_agent"] = ProceduralAgent(
                agent_id="procedure_executor",
                llm_client=self.primary_provider,
                procedural_memory=self.procedural_memory
            )
        
        # Initialize agent sessions
        for agent_name, agent in self.agents.items():
            asyncio.run(agent.initialize_session())

    def _register_technical_tools(self):
        """Register technical tools."""
        if "technical_agent" in self.agents:
            technical_agent = self.agents["technical_agent"]
            technical_agent.technical_tools = {
                "network_diagnostic": self._mock_network_diagnostic,
                "hardware_diagnostic": self._mock_hardware_diagnostic,
                "system_check": self._mock_system_check
            }

    def _register_procedural_tools(self):
        """Register procedural tool handlers."""
        if "procedural_agent" in self.agents:
            # Register communication tools
            self.procedural_memory.register_action_handler("consult_knowledge_agent", self._handle_consult_knowledge)
            self.procedural_memory.register_action_handler("consult_technical_agent", self._handle_consult_technical)
            
            # Register utility tools
            self.procedural_memory.register_action_handler("store_in_memory", self._handle_store_in_memory)
            self.procedural_memory.register_action_handler("retrieve_from_memory", self._handle_retrieve_from_memory)
            
            # Register domain-specific tools
            self.procedural_memory.register_action_handler("check_customer_info", self._handle_check_customer_info)
            self.procedural_memory.register_action_handler("update_customer_info", self._handle_update_customer_info)
            self.procedural_memory.register_action_handler("generate_report", self._handle_generate_report)

    async def seed_knowledge(self):
        """Seed the knowledge base with example information."""
        print("Seeding knowledge base...")
        
        if "knowledge_agent" not in self.agents:
            print("Warning: Knowledge agent not available")
            return
            
        knowledge_agent = self.agents["knowledge_agent"]
        
        # Add product knowledge
        await knowledge_agent.add_knowledge(
            content="The Basic Plan costs $29.99/month and includes 50GB data, standard support, and a 12-month commitment.",
            entry_type="product_info",
            metadata={"product": "Basic Plan"}
        )
        
        await knowledge_agent.add_knowledge(
            content="The Premium Plan costs $59.99/month and includes 200GB data, priority support, and a 12-month commitment.",
            entry_type="product_info",
            metadata={"product": "Premium Plan"}
        )
        
        await knowledge_agent.add_knowledge(
            content="The Enterprise Plan costs $129.99/month and includes unlimited data, 24/7 dedicated support, and a 24-month commitment.",
            entry_type="product_info",
            metadata={"product": "Enterprise Plan"}
        )
        
        # Add support knowledge
        await knowledge_agent.add_knowledge(
            content="To upgrade a plan, log into the customer portal, go to 'My Plan', click 'Upgrade', select the new plan, and follow payment instructions.",
            entry_type="procedure_info",
            metadata={"topic": "plan_upgrade"}
        )
        
        await knowledge_agent.add_knowledge(
            content="To troubleshoot connectivity issues, first check physical connections, restart your device, check for service outages in your area, and reset your network settings.",
            entry_type="troubleshooting",
            metadata={"topic": "connectivity"}
        )
        
        print("Knowledge base seeded successfully")

    async def create_customer_onboarding_procedure(self):
        """
        Create a customer onboarding procedure that integrates multiple agents.
        """
        print("\n=== Creating Customer Onboarding Procedure ===")
        
        # Create a procedure builder
        builder = self.procedural_memory.create_procedure_builder()
        
        # Define the procedure
        builder.set_name("Customer Onboarding")
        builder.set_description("Procedure for onboarding new customers with account verification and setup")
        builder.add_tag("customer")
        builder.add_tag("onboarding")
        
        # Add required inputs
        builder.add_input("customer_name", None)
        builder.add_input("customer_email", None)
        builder.add_input("selected_plan", "Basic Plan")
        
        # Step 1: Collect customer information
        step1 = builder.add_step(
            name="Collect Customer Information",
            description="Collect basic customer information",
            action={
                "type": "get_user_input",
                "prompt": "Please provide customer name and email",
                "required_fields": ["customer_name", "customer_email"]
            }
        )
        
        # Step 2: Verify email address
        step2 = builder.add_step(
            name="Verify Customer Email",
            description="Verify the customer's email address format",
            action={
                "type": "input_validation",
                "validations": [
                    {
                        "field": "customer_email",
                        "rules": [
                            {"type": "required"},
                            {"type": "email"}
                        ]
                    }
                ]
            },
            dependencies=[step1]
        )
        
        # Step 3: Check for existing customer
        step3 = builder.add_step(
            name="Check Existing Customer",
            description="Check if customer already exists in the system",
            action={
                "type": "check_customer_info",
                "email": "{customer_email}"
            },
            dependencies=[step2]
        )
        
        # Step 4: Consult knowledge agent about plan details
        step4 = builder.add_step(
            name="Get Plan Information",
            description="Retrieve information about the selected plan",
            action={
                "type": "consult_knowledge_agent",
                "query": "What are the details of the {selected_plan}?"
            },
            dependencies=[step2]
        )
        
        # Step 5: Create customer account
        step5 = builder.add_step(
            name="Create Customer Account",
            description="Create a new customer account in the system",
            action={
                "type": "update_customer_info",
                "operation": "create",
                "customer_data": {
                    "name": "{customer_name}",
                    "email": "{customer_email}",
                    "plan": "{selected_plan}",
                    "status": "active"
                }
            },
            dependencies=[step3, step4]
        )
        
        # Step 6: Generate welcome email
        step6 = builder.add_step(
            name="Generate Welcome Email",
            description="Generate a personalized welcome email for the customer",
            action={
                "type": "llm_query",
                "prompt": "Generate a personalized welcome email for a new customer named {customer_name} who has signed up for our {selected_plan}. Include information about their plan: {plan_info}"
            },
            dependencies=[step5]
        )
        
        # Step 7: Store welcome email in memory
        step7 = builder.add_step(
            name="Save Welcome Email",
            description="Save the welcome email in working memory",
            action={
                "type": "store_in_memory",
                "content": "{welcome_email}",
                "memory_type": "email_template",
                "metadata": {
                    "customer_email": "{customer_email}",
                    "type": "welcome_email"
                }
            },
            dependencies=[step6]
        )
        
        # Step 8: Generate detailed report
        builder.add_step(
            name="Generate Onboarding Report",
            description="Generate a detailed report of the onboarding process",
            action={
                "type": "generate_report",
                "title": "Customer Onboarding Report",
                "sections": [
                    "Customer Information",
                    "Plan Details",
                    "Account Creation",
                    "Next Steps"
                ],
                "customer_info": {
                    "name": "{customer_name}",
                    "email": "{customer_email}",
                    "plan": "{selected_plan}"
                }
            },
            dependencies=[step7]
        )
        
        # Build the procedure
        procedure_id = builder.build()
        self.procedures["customer_onboarding"] = procedure_id
        
        print(f"Created customer onboarding procedure (ID: {procedure_id})")
        return procedure_id

    async def create_technical_support_procedure(self):
        """
        Create a technical support procedure that integrates with the technical agent.
        """
        print("\n=== Creating Technical Support Procedure ===")
        
        # Create a procedure builder
        builder = self.procedural_memory.create_procedure_builder()
        
        # Define the procedure
        builder.set_name("Technical Support Workflow")
        builder.set_description("Procedure for diagnosing and resolving technical issues")
        builder.add_tag("technical")
        builder.add_tag("support")
        
        # Add required inputs
        builder.add_input("customer_id", None)
        builder.add_input("issue_description", None)
        
        # Step 1: Retrieve customer information
        step1 = builder.add_step(
            name="Retrieve Customer Information",
            description="Get customer information from the system",
            action={
                "type": "check_customer_info",
                "customer_id": "{customer_id}"
            }
        )
        
        # Step 2: Analyze issue intent
        step2 = builder.add_step(
            name="Analyze Issue Intent",
            description="Analyze the customer's issue description to determine the type of problem",
            action={
                "type": "llm_query",
                "prompt": "Analyze this technical issue description and classify it into one of these categories: connectivity, hardware, software, account, or other. Issue: {issue_description}",
                "extract_json": True
            },
            dependencies=[step1]
        )
        
        # Step 3: Consult technical agent for diagnostics
        step3 = builder.add_step(
            name="Perform Technical Diagnostics",
            description="Use the technical agent to diagnose the issue",
            action={
                "type": "consult_technical_agent",
                "issue_description": "{issue_description}",
                "issue_type": "{issue_type}",
                "customer_info": "{customer_info}"
            },
            dependencies=[step2]
        )
        
        # Step 4: Lookup knowledge base for solutions
        step4 = builder.add_step(
            name="Find Potential Solutions",
            description="Search the knowledge base for potential solutions",
            action={
                "type": "consult_knowledge_agent",
                "query": "How to fix {issue_type} issue: {issue_description}"
            },
            dependencies=[step3]
        )
        
        # Step 5: Generate solution report
        step5 = builder.add_step(
            name="Generate Solution Report",
            description="Create a comprehensive solution report for the customer",
            action={
                "type": "llm_query",
                "prompt": "Generate a detailed technical support solution report for the customer. Include:\n1. Problem summary\n2. Diagnostic results\n3. Recommended solution\n4. Additional tips\n\nIssue: {issue_description}\nDiagnostic results: {diagnostic_results}\nPotential solutions: {potential_solutions}\nCustomer plan: {customer_info.plan}"
            },
            dependencies=[step4]
        )
        
        # Step 6: Update customer record
        step6 = builder.add_step(
            name="Update Customer Record",
            description="Update the customer record with the support case",
            action={
                "type": "update_customer_info",
                "operation": "update",
                "customer_id": "{customer_id}",
                "updates": {
                    "last_support_case": {
                        "issue": "{issue_description}",
                        "solution": "Provided technical support for {issue_type} issue",
                        "timestamp": "now"
                    }
                }
            },
            dependencies=[step5]
        )
        
        # Step 7: Generate technical report
        builder.add_step(
            name="Generate Support Report",
            description="Generate a detailed support case report",
            action={
                "type": "generate_report",
                "title": "Technical Support Case Report",
                "sections": [
                    "Customer Information",
                    "Issue Description",
                    "Diagnostic Results",
                    "Solution Provided",
                    "Case Resolution"
                ],
                "customer_info": "{customer_info}",
                "issue_info": {
                    "description": "{issue_description}",
                    "type": "{issue_type}",
                    "diagnostics": "{diagnostic_results}"
                },
                "solution": "{solution_report}"
            },
            dependencies=[step6]
        )
        
        # Build the procedure
        procedure_id = builder.build()
        self.procedures["technical_support"] = procedure_id
        
        print(f"Created technical support procedure (ID: {procedure_id})")
        return procedure_id

    async def process_query(self, customer_id: str, query: str):
        """
        Process a customer query through the multi-agent system.
        
        Args:
            customer_id: Customer identifier
            query: Customer's query
            
        Returns:
            Response from the system
        """
        print(f"\n=== Processing Query from Customer {customer_id} ===")
        print(f"Query: {query}")
        
        # Store query in working memory
        query_id = self.working_memory.add(
            content=query,
            memory_type="customer_query",
            metadata={"customer_id": customer_id}
        )
        
        # Start with intent analysis
        intent_agent = self.agents["intent_agent"]
        intent_result = await intent_agent.process(query)
        print(f"Intent Analysis: {intent_result.get('primary_intent', 'Unknown')}")
        
        # Store intent in working memory
        intent_id = self.working_memory.add(
            content=intent_result,
            memory_type="intent_analysis",
            metadata={"customer_id": customer_id, "query_id": query_id}
        )
        
        # Check for procedural intents
        procedural_intent = self._check_for_procedural_intent(intent_result, query)
        
        if procedural_intent:
            print(f"Detected procedural intent: {procedural_intent}")
            return await self._handle_procedural_intent(procedural_intent, customer_id, query, intent_result)
        
        # For non-procedural intents, use the multi-agent router
        print("Using multi-agent router for non-procedural intent")
        result = await self.router.process_message(
            query,
            context={"customer_id": customer_id, "intent": intent_result},
            session_id=self.session_id
        )
        
        # Store result in working memory
        self.working_memory.add(
            content=result,
            memory_type="agent_response",
            metadata={"customer_id": customer_id, "query_id": query_id}
        )
        
        # Extract the final response
        if result.get("final_result"):
            if isinstance(result["final_result"], dict) and "answer" in result["final_result"]:
                response = result["final_result"]["answer"]
            else:
                response = result["final_result"]
        else:
            response = "I'm sorry, but I couldn't process your request properly."
        
        print(f"Response: {response}")
        return response

    def _check_for_procedural_intent(self, intent_result: Dict, query: str) -> Optional[str]:
        """
        Check if the intent corresponds to a procedural workflow.
        
        Args:
            intent_result: Intent analysis result
            query: Original query
            
        Returns:
            Procedure type if procedural intent detected, None otherwise
        """
        # Extract primary intent
        primary_intent = intent_result.get("primary_intent", "").lower()
        
        # Check for onboarding intent
        if any(term in primary_intent for term in ["signup", "register", "onboard", "new account", "create account"]):
            return "customer_onboarding"
        
        # Check for technical support intent
        if any(term in primary_intent for term in ["technical", "support", "issue", "problem", "not working", "broken"]):
            return "technical_support"
        
        # If no procedural intent detected
        return None

    async def _handle_procedural_intent(
            self, 
            procedure_type: str, 
            customer_id: str, 
            query: str, 
            intent_result: Dict
    ) -> str:
        """
        Handle a procedural intent by executing the appropriate procedure.
        
        Args:
            procedure_type: Type of procedure to execute
            customer_id: Customer identifier
            query: Original query
            intent_result: Intent analysis result
            
        Returns:
            Response from procedure execution
        """
        # Check if we have this procedure
        if procedure_type not in self.procedures:
            if procedure_type == "customer_onboarding":
                await self.create_customer_onboarding_procedure()
            elif procedure_type == "technical_support":
                await self.create_technical_support_procedure()
            else:
                return "I don't have a procedure to handle that request."
        
        # Get procedure ID
        procedure_id = self.procedures[procedure_type]
        
        # Prepare inputs based on procedure type
        inputs = {
            "customer_id": customer_id
        }
        
        if procedure_type == "customer_onboarding":
            # Extract potential inputs from query
            # In a real system, this would use more sophisticated extraction
            inputs["selected_plan"] = "Basic Plan"  # Default
            if "premium" in query.lower():
                inputs["selected_plan"] = "Premium Plan"
            elif "enterprise" in query.lower():
                inputs["selected_plan"] = "Enterprise Plan"
                
        elif procedure_type == "technical_support":
            # Use the original query as the issue description
            inputs["issue_description"] = query
        
        # Execute the procedure via the procedural agent
        procedural_agent = self.agents["procedural_agent"]
        result = await procedural_agent.execute_procedure(procedure_id, inputs)
        
        # Check result
        if result["status"] == ProcedureStatus.COMPLETED.value:
            # Format a response based on procedure outputs
            if procedure_type == "customer_onboarding":
                response = f"Thank you for signing up! Your account has been created successfully. "
                response += f"We've sent a welcome email to your provided address with details about your {inputs['selected_plan']}."
            
            elif procedure_type == "technical_support":
                # Extract solution from procedure outputs
                if "solution_report" in result["outputs"]:
                    response = result["outputs"]["solution_report"]
                else:
                    response = "I've analyzed your technical issue and created a support case. "
                    response += "Our team will follow up with a solution shortly."
            
            else:
                response = f"Procedure completed successfully: {result['status']}"
        else:
            # Handle procedure failure
            response = f"I'm sorry, but I wasn't able to complete the requested process. "
            response += f"Error: {result.get('error', 'Unknown error')}"
        
        return response

    # Tool handlers for the procedural agent
    async def _handle_consult_knowledge(self, action: Dict, inputs: Dict) -> Dict:
        """Handle consulting the knowledge agent."""
        query = action.get("query", "")
        
        # Replace placeholders in query
        for key, value in inputs.items():
            query = query.replace(f"{{{key}}}", str(value))
        
        print(f"  [Tool] Consulting knowledge agent: {query}")
        
        try:
            # Process with knowledge agent
            knowledge_agent = self.agents["knowledge_agent"]
            result = await knowledge_agent.process(query)
            
            # Extract answer
            if isinstance(result, dict) and "answer" in result:
                answer = result["answer"]
            else:
                answer = str(result)
            
            # Determine output field name
            topic = action.get("topic", "info")
            output_field = action.get("output_field", f"{topic}_info")
            
            return {
                "status": "success",
                "outputs": {
                    output_field: answer,
                    "knowledge_result": result
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "outputs": {
                    "error_details": str(e)
                }
            }

    async def _handle_consult_technical(self, action: Dict, inputs: Dict) -> Dict:
        """Handle consulting the technical agent."""
        issue_description = action.get("issue_description", "")
        issue_type = action.get("issue_type", "unknown")
        customer_info = action.get("customer_info", {})
        
        # Replace placeholders
        for key, value in inputs.items():
            if isinstance(issue_description, str):
                issue_description = issue_description.replace(f"{{{key}}}", str(value))
        
        print(f"  [Tool] Consulting technical agent for {issue_type} issue")
        
        try:
            # Process with technical agent
            technical_agent = self.agents["technical_agent"]
            
            # Use diagnose_issue if available
            if hasattr(technical_agent, "diagnose_issue"):
                result = await technical_agent.diagnose_issue(
                    issue_description=issue_description,
                    system_info=customer_info
                )
            else:
                result = await technical_agent.process(issue_description)
            
            return {
                "status": "success",
                "outputs": {
                    "diagnostic_results": result,
                    "issue_type": issue_type
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "outputs": {
                    "error_details": str(e)
                }
            }

    async def _handle_store_in_memory(self, action: Dict, inputs: Dict) -> Dict:
        """Handle storing data in working memory."""
        content = action.get("content", "")
        memory_type = action.get("memory_type", "general")
        metadata = action.get("metadata", {})
        ttl = action.get("ttl")  # Optional TTL
        
        # Replace placeholders in content
        for key, value in inputs.items():
            if isinstance(content, str):
                content = content.replace(f"{{{key}}}", str(value))
            
            # Also replace placeholders in metadata values
            for meta_key, meta_value in metadata.items():
                if isinstance(meta_value, str):
                    metadata[meta_key] = meta_value.replace(f"{{{key}}}", str(value))
        
        print(f"  [Tool] Storing in memory: type={memory_type}")
        
        try:
            # Store in memory
            item_id = self.working_memory.add(
                content=content,
                memory_type=memory_type,
                metadata=metadata,
                ttl=ttl
            )
            
            return {
                "status": "success",
                "outputs": {
                    "memory_id": item_id,
                    "stored": True
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "outputs": {
                    "stored": False
                }
            }

    async def _handle_retrieve_from_memory(self, action: Dict, inputs: Dict) -> Dict:
        """Handle retrieving data from working memory."""
        memory_type = action.get("memory_type")
        query = action.get("query")
        metadata_key = action.get("metadata_key")
        metadata_value = action.get("metadata_value")
        
        # Replace placeholders in query and metadata
        if query and isinstance(query, str):
            for key, value in inputs.items():
                query = query.replace(f"{{{key}}}", str(value))
        
        if metadata_value and isinstance(metadata_value, str):
            for key, value in inputs.items():
                metadata_value = metadata_value.replace(f"{{{key}}}", str(value))
        
        print(f"  [Tool] Retrieving from memory: type={memory_type}")
        
        try:
            # Retrieve based on specified criteria
            if query:
                results = self.working_memory.search(query)
            elif memory_type and metadata_key and metadata_value:
                # Filter by type and metadata
                type_items = self.working_memory.get_by_type(memory_type)
                results = []
                for item in type_items:
                    item_id = next((id for id, stored_item in self.working_memory.items.items() 
                                    if stored_item.content == item), None)
                    if item_id:
                        stored_item = self.working_memory.items[item_id]
                        if stored_item.metadata.get(metadata_key) == metadata_value:
                            results.append(item)
            elif memory_type:
                results = self.working_memory.get_by_type(memory_type)
            elif metadata_key and metadata_value:
                results = self.working_memory.get_by_metadata(metadata_key, metadata_value)
            else:
                # Get recent items as fallback
                results = self.working_memory.get_recent()
            
            return {
                "status": "success",
                "outputs": {
                    "memory_results": results,
                    "result_count": len(results)
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "outputs": {
                    "result_count": 0
                }
            }

    async def _handle_check_customer_info(self, action: Dict, inputs: Dict) -> Dict:
        """Handle checking customer information."""
        customer_id = action.get("customer_id")
        email = action.get("email")
        
        # Replace placeholders
        if customer_id and isinstance(customer_id, str):
            for key, value in inputs.items():
                customer_id = customer_id.replace(f"{{{key}}}", str(value))
                
        if email and isinstance(email, str):
            for key, value in inputs.items():
                email = email.replace(f"{{{key}}}", str(value))
        
        print(f"  [Tool] Checking customer info: id={customer_id}, email={email}")
        
        # Mock customer database
        customers = {
            "cust_123": {
                "id": "cust_123",
                "name": "John Smith",
                "email": "john.smith@example.com",
                "plan": "Premium Plan",
                "status": "active",
                "since": "2023-01-15"
            },
            "cust_456": {
                "id": "cust_456",
                "name": "Sarah Johnson",
                "email": "sarah.j@example.com",
                "plan": "Basic Plan",
                "status": "active",
                "since": "2024-03-10"
            }
        }
        
        # Search by ID or email
        customer_info = None
        customer_exists = False
        
        if customer_id and customer_id in customers:
            customer_info = customers[customer_id]
            customer_exists = True
        elif email:
            for cust in customers.values():
                if cust["email"].lower() == email.lower():
                    customer_info = cust
                    customer_exists = True
                    break
        
        return {
            "status": "success",
            "outputs": {
                "customer_exists": customer_exists,
                "customer_info": customer_info or {
                    "status": "not_found"
                }
            }
        }

    async def _handle_update_customer_info(self, action: Dict, inputs: Dict) -> Dict:
        """Handle updating customer information."""
        operation = action.get("operation", "update")
        customer_id = action.get("customer_id")
        customer_data = action.get("customer_data", {})
        updates = action.get("updates", {})
        
        # Replace placeholders in customer data
        if customer_data:
            for key, value in customer_data.items():
                if isinstance(value, str):
                    for input_key, input_value in inputs.items():
                        customer_data[key] = value.replace(f"{{{input_key}}}", str(input_value))
        
        # Replace placeholders in customer ID
        if customer_id and isinstance(customer_id, str):
            for key, value in inputs.items():
                customer_id = customer_id.replace(f"{{{key}}}", str(value))
        
        print(f"  [Tool] {operation.capitalize()} customer: id={customer_id}")
        
        if operation == "create":
            # Generate a new customer ID
            new_id = f"cust_{uuid.uuid4().hex[:8]}"
            print(f"  [Tool] Created new customer with ID: {new_id}")
            
            return {
                "status": "success",
                "outputs": {
                    "customer_id": new_id,
                    "created": True,
                    "customer_data": {
                        "id": new_id,
                        **customer_data,
                        "created_at": time.time()
                    }
                }
            }
        elif operation == "update":
            # In a real implementation, this would update a database
            print(f"  [Tool] Updated customer {customer_id} with: {updates}")
            
            return {
                "status": "success",
                "outputs": {
                    "customer_id": customer_id,
                    "updated": True,
                    "update_fields": list(updates.keys())
                }
            }
        else:
            return {
                "status": "error",
                "error": f"Unsupported operation: {operation}",
                "outputs": {
                    "supported_operations": ["create", "update"]
                }
            }

    async def _handle_generate_report(self, action: Dict, inputs: Dict) -> Dict:
        """Handle generating a structured report."""
        title = action.get("title", "Report")
        sections = action.get("sections", [])
        
        # Get additional data
        customer_info = action.get("customer_info", {})
        issue_info = action.get("issue_info", {})
        solution = action.get("solution", "")
        
        # Replace placeholders in strings
        if isinstance(solution, str):
            for key, value in inputs.items():
                solution = solution.replace(f"{{{key}}}", str(value))
        
        # Build report content
        report = f"# {title}\n\n"
        report += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add each section
        for section in sections:
            report += f"## {section}\n\n"
            
            if section == "Customer Information" and customer_info:
                if isinstance(customer_info, str) and customer_info.startswith("{"): 
                    # It's likely a placeholder that was replaced with JSON
                    try:
                        if isinstance(inputs.get(customer_info[1:-1]), dict):
                            customer_info = inputs.get(customer_info[1:-1])
                    except:
                        pass
                
                if isinstance(customer_info, dict):
                    for key, value in customer_info.items():
                        report += f"- **{key.replace('_', ' ').title()}**: {value}\n"
                else:
                    report += f"{customer_info}\n"
            
            elif section == "Issue Description" and issue_info:
                if isinstance(issue_info, dict):
                    if "description" in issue_info:
                        report += f"**Issue**: {issue_info['description']}\n\n"
                    if "type" in issue_info:
                        report += f"**Type**: {issue_info['type']}\n\n"
                else:
                    report += f"{issue_info}\n"
            
            elif section == "Diagnostic Results" and issue_info and "diagnostics" in issue_info:
                report += f"{issue_info['diagnostics']}\n"
            
            elif section == "Solution Provided" and solution:
                report += f"{solution}\n"
            
            elif section == "Plan Details" and "plan" in inputs:
                plan_info = inputs.get("plan_info", f"Details for {inputs['plan']}")
                report += f"{plan_info}\n"
            
            elif section == "Account Creation" and "customer_id" in inputs:
                report += f"Account successfully created with ID: {inputs['customer_id']}\n"
            
            elif section == "Next Steps":
                report += "1. Complete account setup\n"
                report += "2. Configure preferences\n"
                report += "3. Explore available features\n"
            
            elif section == "Case Resolution" and "solution" in inputs:
                report += f"Case resolved with solution: {inputs['solution']}\n"
            
            report += "\n"
        
        # Add report ID and timestamp at the end
        report_id = f"report_{uuid.uuid4().hex[:8]}"
        report += f"\nReport ID: {report_id}\n"
        
        print(f"  [Tool] Generated report: {title} (ID: {report_id})")
        
        return {
            "status": "success",
            "outputs": {
                "report_id": report_id,
                "report_content": report,
                "report_sections": sections
            }
        }

    # Mock technical tools
    async def _mock_network_diagnostic(self, issue_description: str, system_info: Dict = None) -> Dict:
        """Mock network diagnostic tool."""
        print(f"  [Technical Tool] Running network diagnostic")
        
        # Simulate diagnostic process
        await asyncio.sleep(0.5)
        
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
            },
            "recommendation": "The network connectivity is unstable with high latency and packet loss. This may be due to local network congestion or ISP issues."
        }

    async def _mock_hardware_diagnostic(self, issue_description: str, system_info: Dict = None) -> Dict:
        """Mock hardware diagnostic tool."""
        print(f"  [Technical Tool] Running hardware diagnostic")
        
        # Simulate diagnostic process
        await asyncio.sleep(0.7)
        
        return {
            "tool": "hardware_diagnostic",
            "status": "completed",
            "results": {
                "device_health": "good",
                "cpu_usage": 45,  # percentage
                "memory_usage": 60,  # percentage
                "disk_space": 75,  # percentage used
                "temperature": "normal"
            },
            "recommendation": "Hardware appears to be functioning normally. No hardware issues detected that would cause the reported problem."
        }

    async def _mock_system_check(self, issue_description: str, system_info: Dict = None) -> Dict:
        """Mock system check tool."""
        print(f"  [Technical Tool] Running system check")
        
        # Simulate diagnostic process
        await asyncio.sleep(0.6)
        
        return {
            "tool": "system_check",
            "status": "completed",
            "results": {
                "os_status": "healthy",
                "driver_status": "up-to-date",
                "firmware_version": "1.2.3",
                "required_updates": []
            },
            "recommendation": "All system components appear to be functioning correctly and are up to date."
        }


async def run_demo():
    """Run the multi-agent procedural demonstration."""
    # Get API keys from environment
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not openai_key and not anthropic_key:
        print("Warning: No API keys provided. Using mock LLM responses.")
    
    system = MultiAgentProceduralSystem(
        openai_key=openai_key,
        anthropic_key=anthropic_key
    )
    
    # Seed knowledge base
    await system.seed_knowledge()
    
    # Start the orchestrator
    await system.orchestrator.start()
    
    # Create procedures
    await system.create_customer_onboarding_procedure()
    await system.create_technical_support_procedure()
    
    # Example customer queries
    queries = [
        # Regular queries
        {"customer_id": "cust_123", "query": "Can you tell me about the Premium Plan?"},
        {"customer_id": "cust_123", "query": "How do I upgrade my current plan?"},
        
        # Procedural queries
        {"customer_id": "cust_456", "query": "I want to sign up for the Enterprise Plan"},
        {"customer_id": "cust_456", "query": "My internet connection keeps dropping every hour or so"}
    ]
    
    # Process each query
    for i, query_info in enumerate(queries):
        print(f"\n{'='*80}")
        print(f"Query {i+1}/{len(queries)}")
        
        response = await system.process_query(
            query_info["customer_id"],
            query_info["query"]
        )
        
        print(f"\nFinal Response: {response}")
        print(f"{'='*80}")
        
        # Small delay between queries
        if i < len(queries) - 1:
            await asyncio.sleep(1)
    
    # Stop the orchestrator
    await system.orchestrator.stop()
    
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    asyncio.run(run_demo())