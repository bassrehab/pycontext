"""
examples/procedural_agent/procedural_demo.py

Example demonstrating the Procedural Agent capabilities.
Shows how to create, execute, and manage complex procedures with dependencies,
error handling, and visualization.
"""
import asyncio
import json
import os
import logging
import time
from typing import Dict, List, Any, Optional
import uuid

from pycontext.core.agents.procedural_agent import ProceduralAgent
from pycontext.core.memory.procedural_memory import ProceduralMemory, ProcedureStatus, StepStatus
from pycontext.integrations.llm_providers.openai_provider import OpenAIProvider
from pycontext.integrations.llm_providers.anthropic_provider import AnthropicProvider

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProceduralDemoApp:
    """
    Demonstration application for the Procedural Agent.
    Shows different use cases and capabilities of procedural workflows.
    """

    def __init__(
            self,
            openai_key: Optional[str] = None,
            anthropic_key: Optional[str] = None
    ):
        """
        Initialize the demo application.

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

        # Initialize procedural memory
        self.memory = ProceduralMemory()

        # Create procedural agent
        self.agent = ProceduralAgent(
            agent_id="procedural_demo_agent",
            llm_client=self.primary_provider,
            procedural_memory=self.memory
        )

        # Register tool handlers
        self._register_tool_handlers()

        # Initialize session
        asyncio.run(self.agent.initialize_session())

        # Track created procedures
        self.procedures = {}

    def _register_tool_handlers(self):
        """Register custom tool handlers for the demo."""
        # Data processing tools
        self.memory.register_action_handler("fetch_data", self._handle_fetch_data)
        self.memory.register_action_handler("process_data", self._handle_process_data)
        self.memory.register_action_handler("validate_data", self._handle_validate_data)
        self.memory.register_action_handler("save_data", self._handle_save_data)
        
        # Notification tools
        self.memory.register_action_handler("send_notification", self._handle_send_notification)
        
        # User interaction tools
        self.memory.register_action_handler("get_user_input", self._handle_get_user_input)
        self.memory.register_action_handler("display_results", self._handle_display_results)

        # Error simulation tool (for testing error handling)
        self.memory.register_action_handler("simulate_error", self._handle_simulate_error)

    async def create_data_processing_procedure(self):
        """
        Create a data processing procedure programmatically.
        This demonstrates manual procedure creation with explicit dependencies.
        """
        print("\n=== Creating Data Processing Procedure ===")
        
        # Create procedure using builder
        builder = self.memory.create_procedure_builder()
        procedure_id = builder\
            .set_name("Data Processing Workflow")\
            .set_description("A workflow to fetch, process, validate, and save data")\
            .add_tag("data")\
            .add_tag("workflow")\
            .set_timeout(60)  # 60 second timeout for entire procedure
            
        # Add inputs with default values
        builder.add_input("data_source", "sample_data.csv")
        builder.add_input("processing_mode", "standard")
        builder.add_input("validation_threshold", 0.8)
            
        # Step 1: Fetch Data
        fetch_step = builder.add_step(
            name="Fetch Data",
            description="Fetch data from the specified source",
            action={
                "type": "fetch_data",
                "source": "{data_source}"  # Reference to procedure input
            },
            max_retries=2,
            retry_delay=1.0,
            timeout=10  # 10 second timeout for this step
        )
        
        # Step 2: Process Data (depends on Fetch Data)
        process_step = builder.add_step(
            name="Process Data",
            description="Process the fetched data",
            action={
                "type": "process_data",
                "mode": "{processing_mode}"
            },
            dependencies=[fetch_step],
            timeout=15
        )
        
        # Step 3: Validate Data (depends on Process Data)
        validate_step = builder.add_step(
            name="Validate Data",
            description="Validate the processed data",
            action={
                "type": "validate_data",
                "threshold": "{validation_threshold}"
            },
            dependencies=[process_step],
            timeout=5
        )
        
        # Step 4: Save Data (depends on Validate Data)
        save_step = builder.add_step(
            name="Save Data",
            description="Save the validated data",
            action={
                "type": "save_data",
                "format": "json"
            },
            dependencies=[validate_step],
            timeout=5
        )
        
        # Step 5: Send Notification (depends on Save Data)
        builder.add_step(
            name="Send Notification",
            description="Notify user of completion",
            action={
                "type": "send_notification",
                "message": "Data processing completed successfully!"
            },
            dependencies=[save_step],
            timeout=2
        )
        
        # Build the procedure
        procedure_id = builder.build()
        self.procedures["data_processing"] = procedure_id
        
        print(f"Created data processing procedure (ID: {procedure_id})")
        return procedure_id

    async def create_user_onboarding_procedure(self):
        """
        Create a user onboarding procedure using natural language.
        This demonstrates creating procedures through the LLM.
        """
        print("\n=== Creating User Onboarding Procedure ===")
        
        # Use the agent to create a procedure from natural language
        result = await self.agent.process(
            "Create a procedure for new user onboarding with these steps: "
            "1. Collect user information (name, email) "
            "2. Verify email address "
            "3. Set up user preferences "
            "4. Create user account "
            "5. Send welcome email"
        )
        
        if result.get("success", False) and "procedure_id" in result:
            procedure_id = result["procedure_id"]
            self.procedures["user_onboarding"] = procedure_id
            print(f"Created user onboarding procedure (ID: {procedure_id})")
            return procedure_id
        else:
            print("Failed to create user onboarding procedure")
            return None

    async def create_error_handling_procedure(self):
        """
        Create a procedure that demonstrates error handling and recovery.
        """
        print("\n=== Creating Error Handling Test Procedure ===")
        
        # Create procedure using builder
        builder = self.memory.create_procedure_builder()
        builder.set_name("Error Handling Test")
        builder.set_description("A procedure to demonstrate error handling and recovery")
        builder.add_tag("test")
        builder.add_tag("error_handling")
        
        # First step (will succeed)
        step1 = builder.add_step(
            name="Successful Step",
            description="This step will succeed",
            action={
                "type": "display_results",
                "message": "Step 1 executed successfully!"
            }
        )
        
        # Second step (will fail with retry)
        step2 = builder.add_step(
            name="Failing Step with Retry",
            description="This step will fail but has retry logic",
            action={
                "type": "simulate_error",
                "fail_count": 1,  # Fail once then succeed
                "error_message": "Simulated error for testing retry logic"
            },
            dependencies=[step1],
            max_retries=2,
            retry_delay=1.0
        )
        
        # Third step (depends on the second)
        builder.add_step(
            name="Recovery Step",
            description="This step runs after recovery from failure",
            action={
                "type": "display_results",
                "message": "Successfully recovered from error!"
            },
            dependencies=[step2]
        )
        
        # Build the procedure
        procedure_id = builder.build()
        self.procedures["error_handling"] = procedure_id
        
        print(f"Created error handling procedure (ID: {procedure_id})")
        return procedure_id

    async def create_decision_tree_procedure(self):
        """
        Create a procedure with branching logic based on conditions.
        """
        print("\n=== Creating Decision Tree Procedure ===")
        
        # Create procedure using builder
        builder = self.memory.create_procedure_builder()
        builder.set_name("Decision Tree Workflow")
        builder.set_description("A workflow with branching logic based on conditions")
        builder.add_tag("decision_tree")
        builder.add_tag("conditional")
        
        # Add inputs
        builder.add_input("user_type", None)
        builder.add_input("priority", "medium")
        
        # First step: Get user input
        step1 = builder.add_step(
            name="Get User Type",
            description="Get user type input if not provided",
            action={
                "type": "conditional",
                "conditions": [
                    {
                        "field": "user_type",
                        "operator": "eq",
                        "value": None,
                        "result": {
                            "action": "get_input"
                        }
                    }
                ],
                "default": {
                    "action": "skip"
                }
            }
        )
        
        # Step for getting input if needed
        step2a = builder.add_step(
            name="Request User Type",
            description="Ask the user for their user type",
            action={
                "type": "get_user_input",
                "prompt": "Please enter your user type (new, returning, or premium):",
                "output_field": "user_type"
            },
            dependencies=[step1]
        )
        
        # Branch for new users
        step3a = builder.add_step(
            name="New User Path",
            description="Actions for new users",
            action={
                "type": "conditional",
                "conditions": [
                    {
                        "field": "user_type",
                        "operator": "eq",
                        "value": "new",
                        "result": {
                            "action": "continue"
                        }
                    }
                ],
                "default": {
                    "action": "skip"
                }
            },
            dependencies=[step1, step2a]
        )
        
        # Branch for returning users
        step3b = builder.add_step(
            name="Returning User Path",
            description="Actions for returning users",
            action={
                "type": "conditional",
                "conditions": [
                    {
                        "field": "user_type",
                        "operator": "eq",
                        "value": "returning",
                        "result": {
                            "action": "continue"
                        }
                    }
                ],
                "default": {
                    "action": "skip"
                }
            },
            dependencies=[step1, step2a]
        )
        
        # Branch for premium users
        step3c = builder.add_step(
            name="Premium User Path",
            description="Actions for premium users",
            action={
                "type": "conditional",
                "conditions": [
                    {
                        "field": "user_type",
                        "operator": "eq",
                        "value": "premium",
                        "result": {
                            "action": "continue"
                        }
                    }
                ],
                "default": {
                    "action": "skip"
                }
            },
            dependencies=[step1, step2a]
        )
        
        # New user specific step
        step4a = builder.add_step(
            name="New User Action",
            description="Take action for new users",
            action={
                "type": "display_results",
                "message": "Welcome, new user! Here's information for getting started."
            },
            dependencies=[step3a]
        )
        
        # Returning user specific step
        step4b = builder.add_step(
            name="Returning User Action",
            description="Take action for returning users",
            action={
                "type": "display_results",
                "message": "Welcome back! Here's what's new since your last visit."
            },
            dependencies=[step3b]
        )
        
        # Premium user specific step
        step4c = builder.add_step(
            name="Premium User Action",
            description="Take action for premium users",
            action={
                "type": "display_results",
                "message": "Welcome, premium member! Here are your exclusive premium features."
            },
            dependencies=[step3c]
        )
        
        # Final step: common for all paths
        builder.add_step(
            name="Final Step",
            description="Common final step for all user types",
            action={
                "type": "display_results",
                "message": "Thank you for using our system!"
            },
            dependencies=[step4a, step4b, step4c]  # Depends on all path endpoints
        )
        
        # Build the procedure
        procedure_id = builder.build()
        self.procedures["decision_tree"] = procedure_id
        
        print(f"Created decision tree procedure (ID: {procedure_id})")
        return procedure_id

    async def execute_data_processing_procedure(self):
        """Execute the data processing procedure."""
        print("\n=== Executing Data Processing Procedure ===")
        
        procedure_id = self.procedures.get("data_processing")
        if not procedure_id:
            print("Error: Data processing procedure not found. Create it first.")
            return
        
        # Execute the procedure
        start_time = time.time()
        print("Starting execution...")
        
        # Define custom inputs
        inputs = {
            "data_source": "customer_transactions.csv",
            "processing_mode": "advanced",
            "validation_threshold": 0.9
        }
        
        # Execute via the agent to get detailed responses
        result = await self.agent.process(f"Execute the Data Processing Workflow with source={inputs['data_source']}, "
                                        f"mode={inputs['processing_mode']}, and "
                                        f"threshold={inputs['validation_threshold']}")
        
        execution_time = time.time() - start_time
        print(f"Execution completed in {execution_time:.2f} seconds")
        print(f"Result: {result['status'] if 'status' in result else 'Unknown'}")
        print(f"Response: {result['response']}")

    async def execute_error_handling_procedure(self):
        """Execute the error handling procedure."""
        print("\n=== Executing Error Handling Procedure ===")
        
        procedure_id = self.procedures.get("error_handling")
        if not procedure_id:
            print("Error: Error handling procedure not found. Create it first.")
            return
        
        # Execute the procedure
        start_time = time.time()
        print("Starting execution...")
        
        # Execute via the agent
        result = await self.agent.process(f"Execute the Error Handling Test")
        
        execution_time = time.time() - start_time
        print(f"Execution completed in {execution_time:.2f} seconds")
        print(f"Result: {result['status'] if 'status' in result else 'Unknown'}")
        print(f"Response: {result['response']}")

    async def execute_decision_tree_procedure(self):
        """Execute the decision tree procedure."""
        print("\n=== Executing Decision Tree Procedure ===")
        
        procedure_id = self.procedures.get("decision_tree")
        if not procedure_id:
            print("Error: Decision tree procedure not found. Create it first.")
            return
        
        # Execute the procedure three times with different inputs to show branching
        user_types = ["new", "returning", "premium"]
        
        for user_type in user_types:
            print(f"\nExecuting for user_type: {user_type}")
            start_time = time.time()
            
            # Execute via the agent
            result = await self.agent.process(f"Execute the Decision Tree Workflow with user_type={user_type}")
            
            execution_time = time.time() - start_time
            print(f"Execution completed in {execution_time:.2f} seconds")
            print(f"Response: {result['response']}")

    async def demonstrate_interactive_procedure(self):
        """Demonstrate interactive procedure execution with user input."""
        print("\n=== Interactive Procedure Demonstration ===")
        
        # Create a simple interactive procedure
        builder = self.memory.create_procedure_builder()
        procedure_id = builder\
            .set_name("Interactive Questionnaire")\
            .set_description("An interactive procedure that collects user information")\
            .add_tag("interactive")
            
        # First step: Ask for name
        step1 = builder.add_step(
            name="Ask for Name",
            description="Get the user's name",
            action={
                "type": "get_user_input",
                "prompt": "What is your name?",
                "output_field": "user_name"
            }
        )
        
        # Second step: Ask for experience level
        step2 = builder.add_step(
            name="Ask for Experience Level",
            description="Get the user's experience level",
            action={
                "type": "get_user_input",
                "prompt": "What is your experience level (beginner, intermediate, advanced)?",
                "output_field": "experience_level"
            },
            dependencies=[step1]
        )
        
        # Third step: Provide personalized response
        builder.add_step(
            name="Provide Personalized Response",
            description="Generate a personalized response based on user input",
            action={
                "type": "llm_query",
                "prompt": "Generate a friendly, personalized response for a user named {user_name} with {experience_level} experience level. Suggest appropriate resources for their level."
            },
            dependencies=[step2]
        )
        
        # Build and store the procedure
        procedure_id = builder.build()
        self.procedures["interactive"] = procedure_id
        
        print(f"Created interactive procedure (ID: {procedure_id})")
        
        # Execute the procedure
        print("\nStarting interactive procedure execution...")
        print("(Note: In a real application, this would collect actual user input)")
        
        # Mock answers for demonstration purposes
        self._mock_user_input = ["Alice", "intermediate"]
        
        # Execute via the agent
        result = await self.agent.process(f"Execute the Interactive Questionnaire")
        
        print(f"Response: {result['response']}")

    async def list_procedure_details(self):
        """List details of all created procedures."""
        print("\n=== Procedure Details ===")
        
        # Get a list of all procedures
        result = await self.agent.process("List available procedures")
        
        print(result["response"])

    async def explain_procedure_execution(self):
        """Explain how a procedure was executed with step-by-step details."""
        print("\n=== Explaining Procedure Execution ===")
        
        procedure_id = self.procedures.get("data_processing")
        if not procedure_id:
            print("Error: Data processing procedure not found. Create and execute it first.")
            return
        
        # Get procedure from memory
        procedure = self.memory.get_procedure(procedure_id)
        if not procedure:
            print(f"Error: Procedure {procedure_id} not found in memory.")
            return
        
        # Print execution details
        if procedure.status == ProcedureStatus.COMPLETED:
            print(f"Procedure '{procedure.name}' was executed successfully.")
            print(f"Execution started at: {time.ctime(procedure.last_executed_at)}")
            
            # Print step details
            for step_id, step in procedure.steps.items():
                step_status = step.status.value
                duration = step.duration if step.started_at else 0
                
                print(f"\nStep: {step.name}")
                print(f"  Status: {step_status}")
                if step.started_at:
                    print(f"  Started at: {time.ctime(step.started_at)}")
                if step.completed_at:
                    print(f"  Completed at: {time.ctime(step.completed_at)}")
                print(f"  Duration: {duration:.2f} seconds")
                
                if step.result:
                    print(f"  Results: {json.dumps(step.result, indent=2)}")
                if step.error:
                    print(f"  Error: {step.error}")
        else:
            print(f"Procedure '{procedure.name}' has not been fully executed. Status: {procedure.status.value}")

    # Tool Handlers
    async def _handle_fetch_data(self, action, inputs):
        """Handle fetch data action."""
        source = action.get("source", "default_source")
        print(f"  [Tool] Fetching data from {source}...")
        
        # Simulate data fetching
        await asyncio.sleep(1)
        
        return {
            "status": "success",
            "outputs": {
                "data": [
                    {"id": 1, "value": 10},
                    {"id": 2, "value": 20},
                    {"id": 3, "value": 30},
                ],
                "metadata": {"source": source, "records": 3}
            }
        }

    async def _handle_process_data(self, action, inputs):
        """Handle process data action."""
        mode = action.get("mode", "standard")
        data = inputs.get("data", [])
        
        print(f"  [Tool] Processing data in {mode} mode...")
        
        # Simulate data processing
        await asyncio.sleep(1.5)
        
        # Add calculated fields based on mode
        processed_data = []
        for item in data:
            processed_item = item.copy()
            
            if mode == "standard":
                processed_item["processed_value"] = item["value"] * 1.1
            elif mode == "advanced":
                processed_item["processed_value"] = item["value"] * 1.5
                processed_item["confidence"] = 0.95
            else:
                processed_item["processed_value"] = item["value"]
            
            processed_data.append(processed_item)
        
        return {
            "status": "success",
            "outputs": {
                "processed_data": processed_data,
                "processing_info": {"mode": mode, "items_processed": len(data)}
            }
        }

    async def _handle_validate_data(self, action, inputs):
        """Handle validate data action."""
        threshold = action.get("threshold", 0.8)
        processed_data = inputs.get("processed_data", [])
        
        print(f"  [Tool] Validating data with threshold {threshold}...")
        
        # Simulate validation
        await asyncio.sleep(0.5)
        
        # Validate each item
        valid_items = []
        invalid_items = []
        
        for item in processed_data:
            # Check if confidence available, or generate a random one
            confidence = item.get("confidence", 0.9)
            
            if confidence >= threshold:
                valid_items.append(item)
            else:
                invalid_items.append(item)
        
        validation_passed = len(invalid_items) == 0
        
        return {
            "status": "success",
            "outputs": {
                "valid_items": valid_items,
                "invalid_items": invalid_items,
                "validation_passed": validation_passed,
                "validation_rate": len(valid_items) / len(processed_data) if processed_data else 0
            }
        }

    async def _handle_save_data(self, action, inputs):
        """Handle save data action."""
        format = action.get("format", "json")
        valid_items = inputs.get("valid_items", [])
        
        print(f"  [Tool] Saving {len(valid_items)} items in {format} format...")
        
        # Simulate saving
        await asyncio.sleep(0.5)
        
        # Generate a filename
        filename = f"data_export_{uuid.uuid4().hex[:8]}.{format}"
        
        return {
            "status": "success",
            "outputs": {
                "filename": filename,
                "items_saved": len(valid_items),
                "format": format
            }
        }

    async def _handle_send_notification(self, action, inputs):
        """Handle send notification action."""
        message = action.get("message", "Notification message")
        print(f"  [Tool] Sending notification: {message}")
        
        # Simulate notification sending
        await asyncio.sleep(0.2)
        
        return {
            "status": "success",
            "outputs": {
                "notification_sent": True,
                "timestamp": time.time()
            }
        }

    async def _handle_get_user_input(self, action, inputs):
        """Handle get user input action."""
        prompt = action.get("prompt", "Please enter your input:")
        output_field = action.get("output_field", "user_input")
        
        print(f"  [Tool] Requesting user input: {prompt}")
        
        # In a real application, this would wait for actual user input
        # For demo purposes, simulate user input
        if hasattr(self, '_mock_user_input') and self._mock_user_input:
            user_input = self._mock_user_input.pop(0)
        else:
            # Default mock response
            user_input = "mock_response"
        
        print(f"  [Tool] Received user input: {user_input}")
        
        return {
            "status": "success",
            "outputs": {
                output_field: user_input
            }
        }

    async def _handle_display_results(self, action, inputs):
        """Handle display results action."""
        message = action.get("message", "")
        print(f"  [Tool] Displaying results: {message}")
        
        return {
            "status": "success",
            "outputs": {
                "displayed": True,
                "message": message
            }
        }

    async def _handle_simulate_error(self, action, inputs):
        """Handle simulate error action for testing error handling."""
        fail_count = action.get("fail_count", 1)
        error_message = action.get("error_message", "Simulated error")
        
        # Check if we've already been called and retry count
        call_count = inputs.get("_simulate_error_call_count", 0) + 1
        
        # Store call count in outputs
        outputs = {
            "_simulate_error_call_count": call_count
        }
        
        print(f"  [Tool] Simulated error handler (attempt {call_count}/{fail_count})")
        
        # Determine if we should fail
        if call_count <= fail_count:
            print(f"  [Tool] Simulating error: {error_message}")
            raise Exception(error_message)
        
        print(f"  [Tool] Error simulation complete, proceeding normally")
        
        return {
            "status": "success",
            "outputs": outputs
        }


async def run_demo():
    """Run the procedural agent demonstration."""
    # Get API keys from environment
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not openai_key and not anthropic_key:
        print("Warning: No API keys provided. Using mock LLM responses.")
    
    demo = ProceduralDemoApp(
        openai_key=openai_key,
        anthropic_key=anthropic_key
    )
    
    # Create example procedures
    await demo.create_data_processing_procedure()
    await demo.create_error_handling_procedure()
    await demo.create_decision_tree_procedure()
    
    # List created procedures
    await demo.list_procedure_details()
    
    # Execute procedures
    await demo.execute_data_processing_procedure()
    await demo.execute_error_handling_procedure()
    await demo.execute_decision_tree_procedure()
    
    # Demonstrate interactive procedures
    await demo.demonstrate_interactive_procedure()
    
    # Explain procedure execution details
    await demo.explain_procedure_execution()
    
    # Create a procedure using natural language (LLM-based)
    if demo.primary_provider:
        await demo.create_user_onboarding_procedure()
    
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    asyncio.run(run_demo())