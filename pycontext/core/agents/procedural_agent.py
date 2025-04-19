"""
pycontext/core/agents/procedural_agent.py

Procedural Agent implementation for PyContext.
"""
from typing import Dict, List, Optional, Any, Union
import json
import asyncio
import logging
import time
import uuid

from .base import BaseAgent
from ..mcp.protocol import ContextType
from ..memory.procedural_memory import (
    ProceduralMemory, Procedure, ProcedureStatus, ProcedureStep
)

logger = logging.getLogger(__name__)


class ProceduralAgent(BaseAgent):
    """
    Procedural Agent that manages and executes step-by-step procedures.
    Integrates with Procedural Memory to store, retrieve, and execute procedures.
    """

    def __init__(
            self,
            agent_id: str = None,
            agent_role: str = "procedural_agent",
            llm_client: Any = None,
            procedural_memory: Optional[ProceduralMemory] = None
    ):
        """
        Initialize the Procedural Agent.

        Args:
            agent_id: Optional unique identifier
            agent_role: Role of the agent
            llm_client: LLM client for generating responses
            procedural_memory: Procedural memory for storing procedures
        """
        super().__init__(
            agent_id=agent_id,
            agent_role=agent_role
        )
        self.llm_client = llm_client
        self.procedural_memory = procedural_memory or ProceduralMemory()
        
        # Register built-in action handlers
        self._register_action_handlers()
        
        # Keep track of currently running procedures
        self.running_procedures: Dict[str, asyncio.Task] = {}
        
        # Active procedure ID for the current session
        self.active_procedure_id: Optional[str] = None

    async def _load_role_prompt(self) -> str:
        """Load the procedural agent's system prompt."""
        return f"""You are a Procedural Agent specializing in executing step-by-step procedures.
Your role is to:
1. Execute procedures with precise timing and order
2. Track execution state and handle failures
3. Provide detailed execution reports
4. Explain procedures in a clear, step-by-step manner

When working with procedures:
1. Follow steps in the specified order
2. Handle dependencies between steps
3. Validate inputs before execution
4. Provide clear status updates
5. Document execution results

Always provide clear explanations of what steps are being taken and why.
"""

    async def process(self, input_text: str) -> Dict[str, Any]:
        """
        Process user input to execute or manage procedures.

        Args:
            input_text: User's message

        Returns:
            Dict containing response information
        """
        # Add user input to context
        await self.add_user_context(input_text, {"type": "procedural_input"})
        
        # If we have an active procedure, check if this is related to it
        if self.active_procedure_id:
            active_procedure = self.procedural_memory.get_procedure(self.active_procedure_id)
            if active_procedure and active_procedure.status == ProcedureStatus.RUNNING:
                # This might be input for the active procedure
                return await self._handle_procedure_input(input_text, active_procedure)
        
        # Analyze the input to determine what action to take
        action = await self._determine_action(input_text)
        
        # Handle action
        if action["type"] == "execute_procedure":
            return await self._execute_procedure(action["procedure_id"], action.get("inputs"))
        elif action["type"] == "create_procedure":
            return await self._create_procedure(input_text)
        elif action["type"] == "list_procedures":
            return await self._list_procedures(action.get("filter"))
        elif action["type"] == "explain_procedure":
            return await self._explain_procedure(action["procedure_id"])
        elif action["type"] == "cancel_procedure":
            return await self._cancel_procedure(action["procedure_id"])
        else:
            # Default response
            return {
                "response": "I'm not sure what you'd like me to do. Would you like me to execute a specific procedure, list available procedures, or create a new one?",
                "action": "unknown"
            }

    async def execute_procedure(
            self,
            procedure_id: str,
            inputs: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a procedure by ID.

        Args:
            procedure_id: ID of the procedure to execute
            inputs: Optional inputs for the procedure

        Returns:
            Execution result
        """
        # Check if procedure exists
        procedure = self.procedural_memory.get_procedure(procedure_id)
        if not procedure:
            return {
                "success": False,
                "error": f"Procedure {procedure_id} not found",
                "status": "not_found"
            }
        
        # Add execution to context
        await self.add_memory_context(
            content=f"Executing procedure '{procedure.name}' ({procedure_id})",
            relevance_score=0.9,
            metadata={
                "type": "procedure_execution",
                "procedure_id": procedure_id,
                "procedure_name": procedure.name
            }
        )
        
        # Set as active procedure
        self.active_procedure_id = procedure_id
        
        try:
            # Execute the procedure
            result = await self.procedural_memory.execute_procedure(procedure_id, inputs)
            
            # Add result to context
            execution_summary = self._format_execution_summary(result)
            self.context_manager.add_context(
                session_id=self.session_id,
                content=json.dumps(execution_summary),
                context_type=ContextType.AGENT,
                relevance_score=0.9,
                metadata={
                    "type": "procedure_result",
                    "procedure_id": procedure_id,
                    "status": result.status.value
                }
            )
            
            return {
                "success": result.status == ProcedureStatus.COMPLETED,
                "status": result.status.value,
                "result": execution_summary,
                "procedure_id": procedure_id
            }
        
        except Exception as e:
            logger.exception(f"Error executing procedure {procedure_id}: {str(e)}")
            
            # Add error to context
            self.context_manager.add_context(
                session_id=self.session_id,
                content=f"Error executing procedure: {str(e)}",
                context_type=ContextType.AGENT,
                relevance_score=0.9,
                metadata={
                    "type": "procedure_error",
                    "procedure_id": procedure_id
                }
            )
            
            return {
                "success": False,
                "error": str(e),
                "status": "error",
                "procedure_id": procedure_id
            }
        finally:
            # Clear active procedure
            if self.active_procedure_id == procedure_id:
                self.active_procedure_id = None

    async def create_procedure(
            self,
            name: str,
            description: str,
            steps: List[Dict[str, Any]],
            tags: List[str] = None,
            metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create a new procedure.

        Args:
            name: Procedure name
            description: Procedure description
            steps: List of step definitions
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            Creation result
        """
        try:
            # Create a procedure builder
            builder = self.procedural_memory.create_procedure_builder()
            
            # Set basic properties
            builder.set_name(name)
            builder.set_description(description)
            
            # Add tags
            if tags:
                for tag in tags:
                    builder.add_tag(tag)
            
            # Add metadata
            if metadata:
                for key, value in metadata.items():
                    builder.set_metadata(key, value)
            
            # Add steps
            for i, step_def in enumerate(steps):
                step_id = f"step{i + 1}"
                dependencies = []
                
                # Convert dependency names to IDs if needed
                if "dependencies" in step_def:
                    for dep in step_def["dependencies"]:
                        if dep.startswith("step"):
                            dependencies.append(dep)
                        else:
                            # Find step ID by name
                            for j, prev_step in enumerate(steps[:i]):
                                if prev_step.get("name") == dep:
                                    dependencies.append(f"step{j + 1}")
                                    break
                
                builder.add_step(
                    name=step_def["name"],
                    description=step_def.get("description", ""),
                    action=step_def["action"],
                    step_id=step_id,
                    inputs=step_def.get("inputs"),
                    dependencies=dependencies,
                    max_retries=step_def.get("max_retries", 0),
                    retry_delay=step_def.get("retry_delay", 1.0),
                    timeout=step_def.get("timeout"),
                    metadata=step_def.get("metadata")
                )
            
            # Build the procedure
            procedure_id = builder.build()
            
            # Get the created procedure
            procedure = self.procedural_memory.get_procedure(procedure_id)
            
            # Add to context
            self.context_manager.add_context(
                session_id=self.session_id,
                content=f"Created procedure '{name}' with {len(steps)} steps",
                context_type=ContextType.AGENT,
                relevance_score=0.9,
                metadata={
                    "type": "procedure_creation",
                    "procedure_id": procedure_id,
                    "procedure_name": name
                }
            )
            
            return {
                "success": True,
                "procedure_id": procedure_id,
                "name": name,
                "description": description,
                "steps": len(steps)
            }
        
        except Exception as e:
            logger.exception(f"Error creating procedure: {str(e)}")
            
            return {
                "success": False,
                "error": str(e)
            }

    async def register_action_handler(
            self,
            action_type: str,
            handler: Any
    ) -> None:
        """
        Register a handler for a specific action type.

        Args:
            action_type: Type of action
            handler: Handler function for the action
        """
        self.procedural_memory.register_action_handler(action_type, handler)

    async def _determine_action(self, input_text: str) -> Dict[str, Any]:
        """
        Determine what action to take based on user input.

        Args:
            input_text: User's message

        Returns:
            Action to take
        """
        if self.llm_client:
            # Use LLM to determine action
            return await self._determine_action_with_llm(input_text)
        else:
            # Simple rule-based determination
            input_lower = input_text.lower()
            
            # Check for procedure execution
            if ("execute" in input_lower or "run" in input_lower) and "procedure" in input_lower:
                # Extract procedure name or ID
                # Simple extraction logic - in a real implementation this would be more sophisticated
                procedure_id = None
                for proc_id, proc in self.procedural_memory.procedures.items():
                    if proc.name.lower() in input_lower:
                        procedure_id = proc_id
                        break
                
                if procedure_id:
                    return {
                        "type": "execute_procedure",
                        "procedure_id": procedure_id
                    }
            
            # Check for procedure listing
            if "list" in input_lower and "procedure" in input_lower:
                filter_tag = None
                if "tagged" in input_lower:
                    # Simple tag extraction
                    parts = input_lower.split("tagged")
                    if len(parts) > 1:
                        tag_part = parts[1].strip()
                        filter_tag = tag_part.split(" ")[0] if tag_part else None
                
                return {
                    "type": "list_procedures",
                    "filter": {"tag": filter_tag} if filter_tag else None
                }
            
            # Check for procedure explanation
            if ("explain" in input_lower or "describe" in input_lower) and "procedure" in input_lower:
                procedure_id = None
                for proc_id, proc in self.procedural_memory.procedures.items():
                    if proc.name.lower() in input_lower:
                        procedure_id = proc_id
                        break
                
                if procedure_id:
                    return {
                        "type": "explain_procedure",
                        "procedure_id": procedure_id
                    }
            
            # Check for procedure creation
            if "create" in input_lower and "procedure" in input_lower:
                return {
                    "type": "create_procedure"
                }
            
            # Check for procedure cancellation
            if ("cancel" in input_lower or "stop" in input_lower) and "procedure" in input_lower:
                procedure_id = self.active_procedure_id
                for proc_id, proc in self.procedural_memory.procedures.items():
                    if proc.name.lower() in input_lower:
                        procedure_id = proc_id
                        break
                
                if procedure_id:
                    return {
                        "type": "cancel_procedure",
                        "procedure_id": procedure_id
                    }
            
            # Default to unknown action
            return {
                "type": "unknown"
            }

    async def _determine_action_with_llm(self, input_text: str) -> Dict[str, Any]:
        """
        Use LLM to determine what action to take.

        Args:
            input_text: User's message

        Returns:
            Action to take
        """
        # Get formatted context
        formatted_context = self.context_manager.get_formatted_context(self.session_id)
        
        # Create prompt for action determination
        prompt = f"""
Based on the user message, determine what action should be taken.
Available procedure IDs and names:
"""
        
        # Add available procedures
        for proc_id, proc in self.procedural_memory.procedures.items():
            prompt += f"- {proc.name} (ID: {proc_id})\n"
        
        prompt += f"""
Active procedure: {self.active_procedure_id or "None"}

User message: "{input_text}"

Determine the appropriate action and respond in JSON format with one of these structures:

For executing a procedure:
{{
    "type": "execute_procedure",
    "procedure_id": "procedure_id",
    "inputs": {{}}  # Optional inputs
}}

For listing procedures:
{{
    "type": "list_procedures",
    "filter": {{}}  # Optional filters like tag
}}

For explaining a procedure:
{{
    "type": "explain_procedure",
    "procedure_id": "procedure_id"
}}

For creating a procedure:
{{
    "type": "create_procedure"
}}

For cancelling a procedure:
{{
    "type": "cancel_procedure",
    "procedure_id": "procedure_id"
}}

For unknown actions:
{{
    "type": "unknown"
}}
"""
        
        # Process with LLM
        response = await self._process_with_llm(formatted_context, prompt)
        
        try:
            # Parse JSON response
            action = json.loads(response)
            return action
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error parsing LLM action response: {e}")
            # Default to unknown action
            return {"type": "unknown"}

    async def _handle_procedure_input(
            self,
            input_text: str,
            procedure: Procedure
    ) -> Dict[str, Any]:
        """
        Handle input for an active procedure.

        Args:
            input_text: User's message
            procedure: Active procedure

        Returns:
            Response information
        """
        # Add input to context
        self.context_manager.add_context(
            session_id=self.session_id,
            content=f"Procedure input: {input_text}",
            context_type=ContextType.AGENT,
            relevance_score=0.9,
            metadata={
                "type": "procedure_input",
                "procedure_id": procedure.id
            }
        )
        
        # Check if input is a control command
        input_lower = input_text.lower()
        
        if "cancel" in input_lower or "stop" in input_lower:
            return await self._cancel_procedure(procedure.id)
        
        if "status" in input_lower:
            return await self._get_procedure_status(procedure.id)
        
        # For other inputs, provide a status update
        return {
            "response": f"The procedure '{procedure.name}' is currently running. Current status: {procedure.status.value}.",
            "action": "status_update",
            "procedure_id": procedure.id,
            "status": procedure.status.value
        }

    async def _execute_procedure(
            self,
            procedure_id: str,
            inputs: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a procedure and format the response.

        Args:
            procedure_id: ID of the procedure to execute
            inputs: Optional inputs for the procedure

        Returns:
            Execution result with formatted response
        """
        # Execute the procedure
        result = await self.execute_procedure(procedure_id, inputs)
        
        # Generate a response
        if result["success"]:
            summary = result["result"]
            response = f"Successfully executed procedure '{summary['name']}'.\n"
            response += f"Status: {summary['status']}\n"
            
            if summary["steps"]:
                response += f"Steps completed: {summary['steps_completed']}/{summary['total_steps']}\n"
            
            if summary.get("outputs"):
                response += "\nOutputs:\n"
                for key, value in summary["outputs"].items():
                    response += f"- {key}: {value}\n"
        else:
            response = f"Failed to execute procedure. Error: {result.get('error', 'Unknown error')}"
        
        return {
            "success": result["success"],
            "response": response,
            "status": result.get("status"),
            "procedure_id": procedure_id,
            "execution_result": result.get("result")
        }

    async def _create_procedure(self, input_text: str) -> Dict[str, Any]:
        """
        Handle procedure creation request.

        Args:
            input_text: User's message

        Returns:
            Creation result
        """
        if not self.llm_client:
            return {
                "success": False,
                "response": "Procedure creation requires an LLM. Please provide an LLM client.",
                "action": "create_procedure_failed"
            }
        
        # Get formatted context
        formatted_context = self.context_manager.get_formatted_context(self.session_id)
        
        # Create prompt for procedure creation
        prompt = f"""
Create a procedure based on the user's description:

User message: "{input_text}"

Create a procedure with appropriate steps, dependencies, and actions.
Respond in this JSON format:

{{
    "name": "Procedure Name",
    "description": "Procedure description",
    "tags": ["tag1", "tag2"],
    "steps": [
        {{
            "name": "Step 1",
            "description": "Step description",
            "action": {{
                "type": "action_type",
                "parameters": {{}}
            }},
            "inputs": {{}},
            "dependencies": [],
            "max_retries": 0
        }},
        // More steps...
    ],
    "metadata": {{}}
}}

Available action types: "llm_query", "tool_execution", "input_validation", "output_transformation"
"""
        
        # Process with LLM
        response = await self._process_with_llm(formatted_context, prompt)
        
        try:
            # Parse JSON response
            procedure_def = json.loads(response)
            
            # Create the procedure
            result = await self.create_procedure(
                name=procedure_def["name"],
                description=procedure_def["description"],
                steps=procedure_def["steps"],
                tags=procedure_def.get("tags", []),
                metadata=procedure_def.get("metadata", {})
            )
            
            if result["success"]:
                response_text = f"Successfully created procedure '{procedure_def['name']}' with {len(procedure_def['steps'])} steps."
            else:
                response_text = f"Failed to create procedure. Error: {result.get('error', 'Unknown error')}"
            
            return {
                "success": result["success"],
                "response": response_text,
                "procedure_id": result.get("procedure_id"),
                "procedure_def": procedure_def
            }
        
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error creating procedure: {e}")
            
            return {
                "success": False,
                "response": f"Failed to create procedure. Error: {str(e)}",
                "action": "create_procedure_failed"
            }

    async def _list_procedures(self, filter_dict: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        List available procedures.

        Args:
            filter_dict: Optional filter criteria

        Returns:
            List of procedures
        """
        procedures = []
        
        # Apply filters if specified
        if filter_dict and "tag" in filter_dict:
            tag = filter_dict["tag"]
            procedures = self.procedural_memory.get_procedures_by_tag(tag)
        else:
            procedures = list(self.procedural_memory.procedures.values())
        
        # Format the response
        if procedures:
            response = "Available procedures:\n\n"
            
            for proc in procedures:
                response += f"- {proc.name} (ID: {proc.id})\n"
                response += f"  Description: {proc.description}\n"
                if proc.tags:
                    response += f"  Tags: {', '.join(proc.tags)}\n"
                response += f"  Steps: {len(proc.steps)}\n"
                response += f"  Status: {proc.status.value}\n\n"
        else:
            if filter_dict and "tag" in filter_dict:
                response = f"No procedures found with tag '{filter_dict['tag']}'."
            else:
                response = "No procedures available. You can create a new procedure."
        
        return {
            "success": True,
            "response": response,
            "procedures": [
                {
                    "id": proc.id,
                    "name": proc.name,
                    "description": proc.description,
                    "tags": proc.tags,
                    "steps": len(proc.steps),
                    "status": proc.status.value
                }
                for proc in procedures
            ],
            "action": "list_procedures"
        }

    async def _explain_procedure(self, procedure_id: str) -> Dict[str, Any]:
        """
        Explain a procedure in detail.

        Args:
            procedure_id: ID of the procedure to explain

        Returns:
            Procedure explanation
        """
        procedure = self.procedural_memory.get_procedure(procedure_id)
        
        if not procedure:
            return {
                "success": False,
                "response": f"Procedure with ID '{procedure_id}' not found.",
                "action": "explain_procedure_failed"
            }
        
        # Create explanation
        response = f"## Procedure: {procedure.name}\n\n"
        response += f"**Description**: {procedure.description}\n\n"
        
        if procedure.tags:
            response += f"**Tags**: {', '.join(procedure.tags)}\n\n"
        
        response += f"**Status**: {procedure.status.value}\n\n"
        
        if procedure.inputs:
            response += "**Required Inputs**:\n"
            for key, value in procedure.inputs.items():
                if value is None:
                    response += f"- {key}\n"
                else:
                    response += f"- {key} (default: {value})\n"
            response += "\n"
        
        response += "**Steps**:\n\n"
        
        # Sort steps according to execution order
        ordered_steps = []
        if procedure.execution_order:
            for step_id in procedure.execution_order:
                if step_id in procedure.steps:
                    ordered_steps.append(procedure.steps[step_id])
        else:
            ordered_steps = list(procedure.steps.values())
        
        for i, step in enumerate(ordered_steps):
            response += f"{i+1}. **{step.name}**\n"
            response += f"   - Description: {step.description}\n"
            response += f"   - Action: {step.action.get('type', 'unknown')}\n"
            
            if step.dependencies:
                depends_on = []
                for dep_id in step.dependencies:
                    if dep_id in procedure.steps:
                        depends_on.append(procedure.steps[dep_id].name)
                if depends_on:
                    response += f"   - Depends on: {', '.join(depends_on)}\n"
            
            response += "\n"
        
        return {
            "success": True,
            "response": response,
            "procedure": {
                "id": procedure.id,
                "name": procedure.name,
                "description": procedure.description,
                "steps": [
                    {
                        "id": step.id,
                        "name": step.name,
                        "description": step.description,
                        "action": step.action,
                        "dependencies": step.dependencies
                    }
                    for step in ordered_steps
                ]
            },
            "action": "explain_procedure"
        }

    async def _cancel_procedure(self, procedure_id: str) -> Dict[str, Any]:
        """
        Cancel an executing procedure.

        Args:
            procedure_id: ID of the procedure to cancel

        Returns:
            Cancellation result
        """
        procedure = self.procedural_memory.get_procedure(procedure_id)
        
        if not procedure:
            return {
                "success": False,
                "response": f"Procedure with ID '{procedure_id}' not found.",
                "action": "cancel_procedure_failed"
            }
        
        # Check if procedure is running
        if procedure.status != ProcedureStatus.RUNNING:
            return {
                "success": False,
                "response": f"Cannot cancel procedure '{procedure.name}' because it is not running (current status: {procedure.status.value}).",
                "action": "cancel_procedure_failed"
            }
        
        # Cancel the procedure
        # In a real implementation, this would involve more complex logic to gracefully stop execution
        procedure.status = ProcedureStatus.CANCELED
        self.procedural_memory.update_procedure(procedure)
        
        # Clear active procedure if it was the one being cancelled
        if self.active_procedure_id == procedure_id:
            self.active_procedure_id = None
        
        return {
            "success": True,
            "response": f"Successfully cancelled procedure '{procedure.name}'.",
            "procedure_id": procedure_id,
            "action": "cancel_procedure"
        }

    async def _get_procedure_status(self, procedure_id: str) -> Dict[str, Any]:
        """
        Get the status of a procedure.

        Args:
            procedure_id: ID of the procedure

        Returns:
            Status information
        """
        procedure = self.procedural_memory.get_procedure(procedure_id)
        
        if not procedure:
            return {
                "success": False,
                "response": f"Procedure with ID '{procedure_id}' not found.",
                "action": "get_status_failed"
            }
        
        # Count completed steps
        completed_steps = sum(1 for step in procedure.steps.values() if step.status == StepStatus.COMPLETED)
        total_steps = len(procedure.steps)
        
        # Create status report
        response = f"Status of procedure '{procedure.name}':\n\n"
        response += f"- Overall Status: {procedure.status.value}\n"
        response += f"- Progress: {completed_steps}/{total_steps} steps completed"
        
        if procedure.current_step_id and procedure.current_step_id in procedure.steps:
            current_step = procedure.steps[procedure.current_step_id]
            response += f"\n- Current Step: {current_step.name} ({current_step.status.value})"
        
        if procedure.error:
            response += f"\n- Error: {procedure.error}"
        
        return {
            "success": True,
            "response": response,
            "status": procedure.status.value,
            "progress": {
                "completed": completed_steps,
                "total": total_steps
            },
            "procedure_id": procedure_id,
            "action": "get_procedure_status"
        }

    def _format_execution_summary(self, procedure: Procedure) -> Dict[str, Any]:
        """
        Format procedure execution summary.

        Args:
            procedure: Executed procedure

        Returns:
            Formatted summary
        """
        # Count completed steps
        completed_steps = sum(1 for step in procedure.steps.values() if step.status == StepStatus.COMPLETED)
        
        # Format step results
        step_results = []
        for step_id, step in procedure.steps.items():
            step_results.append({
                "id": step_id,
                "name": step.name,
                "status": step.status.value,
                "duration": step.duration,
                "result": step.result,
                "error": step.error
            })
        
        return {
            "id": procedure.id,
            "name": procedure.name,
            "status": procedure.status.value,
            "steps_completed": completed_steps,
            "total_steps": len(procedure.steps),
            "steps": step_results,
            "outputs": procedure.outputs,
            "error": procedure.error,
            "execution_time": sum(step.duration for step in procedure.steps.values() if step.completed_at)
        }

    def _register_action_handlers(self) -> None:
        """Register built-in action handlers."""
        self.procedural_memory.register_action_handler("llm_query", self._handle_llm_query)
        self.procedural_memory.register_action_handler("tool_execution", self._handle_tool_execution)
        self.procedural_memory.register_action_handler("input_validation", self._handle_input_validation)
        self.procedural_memory.register_action_handler("output_transformation", self._handle_output_transformation)
        self.procedural_memory.register_action_handler("conditional", self._handle_conditional)
        self.procedural_memory.register_action_handler("wait", self._handle_wait)

    async def _handle_llm_query(self, action: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle LLM query action.

        Args:
            action: Action configuration
            inputs: Action inputs

        Returns:
            Action result
        """
        if not self.llm_client:
            return {
                "status": "error",
                "error": "No LLM client available",
                "outputs": {}
            }
        
        prompt = action.get("prompt", "")
        
        # Replace placeholders in prompt
        for key, value in inputs.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        
        try:
            # Get formatted context
            formatted_context = self.context_manager.get_formatted_context(self.session_id)
            
            # Process with LLM
            response = await self._process_with_llm(formatted_context, prompt)
            
            # Extract structured data if specified
            if action.get("extract_json", False):
                try:
                    # Extract JSON from response
                    json_start = response.find("{")
                    json_end = response.rfind("}") + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        structured_data = json.loads(json_str)
                        return {
                            "status": "success",
                            "outputs": {
                                "response": response,
                                "structured_data": structured_data
                            }
                        }
                except Exception:
                    # If JSON extraction fails, return the raw response
                    pass
            
            return {
                "status": "success",
                "outputs": {
                    "response": response
                }
            }
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "outputs": {}
            }
        