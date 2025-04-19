"""
pycontext/core/memory/procedural_memory.py

Procedural Memory implementation for PyContext.
"""
from typing import Dict, List, Optional, Any, Union, Callable, Set
import time
import json
import uuid
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ProcedureStatus(Enum):
    """Status of a procedure in procedural memory."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class StepStatus(Enum):
    """Status of a single step in a procedure."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProcedureStep:
    """A single step in a procedure."""
    id: str
    name: str
    description: str
    action: Dict[str, Any]  # Action configuration
    status: StepStatus = StepStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # IDs of steps this step depends on
    max_retries: int = 0
    retry_count: int = 0
    retry_delay: float = 1.0  # seconds
    timeout: Optional[float] = None  # seconds
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Calculate the step duration in seconds."""
        if self.started_at is None:
            return 0

        end_time = self.completed_at or time.time()
        return end_time - self.started_at


@dataclass
class Procedure:
    """A procedure in procedural memory."""
    id: str
    name: str
    description: str
    steps: Dict[str, ProcedureStep]
    status: ProcedureStatus = ProcedureStatus.PENDING
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_executed_at: Optional[float] = None
    timeout: Optional[float] = None  # seconds
    execution_order: List[str] = field(default_factory=list)  # Step IDs in execution order
    step_dependencies: Dict[str, List[str]] = field(default_factory=dict)  # step_id -> dependent step IDs
    inputs: Dict[str, Any] = field(default_factory=dict)  # Procedure-level inputs
    outputs: Dict[str, Any] = field(default_factory=dict)  # Procedure-level outputs
    current_step_id: Optional[str] = None
    error: Optional[str] = None
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": {step_id: self._step_to_dict(step) for step_id, step in self.steps.items()},
            "status": self.status.value,
            "version": self.version,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_executed_at": self.last_executed_at,
            "timeout": self.timeout,
            "execution_order": self.execution_order,
            "step_dependencies": self.step_dependencies,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "current_step_id": self.current_step_id,
            "error": self.error,
            "retry_policy": self.retry_policy,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Procedure':
        """Create from dictionary representation."""
        steps = {
            step_id: cls._dict_to_step(step_data)
            for step_id, step_data in data.get("steps", {}).items()
        }

        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            steps=steps,
            status=ProcedureStatus(data["status"]),
            version=data.get("version", "1.0.0"),
            tags=data.get("tags", []),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            last_executed_at=data.get("last_executed_at"),
            timeout=data.get("timeout"),
            execution_order=data.get("execution_order", []),
            step_dependencies=data.get("step_dependencies", {}),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            current_step_id=data.get("current_step_id"),
            error=data.get("error"),
            retry_policy=data.get("retry_policy", {}),
            metadata=data.get("metadata", {})
        )

    @staticmethod
    def _step_to_dict(step: ProcedureStep) -> Dict[str, Any]:
        """Convert a procedure step to dictionary."""
        return {
            "id": step.id,
            "name": step.name,
            "description": step.description,
            "action": step.action,
            "status": step.status.value,
            "result": step.result,
            "error": step.error,
            "inputs": step.inputs,
            "outputs": step.outputs,
            "dependencies": step.dependencies,
            "max_retries": step.max_retries,
            "retry_count": step.retry_count,
            "retry_delay": step.retry_delay,
            "timeout": step.timeout,
            "started_at": step.started_at,
            "completed_at": step.completed_at,
            "metadata": step.metadata
        }

    @classmethod
    def _dict_to_step(cls, data: Dict[str, Any]) -> ProcedureStep:
        """Create a procedure step from dictionary."""
        return ProcedureStep(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            action=data["action"],
            status=StepStatus(data["status"]),
            result=data.get("result"),
            error=data.get("error"),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            dependencies=data.get("dependencies", []),
            max_retries=data.get("max_retries", 0),
            retry_count=data.get("retry_count", 0),
            retry_delay=data.get("retry_delay", 1.0),
            timeout=data.get("timeout"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            metadata=data.get("metadata", {})
        )

    def get_next_steps(self) -> List[str]:
        """Get IDs of steps that are ready to execute."""
        ready_steps = []

        # Check each pending step
        for step_id, step in self.steps.items():
            if step.status != StepStatus.PENDING:
                continue

            # Check if all dependencies are completed
            dependencies_met = True
            for dep_id in step.dependencies:
                if dep_id not in self.steps:
                    dependencies_met = False
                    break
                
                dep_step = self.steps[dep_id]
                if dep_step.status != StepStatus.COMPLETED:
                    dependencies_met = False
                    break

            if dependencies_met:
                ready_steps.append(step_id)

        # If we have specific execution order, respect it
        if self.execution_order:
            # Filter to only include steps in the ready list
            ordered_ready_steps = [
                step_id for step_id in self.execution_order 
                if step_id in ready_steps
            ]
            return ordered_ready_steps

        return ready_steps


class ProcedureExecutor:
    """Executes procedures and tracks their execution state."""

    def __init__(self, action_handlers: Dict[str, Callable] = None):
        """
        Initialize the procedure executor.

        Args:
            action_handlers: Dictionary mapping action types to handler functions
        """
        self.action_handlers = action_handlers or {}
        self.running_procedures: Dict[str, asyncio.Task] = {}

    async def execute_procedure(
        self, 
        procedure: Procedure,
        inputs: Dict[str, Any] = None
    ) -> Procedure:
        """
        Execute a procedure.

        Args:
            procedure: Procedure to execute
            inputs: Optional procedure inputs

        Returns:
            Updated procedure with results
        """
        # Update procedure state
        procedure.status = ProcedureStatus.RUNNING
        procedure.last_executed_at = time.time()
        procedure.error = None
        
        if inputs:
            procedure.inputs.update(inputs)

        try:
            # Set up timeout if specified
            if procedure.timeout:
                return await asyncio.wait_for(
                    self._execute_procedure_steps(procedure),
                    timeout=procedure.timeout
                )
            else:
                return await self._execute_procedure_steps(procedure)
                
        except asyncio.TimeoutError:
            procedure.status = ProcedureStatus.FAILED
            procedure.error = f"Procedure timed out after {procedure.timeout} seconds"
            return procedure
        except Exception as e:
            procedure.status = ProcedureStatus.FAILED
            procedure.error = str(e)
            logger.exception(f"Error executing procedure {procedure.id}: {e}")
            return procedure

    async def _execute_procedure_steps(self, procedure: Procedure) -> Procedure:
        """
        Execute the steps of a procedure in the correct order.

        Args:
            procedure: Procedure to execute

        Returns:
            Updated procedure with results
        """
        # Initialize execution state
        remaining_steps = set(procedure.steps.keys())
        completed_steps = set()
        failed = False

        # Continue until all steps are processed or a failure occurs
        while remaining_steps and not failed:
            # Get next executable steps
            next_steps = procedure.get_next_steps()
            
            if not next_steps:
                # If there are remaining steps but none are ready, we have a dependency issue
                if len(completed_steps) < len(procedure.steps):
                    procedure.status = ProcedureStatus.FAILED
                    procedure.error = "Unable to execute all steps due to dependency issues"
                    return procedure
                break

            # Execute ready steps (in parallel if possible)
            step_executions = []
            for step_id in next_steps:
                step = procedure.steps[step_id]
                
                # Update step inputs from previous step outputs if needed
                self._update_step_inputs(procedure, step)
                
                # Start step execution
                step_executions.append(self._execute_step(procedure, step))

            # Wait for all current steps to complete
            step_results = await asyncio.gather(*step_executions, return_exceptions=True)
            
            # Process results
            for i, step_id in enumerate(next_steps):
                step = procedure.steps[step_id]
                result = step_results[i]
                
                if isinstance(result, Exception):
                    step.status = StepStatus.FAILED
                    step.error = str(result)
                    
                    # Check retry policy
                    if step.retry_count < step.max_retries:
                        step.retry_count += 1
                        step.status = StepStatus.PENDING
                        logger.info(f"Retrying step {step_id} ({step.retry_count}/{step.max_retries})")
                        await asyncio.sleep(step.retry_delay)
                    else:
                        failed = True
                        break
                else:
                    # Step completed successfully
                    remaining_steps.remove(step_id)
                    completed_steps.add(step_id)

            if failed:
                procedure.status = ProcedureStatus.FAILED
                break

        # Update procedure status
        if not failed and not remaining_steps:
            procedure.status = ProcedureStatus.COMPLETED

        # Update procedure outputs from final step outputs
        self._update_procedure_outputs(procedure)

        return procedure

    async def _execute_step(self, procedure: Procedure, step: ProcedureStep) -> Dict[str, Any]:
        """
        Execute a single procedure step.

        Args:
            procedure: Parent procedure
            step: Step to execute

        Returns:
            Step execution result
        """
        # Update step state
        step.status = StepStatus.RUNNING
        step.started_at = time.time()
        step.error = None
        
        # Update current step in procedure
        procedure.current_step_id = step.id

        try:
            # Get the appropriate action handler
            action_type = step.action.get("type")
            handler = self.action_handlers.get(action_type)
            
            if not handler:
                raise ValueError(f"No handler found for action type: {action_type}")

            # Execute the action
            if step.timeout:
                result = await asyncio.wait_for(
                    handler(step.action, step.inputs),
                    timeout=step.timeout
                )
            else:
                result = await handler(step.action, step.inputs)

            # Update step with result
            step.result = result
            step.outputs = result.get("outputs", {})
            step.status = StepStatus.COMPLETED
            
            return result
        except asyncio.TimeoutError:
            step.status = StepStatus.FAILED
            step.error = f"Step timed out after {step.timeout} seconds"
            raise
        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
            logger.exception(f"Error executing step {step.id}: {e}")
            raise
        finally:
            step.completed_at = time.time()

    def _update_step_inputs(self, procedure: Procedure, step: ProcedureStep) -> None:
        """
        Update step inputs from procedure inputs and previous step outputs.

        Args:
            procedure: Parent procedure
            step: Step to update
        """
        # Include procedure-level inputs
        for input_key, input_value in procedure.inputs.items():
            if input_key not in step.inputs:
                step.inputs[input_key] = input_value

        # Include outputs from dependency steps
        for dep_id in step.dependencies:
            if dep_id in procedure.steps:
                dep_step = procedure.steps[dep_id]
                if dep_step.status == StepStatus.COMPLETED:
                    for output_key, output_value in dep_step.outputs.items():
                        # Use explicit input mappings if defined
                        mapping_key = f"{dep_id}.{output_key}"
                        if mapping_key in step.inputs:
                            target_key = step.inputs[mapping_key]
                            step.inputs[target_key] = output_value
                        # Otherwise use the output key directly
                        elif output_key not in step.inputs:
                            step.inputs[output_key] = output_value

    def _update_procedure_outputs(self, procedure: Procedure) -> None:
        """
        Update procedure outputs from step outputs.

        Args:
            procedure: Procedure to update
        """
        # Get the final steps (those with no dependencies)
        final_steps = []
        for step_id, step in procedure.steps.items():
            if step.status == StepStatus.COMPLETED:
                # Check if this step is not a dependency for any other step
                is_dependency = False
                for deps in procedure.step_dependencies.values():
                    if step_id in deps:
                        is_dependency = True
                        break
                
                if not is_dependency:
                    final_steps.append(step)

        # If no final steps identified, use all completed steps
        if not final_steps:
            final_steps = [step for step in procedure.steps.values() 
                           if step.status == StepStatus.COMPLETED]

        # Update procedure outputs from final step outputs
        for step in final_steps:
            for output_key, output_value in step.outputs.items():
                # Only add to procedure outputs if not already present
                if output_key not in procedure.outputs:
                    procedure.outputs[output_key] = output_value


class ProceduralMemory:
    """
    Procedural memory for storing and executing step-by-step procedures.
    """

    def __init__(self):
        """Initialize procedural memory."""
        self.procedures: Dict[str, Procedure] = {}
        self.procedure_types: Dict[str, List[str]] = {}  # Type -> list of procedure IDs
        self.procedure_tags: Dict[str, List[str]] = {}  # Tag -> list of procedure IDs
        self.executor = ProcedureExecutor()

    def register_action_handler(self, action_type: str, handler: Callable) -> None:
        """
        Register a handler for a specific action type.

        Args:
            action_type: Type of action
            handler: Handler function for the action
        """
        self.executor.action_handlers[action_type] = handler

    def add_procedure(self, procedure: Procedure) -> str:
        """
        Add a procedure to memory.

        Args:
            procedure: Procedure to add

        Returns:
            Procedure ID
        """
        # Store procedure
        self.procedures[procedure.id] = procedure

        # Update indices
        procedure_type = procedure.metadata.get("type", "general")
        if procedure_type not in self.procedure_types:
            self.procedure_types[procedure_type] = []
        self.procedure_types[procedure_type].append(procedure.id)

        # Update tag index
        for tag in procedure.tags:
            if tag not in self.procedure_tags:
                self.procedure_tags[tag] = []
            self.procedure_tags[tag].append(procedure.id)

        return procedure.id

    def get_procedure(self, procedure_id: str) -> Optional[Procedure]:
        """
        Get a procedure by ID.

        Args:
            procedure_id: Procedure identifier

        Returns:
            Procedure if found, None otherwise
        """
        return self.procedures.get(procedure_id)

    def get_procedures_by_type(self, procedure_type: str) -> List[Procedure]:
        """
        Get procedures by type.

        Args:
            procedure_type: Procedure type

        Returns:
            List of procedures
        """
        procedure_ids = self.procedure_types.get(procedure_type, [])
        return [self.procedures[pid] for pid in procedure_ids if pid in self.procedures]

    def get_procedures_by_tag(self, tag: str) -> List[Procedure]:
        """
        Get procedures by tag.

        Args:
            tag: Procedure tag

        Returns:
            List of procedures
        """
        procedure_ids = self.procedure_tags.get(tag, [])
        return [self.procedures[pid] for pid in procedure_ids if pid in self.procedures]

    def update_procedure(self, procedure: Procedure) -> bool:
        """
        Update an existing procedure.

        Args:
            procedure: Updated procedure

        Returns:
            Whether the update was successful
        """
        if procedure.id not in self.procedures:
            return False

        # Get old procedure for updating indices
        old_procedure = self.procedures[procedure.id]

        # Update procedure
        procedure.updated_at = time.time()
        self.procedures[procedure.id] = procedure

        # Update type index if type changed
        old_type = old_procedure.metadata.get("type", "general")
        new_type = procedure.metadata.get("type", "general")

        if old_type != new_type:
            # Remove from old type
            if old_type in self.procedure_types and procedure.id in self.procedure_types[old_type]:
                self.procedure_types[old_type].remove(procedure.id)

            # Add to new type
            if new_type not in self.procedure_types:
                self.procedure_types[new_type] = []
            self.procedure_types[new_type].append(procedure.id)

        # Update tag index if tags changed
        old_tags = set(old_procedure.tags)
        new_tags = set(procedure.tags)

        # Tags that were removed
        for tag in old_tags - new_tags:
            if tag in self.procedure_tags and procedure.id in self.procedure_tags[tag]:
                self.procedure_tags[tag].remove(procedure.id)

        # Tags that were added
        for tag in new_tags - old_tags:
            if tag not in self.procedure_tags:
                self.procedure_tags[tag] = []
            self.procedure_tags[tag].append(procedure.id)

        return True

    def remove_procedure(self, procedure_id: str) -> bool:
        """
        Remove a procedure from memory.

        Args:
            procedure_id: Procedure identifier

        Returns:
            Whether the removal was successful
        """
        if procedure_id not in self.procedures:
            return False

        procedure = self.procedures[procedure_id]

        # Remove from type index
        procedure_type = procedure.metadata.get("type", "general")
        if procedure_type in self.procedure_types and procedure_id in self.procedure_types[procedure_type]:
            self.procedure_types[procedure_type].remove(procedure_id)

        # Remove from tag index
        for tag in procedure.tags:
            if tag in self.procedure_tags and procedure_id in self.procedure_tags[tag]:
                self.procedure_tags[tag].remove(procedure_id)

        # Remove procedure
        del self.procedures[procedure_id]

        return True

    async def execute_procedure(
        self,
        procedure_id: str,
        inputs: Dict[str, Any] = None
    ) -> Procedure:
        """
        Execute a procedure.

        Args:
            procedure_id: Procedure identifier
            inputs: Optional procedure inputs

        Returns:
            Updated procedure with results
        """
        procedure = self.get_procedure(procedure_id)
        if not procedure:
            raise ValueError(f"Procedure {procedure_id} not found")

        # Create a copy of the procedure for execution
        execution_copy = Procedure.from_dict(procedure.to_dict())

        # Execute procedure
        result = await self.executor.execute_procedure(execution_copy, inputs)

        # Update the stored procedure with execution results
        self.update_procedure(result)

        return result

    def create_procedure_builder(self) -> 'ProcedureBuilder':
        """
        Create a procedure builder.

        Returns:
            New procedure builder
        """
        return ProcedureBuilder(self)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "procedures": {
                proc_id: proc.to_dict() for proc_id, proc in self.procedures.items()
            },
            "procedure_types": self.procedure_types,
            "procedure_tags": self.procedure_tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProceduralMemory':
        """
        Create from dictionary representation.

        Args:
            data: Dictionary representation

        Returns:
            ProceduralMemory instance
        """
        memory = cls()

        # Load procedures
        for proc_id, proc_dict in data.get("procedures", {}).items():
            procedure = Procedure.from_dict(proc_dict)
            memory.procedures[proc_id] = procedure

        # Load indices
        memory.procedure_types = data.get("procedure_types", {})
        memory.procedure_tags = data.get("procedure_tags", {})

        return memory

    def save_to_file(self, filename: str) -> None:
        """
        Save procedural memory to a file.

        Args:
            filename: Path to the file
        """
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filename: str) -> 'ProceduralMemory':
        """
        Load procedural memory from a file.

        Args:
            filename: Path to the file

        Returns:
            ProceduralMemory instance
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class ProcedureBuilder:
    """Builder for creating procedures step by step."""

    def __init__(self, memory: ProceduralMemory):
        """
        Initialize the procedure builder.

        Args:
            memory: Procedural memory to add the procedure to
        """
        self.memory = memory
        self.procedure_id = str(uuid.uuid4())
        self.name = ""
        self.description = ""
        self.version = "1.0.0"
        self.tags: List[str] = []
        self.steps: Dict[str, ProcedureStep] = {}
        self.inputs: Dict[str, Any] = {}
        self.timeout: Optional[float] = None
        self.execution_order: List[str] = []
        self.step_dependencies: Dict[str, List[str]] = {}
        self.metadata: Dict[str, Any] = {}
        self.retry_policy: Dict[str, Any] = {}

    def set_name(self, name: str) -> 'ProcedureBuilder':
        """Set procedure name."""
        self.name = name
        return self

    def set_description(self, description: str) -> 'ProcedureBuilder':
        """Set procedure description."""
        self.description = description
        return self

    def set_version(self, version: str) -> 'ProcedureBuilder':
        """Set procedure version."""
        self.version = version
        return self

    def add_tag(self, tag: str) -> 'ProcedureBuilder':
        """Add a tag to the procedure."""
        if tag not in self.tags:
            self.tags.append(tag)
        return self

    def set_timeout(self, timeout: float) -> 'ProcedureBuilder':
        """Set procedure timeout in seconds."""
        self.timeout = timeout
        return self

    def add_input(self, name: str, value: Any = None) -> 'ProcedureBuilder':
        """Add an input to the procedure."""
        self.inputs[name] = value
        return self

    def set_metadata(self, key: str, value: Any) -> 'ProcedureBuilder':
        """Set metadata value."""
        self.metadata[key] = value
        return self

    def set_retry_policy(self, max_retries: int, retry_delay: float = 1.0) -> 'ProcedureBuilder':
        """Set retry policy for the procedure."""
        self.retry_policy = {
            "max_retries": max_retries,
            "retry_delay": retry_delay
        }
        return self

    def add_step(
        self,
        name: str,
        description: str,
        action: Dict[str, Any],
        step_id: Optional[str] = None,
        inputs: Dict[str, Any] = None,
        dependencies: List[str] = None,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        timeout: Optional[float] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Add a step to the procedure.

        Args:
            name: Step name
            description: Step description
            action: Action configuration
            step_id: Optional step ID
            inputs: Optional step inputs
            dependencies: Optional step dependencies
            max_retries: Maximum retry count
            retry_delay: Delay between retries
            timeout: Step timeout in seconds
            metadata: Step metadata

        Returns:
            Step ID
        """
        step_id = step_id or str(uuid.uuid4())

        step = ProcedureStep(
            id=step_id,
            name=name,
            description=description,
            action=action,
            inputs=inputs or {},
            dependencies=dependencies or [],
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
            metadata=metadata or {}
        )

        self.steps[step_id] = step
        self.execution_order.append(step_id)

        # Update step dependencies
        for dep_id in step.dependencies:
            if dep_id not in self.step_dependencies:
                self.step_dependencies[dep_id] = []
            self.step_dependencies[dep_id].append(step_id)

        return step_id

    def build(self) -> str:
        """
        Build the procedure and add it to memory.

        Returns:
            Procedure ID
        """
        procedure = Procedure(
            id=self.procedure_id,
            name=self.name,
            description=self.description,
            steps=self.steps,
            version=self.version,
            tags=self.tags,
            timeout=self.timeout,
            execution_order=self.execution_order,
            step_dependencies=self.step_dependencies,
            inputs=self.inputs,
            retry_policy=self.retry_policy,
            metadata=self.metadata
        )

        return self.memory.add_procedure(procedure)