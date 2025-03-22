"""
pycontext/core/coordination/orchestrator.py

Agent Orchestrator for coordinating multiple agents.
"""
from typing import Dict, List, Optional, Any, Union, Tuple
import asyncio
import json
import logging
import uuid
import time
from enum import Enum
from dataclasses import dataclass, field

from ..agents.base import BaseAgent
from ..context.manager import ContextManager
from ..mcp.protocol import ContextType

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a task in the agent orchestrator."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class TaskPriority(Enum):
    """Priority of a task in the agent orchestrator."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Task:
    """A task to be executed by an agent."""
    id: str
    agent_type: str
    input_data: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    parent_task_id: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    max_retries: int = 0
    retry_count: int = 0
    retry_delay: float = 1.0  # seconds
    context_id: Optional[str] = None

    @property
    def duration(self) -> float:
        """Calculate the task duration in seconds."""
        if self.started_at is None:
            return 0

        end_time = self.completed_at or time.time()
        return end_time - self.started_at


class AgentOrchestrator:
    """
    Coordinates multiple agents to solve complex tasks.
    Manages task dispatching, context sharing, and result aggregation.
    """

    def __init__(
            self,
            agents: Dict[str, BaseAgent],
            context_manager: Optional[ContextManager] = None,
            max_parallel_tasks: int = 5
    ):
        """
        Initialize the agent orchestrator.

        Args:
            agents: Dictionary mapping agent types to agent instances
            context_manager: Optional context manager for sharing context
            max_parallel_tasks: Maximum number of tasks to run in parallel
        """
        self.agents = agents
        self.context_manager = context_manager or ContextManager()
        self.max_parallel_tasks = max_parallel_tasks

        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[str] = []  # List of task IDs
        self.running_tasks: Dict[str, asyncio.Task] = {}

        # Event for notifying task completion
        self.task_completed_event = asyncio.Event()

    async def start(self) -> None:
        """Start the orchestrator and begin processing tasks."""
        logger.info("Starting agent orchestrator")
        await self._start_task_processor()

    async def stop(self) -> None:
        """Stop the orchestrator and cancel all running tasks."""
        logger.info("Stopping agent orchestrator")

        # Cancel all running tasks
        for task_id, task in self.running_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"Task {task_id} cancelled")

        self.running_tasks.clear()

    async def create_task(
            self,
            agent_type: str,
            input_data: Dict[str, Any],
            priority: TaskPriority = TaskPriority.NORMAL,
            parent_task_id: Optional[str] = None,
            context_id: Optional[str] = None,
            max_retries: int = 0
    ) -> str:
        """
        Create a new task for an agent to execute.

        Args:
            agent_type: Type of agent to execute the task
            input_data: Input data for the task
            priority: Task priority
            parent_task_id: Optional parent task ID for subtasks
            context_id: Optional context ID to use
            max_retries: Maximum number of retries on failure

        Returns:
            Task ID
        """
        # Check if agent type exists
        if agent_type not in self.agents:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Create task ID
        task_id = str(uuid.uuid4())

        # Create task
        task = Task(
            id=task_id,
            agent_type=agent_type,
            input_data=input_data,
            priority=priority,
            parent_task_id=parent_task_id,
            max_retries=max_retries,
            context_id=context_id
        )

        # Add to parent task's subtasks if applicable
        if parent_task_id and parent_task_id in self.tasks:
            self.tasks[parent_task_id].subtasks.append(task_id)

        # Store task
        self.tasks[task_id] = task

        # Add to queue based on priority
        self._add_to_queue(task_id)

        logger.info(f"Created task {task_id} for agent {agent_type}")

        return task_id

    async def create_composite_task(
            self,
            subtasks: List[Dict[str, Any]],
            aggregate_results: bool = True,
            sequence: bool = True
    ) -> str:
        """
        Create a composite task with multiple subtasks.

        Args:
            subtasks: List of subtask specifications
            aggregate_results: Whether to aggregate subtask results
            sequence: Whether to run subtasks in sequence or parallel

        Returns:
            Parent task ID
        """
        # Create parent task (handled by orchestrator itself)
        parent_task_id = str(uuid.uuid4())

        # Create parent task object
        parent_task = Task(
            id=parent_task_id,
            agent_type="orchestrator",
            input_data={
                "subtasks": subtasks,
                "aggregate_results": aggregate_results,
                "sequence": sequence
            },
            priority=TaskPriority.NORMAL
        )

        # Store parent task
        self.tasks[parent_task_id] = parent_task

        # Create context for the composite task if needed
        if parent_task.context_id is None:
            parent_task.context_id = self.context_manager.create_session("orchestrator")

        # Create subtasks
        for subtask_spec in subtasks:
            await self.create_task(
                agent_type=subtask_spec["agent_type"],
                input_data=subtask_spec["input_data"],
                priority=TaskPriority(subtask_spec.get("priority", TaskPriority.NORMAL.value)),
                parent_task_id=parent_task_id,
                context_id=parent_task.context_id,
                max_retries=subtask_spec.get("max_retries", 0)
            )

        # Add parent task to queue
        self._add_to_queue(parent_task_id)

        return parent_task_id

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task.

        Args:
            task_id: Task identifier

        Returns:
            Task status information
        """
        if task_id not in self.tasks:
            return {"error": "Task not found"}

        task = self.tasks[task_id]

        # Get subtask statuses
        subtask_statuses = []
        for subtask_id in task.subtasks:
            if subtask_id in self.tasks:
                subtask = self.tasks[subtask_id]
                subtask_statuses.append({
                    "id": subtask_id,
                    "status": subtask.status.value,
                    "agent_type": subtask.agent_type
                })

        return {
            "id": task.id,
            "status": task.status.value,
            "agent_type": task.agent_type,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "duration": task.duration,
            "subtasks": subtask_statuses,
            "result": task.result,
            "error": task.error
        }

    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Wait for a task to complete.

        Args:
            task_id: Task identifier
            timeout: Optional timeout in seconds

        Returns:
            Task result or status on timeout
        """
        if task_id not in self.tasks:
            return {"error": "Task not found"}

        task = self.tasks[task_id]

        # If task is already completed, return immediately
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            return self._get_task_result(task_id)

        # Wait for task to complete
        start_time = time.time()
        while True:
            # Check if task is completed
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                break

            # Check timeout
            if timeout is not None and time.time() - start_time > timeout:
                return {
                    "status": "timeout",
                    "task_id": task_id,
                    "current_status": task.status.value
                }

            # Wait for task completion event
            try:
                await asyncio.wait_for(
                    self.task_completed_event.wait(),
                    timeout=0.1 if timeout is None else min(0.1, timeout)
                )
                # Reset event for next notification
                self.task_completed_event.clear()
            except asyncio.TimeoutError:
                # Check if we've exceeded the overall timeout
                if timeout is not None and time.time() - start_time > timeout:
                    return {
                        "status": "timeout",
                        "task_id": task_id,
                        "current_status": task.status.value
                    }

        return self._get_task_result(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task and its subtasks.

        Args:
            task_id: Task identifier

        Returns:
            Whether the cancellation was successful
        """
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]

        # Cancel subtasks first
        for subtask_id in task.subtasks:
            await self.cancel_task(subtask_id)

        # Cancel running task if applicable
        if task_id in self.running_tasks and not self.running_tasks[task_id].done():
            self.running_tasks[task_id].cancel()
            try:
                await self.running_tasks[task_id]
            except asyncio.CancelledError:
                logger.info(f"Task {task_id} cancelled")

        # Update task status
        task.status = TaskStatus.CANCELED

        # Remove from queue if present
        if task_id in self.task_queue:
            self.task_queue.remove(task_id)

        return True

    def _add_to_queue(self, task_id: str) -> None:
        """
        Add a task to the queue based on priority.

        Args:
            task_id: Task identifier
        """
        if task_id not in self.tasks:
            return

        task = self.tasks[task_id]

        # Find the right position based on priority
        position = len(self.task_queue)
        for i, queued_task_id in enumerate(self.task_queue):
            queued_task = self.tasks[queued_task_id]
            if task.priority.value > queued_task.priority.value:
                position = i
                break

        # Insert at the determined position
        self.task_queue.insert(position, task_id)

    async def _start_task_processor(self) -> None:
        """Start the background task processor."""
        while True:
            # Process tasks if queue is not empty and we have capacity
            if self.task_queue and len(self.running_tasks) < self.max_parallel_tasks:
                # Get next task ID
                task_id = self.task_queue.pop(0)

                # Skip if task doesn't exist anymore
                if task_id not in self.tasks:
                    continue

                task = self.tasks[task_id]

                # Skip if parent task is not in progress (for sequenced tasks)
                if task.parent_task_id and task.parent_task_id in self.tasks:
                    parent_task = self.tasks[task.parent_task_id]

                    # Check if this is a sequenced task
                    if parent_task.input_data.get("sequence", True):
                        # Check if previous sibling tasks have completed
                        siblings_completed = True
                        for sibling_id in parent_task.subtasks:
                            if sibling_id == task_id:
                                break

                            if sibling_id in self.tasks:
                                sibling = self.tasks[sibling_id]
                                if sibling.status != TaskStatus.COMPLETED:
                                    siblings_completed = False
                                    break

                        if not siblings_completed:
                            # Put back in queue and continue
                            self.task_queue.append(task_id)
                            await asyncio.sleep(0.1)
                            continue

                # Start task execution
                asyncio_task = asyncio.create_task(self._execute_task(task_id))
                self.running_tasks[task_id] = asyncio_task

            # Sleep briefly to avoid CPU hogging
            await asyncio.sleep(0.01)

    async def _execute_task(self, task_id: str) -> None:
        """
        Execute a task with an agent.

        Args:
            task_id: Task identifier
        """
        if task_id not in self.tasks:
            return

        task = self.tasks[task_id]

        # Mark task as in progress
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = time.time()

        try:
            # Check if this is a composite task handled by the orchestrator
            if task.agent_type == "orchestrator":
                result = await self._execute_composite_task(task)
            else:
                # Get the agent
                agent = self.agents.get(task.agent_type)

                if not agent:
                    raise ValueError(f"Agent type {task.agent_type} not found")

                # Set up context sharing if context ID is provided
                if task.context_id and agent.context_manager != self.context_manager:
                    if task.context_id in self.context_manager.sessions:
                        # Export context
                        context_data = self.context_manager.export_session(task.context_id)

                        # Import into agent's context manager
                        agent.context_manager.import_session(context_data)

                        # Update agent's session ID
                        agent.session_id = task.context_id

                # Execute task with the agent
                result = await self._execute_with_agent(agent, task)

            # Update task with result
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()

            logger.info(f"Task {task_id} completed, duration: {task.duration:.2f}s")

        except asyncio.CancelledError:
            # Task was cancelled
            task.status = TaskStatus.CANCELED
            task.completed_at = time.time()
            logger.info(f"Task {task_id} cancelled")
            raise

        except Exception as e:
            # Task failed
            logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)

            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()

            # Check if we should retry
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                task.error = f"Retry {task.retry_count}/{task.max_retries}: {str(e)}"

                # Add back to queue after delay
                await asyncio.sleep(task.retry_delay)
                self._add_to_queue(task_id)

                logger.info(f"Task {task_id} will be retried ({task.retry_count}/{task.max_retries})")

        finally:
            # Clean up
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]

            # Notify waiting tasks
            self.task_completed_event.set()

    async def _execute_composite_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a composite task by orchestrating its subtasks.

        Args:
            task: Composite task

        Returns:
            Aggregated results
        """
        # Wait for all subtasks to complete
        results = {}

        for subtask_id in task.subtasks:
            # Wait for subtask to complete
            result = await self.wait_for_task(subtask_id)

            # Store result
            results[subtask_id] = result

        # Check if we should aggregate results
        if task.input_data.get("aggregate_results", True):
            return {
                "aggregated_result": self._aggregate_results(results),
                "subtask_results": results
            }
        else:
            return {
                "subtask_results": results
            }

    async def _execute_with_agent(self, agent: BaseAgent, task: Task) -> Dict[str, Any]:
        """
        Execute a task with an agent.

        Args:
            agent: Agent to execute the task
            task: Task to execute

        Returns:
            Task result
        """
        # Handle different input types based on agent type
        if task.agent_type == "intent_agent":
            # Assume input_data has 'query' field
            query = task.input_data.get("query", "")
            result = await agent.process(query)
            return {"intent_analysis": result}

        elif task.agent_type == "technical_agent":
            # Handle technical agent input
            query = task.input_data.get("query", "")
            system_info = task.input_data.get("system_info")
            customer_id = task.input_data.get("customer_id")

            if "diagnose" in task.input_data:
                result = await agent.diagnose_issue(query, system_info, customer_id)
                return {"diagnosis": result}
            elif "suggest_solution" in task.input_data:
                diagnosis_id = task.input_data.get("diagnosis_id")
                result = await agent.suggest_solution(diagnosis_id)
                return {"solution": result}
            else:
                result = await agent.process(query)
                return {"result": result}

        elif task.agent_type == "knowledge_agent":
            # Handle knowledge agent input
            query = task.input_data.get("query", "")

            if "add_knowledge" in task.input_data:
                knowledge = task.input_data.get("add_knowledge", {})
                result = await agent.add_knowledge(
                    content=knowledge.get("content", ""),
                    entry_type=knowledge.get("entry_type", "general"),
                    metadata=knowledge.get("metadata"),
                    confidence=knowledge.get("confidence", 1.0)
                )
                return {"knowledge_added": result}
            elif "retrieve_knowledge" in task.input_data:
                retrieve_params = task.input_data.get("retrieve_knowledge", {})
                result = await agent.retrieve_knowledge(
                    query=query,
                    limit=retrieve_params.get("limit", 5),
                    entry_type=retrieve_params.get("entry_type")
                )
                return {"knowledge_entries": result}
            else:
                result = await agent.process(query)
                return {"knowledge_synthesis": result}

        else:
            # Generic processing for other agent types
            if "method" in task.input_data:
                # Call a specific method on the agent
                method_name = task.input_data.get("method")
                params = task.input_data.get("params", {})

                if not hasattr(agent, method_name):
                    raise ValueError(f"Agent {task.agent_type} does not have method {method_name}")

                method = getattr(agent, method_name)
                result = await method(**params)
                return {method_name: result}
            else:
                # Default to process method
                input_text = task.input_data.get("input", "")
                result = await agent.process(input_text)
                return {"result": result}

    def _get_task_result(self, task_id: str) -> Dict[str, Any]:
        """
        Get the result of a completed task.

        Args:
            task_id: Task identifier

        Returns:
            Task result information
        """
        if task_id not in self.tasks:
            return {"error": "Task not found"}

        task = self.tasks[task_id]

        return {
            "task_id": task_id,
            "status": task.status.value,
            "result": task.result,
            "error": task.error,
            "duration": task.duration
        }

    def _aggregate_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple subtasks.

        Args:
            results: Dictionary mapping subtask IDs to results

        Returns:
            Aggregated result
        """
        # Simple aggregation: combine all results into a single dictionary
        aggregated = {}

        for subtask_id, result in results.items():
            for key, value in result.items():
                if key not in aggregated:
                    aggregated[key] = []

                aggregated[key].append(value)

        return aggregated
    