"""
pycontext/core/coordination/planner.py

Task planner for breaking down complex tasks into subtasks.
"""
from typing import Dict, List, Optional, Any, Union, Callable
import json
import asyncio
import logging

from ..agents.base import BaseAgent
from ..context.manager import ContextManager
from ..mcp.protocol import ContextType
from .orchestrator import TaskPriority

logger = logging.getLogger(__name__)


class TaskPlan:
    """A plan for accomplishing a complex task through multiple subtasks."""

    def __init__(
            self,
            goal: str,
            subtasks: List[Dict[str, Any]],
            context_id: Optional[str] = None
    ):
        """
        Initialize a task plan.

        Args:
            goal: The overall goal of the plan
            subtasks: List of subtask specifications
            context_id: Optional context ID for the plan
        """
        self.goal = goal
        self.subtasks = subtasks
        self.context_id = context_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert the plan to a dictionary."""
        return {
            "goal": self.goal,
            "subtasks": self.subtasks,
            "context_id": self.context_id
        }


class TaskPlanner:
    """
    Plans and breaks down complex tasks into manageable subtasks.
    Can use an LLM to help with planning for unstructured requests.
    """

    def __init__(
            self,
            available_agents: List[str],
            llm_client: Optional[Any] = None,
            context_manager: Optional[ContextManager] = None,
            task_templates: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the task planner.

        Args:
            available_agents: List of available agent types
            llm_client: Optional LLM client for planning
            context_manager: Optional context manager
            task_templates: Optional predefined task templates
        """
        self.available_agents = available_agents
        self.llm_client = llm_client
        self.context_manager = context_manager or ContextManager()
        self.task_templates = task_templates or {}

    async def create_plan(
            self,
            goal: str,
            context: Optional[Dict[str, Any]] = None,
            template_name: Optional[str] = None
    ) -> TaskPlan:
        """
        Create a plan for accomplishing a goal.

        Args:
            goal: The goal to accomplish
            context: Optional context information
            template_name: Optional template to use

        Returns:
            A task plan
        """
        # Create context session for the plan
        context_id = self.context_manager.create_session("planner")

        # Store goal and context in context manager
        self.context_manager.add_context(
            session_id=context_id,
            content=goal,
            context_type=ContextType.USER,
            relevance_score=1.0,
            metadata={"type": "goal"}
        )

        if context:
            self.context_manager.add_context(
                session_id=context_id,
                content=json.dumps(context),
                context_type=ContextType.SYSTEM,
                relevance_score=0.9,
                metadata={"type": "planning_context"}
            )

        # Use template if specified
        if template_name and template_name in self.task_templates:
            subtasks = await self._apply_template(template_name, goal, context)

        # Otherwise use LLM or rule-based planning
        else:
            if self.llm_client:
                subtasks = await self._llm_based_planning(goal, context, context_id)
            else:
                subtasks = await self._rule_based_planning(goal, context)

        # Create task plan
        plan = TaskPlan(
            goal=goal,
            subtasks=subtasks,
            context_id=context_id
        )

        # Store plan in context
        self.context_manager.add_context(
            session_id=context_id,
            content=json.dumps(plan.to_dict()),
            context_type=ContextType.AGENT,
            relevance_score=1.0,
            metadata={"type": "plan"}
        )

        return plan

    async def _apply_template(
            self,
            template_name: str,
            goal: str,
            context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Apply a task template to create subtasks.

        Args:
            template_name: Name of the template to use
            goal: The goal to accomplish
            context: Optional context information

        Returns:
            List of subtask specifications
        """
        template = self.task_templates[template_name]

        # Templates are functions that generate subtasks
        if callable(template):
            return template(goal, context)

        # Or they can be static subtask lists with placeholders
        subtasks = []
        for subtask_template in template:
            subtask = subtask_template.copy()

            # Replace placeholders in input data
            if "input_data" in subtask:
                for key, value in subtask["input_data"].items():
                    if isinstance(value, str) and "{goal}" in value:
                        subtask["input_data"][key] = value.replace("{goal}", goal)

            subtasks.append(subtask)

        return subtasks

    async def _llm_based_planning(
            self,
            goal: str,
            context: Optional[Dict[str, Any]] = None,
            context_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Use an LLM to create a plan.

        Args:
            goal: The goal to accomplish
            context: Optional context information
            context_id: Optional context ID

        Returns:
            List of subtask specifications
        """
        if not self.llm_client:
            raise ValueError("LLM client required for LLM-based planning")

        # Create planning prompt
        prompt = self._create_planning_prompt(goal, context)

        # Get formatted context if context ID is provided
        formatted_context = None
        if context_id:
            formatted_context = self.context_manager.get_formatted_context(context_id)

        # Process with LLM
        response = await self._process_with_llm(formatted_context, prompt)

        try:
            # Parse JSON response
            plan_data = json.loads(response)

            # Extract subtasks
            if "subtasks" in plan_data:
                subtasks = plan_data["subtasks"]
            else:
                subtasks = plan_data

            # Validate subtasks
            validated_subtasks = []
            for subtask in subtasks:
                # Ensure subtask has required fields
                if "agent_type" not in subtask:
                    logger.warning(f"Subtask missing agent_type: {subtask}")
                    continue

                # Check if agent type is available
                if subtask["agent_type"] not in self.available_agents:
                    logger.warning(f"Agent type {subtask['agent_type']} not available")
                    continue

                # Ensure subtask has input_data
                if "input_data" not in subtask:
                    subtask["input_data"] = {"input": goal}

                validated_subtasks.append(subtask)

            return validated_subtasks

        except json.JSONDecodeError:
            logger.error("Failed to parse LLM planning response as JSON")
            return self._fallback_planning(goal, context)

    async def _rule_based_planning(
            self,
            goal: str,
            context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Use rules to create a plan without an LLM.

        Args:
            goal: The goal to accomplish
            context: Optional context information

        Returns:
            List of subtask specifications
        """
        subtasks = []
        goal_lower = goal.lower()

        # Default to using intent agent first
        if "intent_agent" in self.available_agents:
            subtasks.append({
                "agent_type": "intent_agent",
                "input_data": {"query": goal},
                "priority": TaskPriority.HIGH.value
            })

        # Check for technical issues
        if any(term in goal_lower for term in [
            "problem", "issue", "error", "broken", "not working", "fix", "repair"
        ]):
            if "technical_agent" in self.available_agents:
                subtasks.append({
                    "agent_type": "technical_agent",
                    "input_data": {
                        "query": goal,
                        "diagnose": True,
                        "system_info": context.get("system_info") if context else None
                    },
                    "priority": TaskPriority.NORMAL.value
                })

        # Check for knowledge queries
        if any(term in goal_lower for term in [
            "what", "how", "why", "explain", "tell me about", "information"
        ]):
            if "knowledge_agent" in self.available_agents:
                subtasks.append({
                    "agent_type": "knowledge_agent",
                    "input_data": {"query": goal},
                    "priority": TaskPriority.NORMAL.value
                })

        # Add a default agent if no specific agents were identified
        if not subtasks and self.available_agents:
            default_agent = self.available_agents[0]
            subtasks.append({
                "agent_type": default_agent,
                "input_data": {"input": goal},
                "priority": TaskPriority.NORMAL.value
            })

        return subtasks

    def _fallback_planning(
            self,
            goal: str,
            context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create a fallback plan when LLM planning fails.

        Args:
            goal: The goal to accomplish
            context: Optional context information

        Returns:
            List of subtask specifications
        """
        # Use rule-based planning as fallback
        return self._rule_based_planning(goal, context)

    def _create_planning_prompt(
            self,
            goal: str,
            context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a prompt for LLM-based planning.

        Args:
            goal: The goal to accomplish
            context: Optional context information

        Returns:
            Planning prompt
        """
        prompt = f"""
You need to create a plan to accomplish the following goal:

GOAL: {goal}

Available agent types:
"""

        for agent_type in self.available_agents:
            prompt += f"- {agent_type}\n"

        if context:
            prompt += f"\nAdditional context:\n{json.dumps(context, indent=2)}\n"

        prompt += """
Break down this goal into subtasks that can be assigned to the available agents.
Each subtask should specify:
1. The agent type to use
2. The input data needed by the agent

Return your plan in JSON format like this:
{
  "subtasks": [
    {
      "agent_type": "agent_name",
      "input_data": {
        "key1": "value1",
        "key2": "value2"
      },
      "priority": 1  // Optional, default is 1 (NORMAL), 0 (LOW), 2 (HIGH), 3 (CRITICAL)
    },
    ...
  ]
}

The subtasks should be executed in the order they are listed.
"""

        return prompt

    async def _process_with_llm(
            self,
            context: Optional[str],
            prompt: str
    ) -> str:
        """
        Process context with LLM to generate a response.

        Args:
            context: Optional formatted context
            prompt: Planning prompt

        Returns:
            LLM response
        """
        if not self.llm_client:
            return "No LLM client available"

        # Placeholder - actual implementation depends on LLM client
        # For example, with OpenAI:
        # response = await self.llm_client.chat.completions.create(
        #     model="gpt-4",
        #     messages=[
        #         {"role": "system", "content": "You are a planning assistant that creates task plans."},
        #         {"role": "user", "content": prompt}
        #     ] if context is None else [
        #         {"role": "system", "content": context},
        #         {"role": "user", "content": prompt}
        #     ]
        # )
        # return response.choices[0].message.content

        return "Placeholder LLM response"


# Common task templates
CUSTOMER_SERVICE_TEMPLATE = [
    {
        "agent_type": "intent_agent",
        "input_data": {"query": "{goal}"},
        "priority": TaskPriority.HIGH.value
    },
    {
        "agent_type": "knowledge_agent",
        "input_data": {"query": "{goal}"},
        "priority": TaskPriority.NORMAL.value
    }
]

TECHNICAL_SUPPORT_TEMPLATE = [
    {
        "agent_type": "intent_agent",
        "input_data": {"query": "{goal}"},
        "priority": TaskPriority.HIGH.value
    },
    {
        "agent_type": "technical_agent",
        "input_data": {"query": "{goal}", "diagnose": True},
        "priority": TaskPriority.NORMAL.value
    },
    {
        "agent_type": "knowledge_agent",
        "input_data": {"query": "{goal}"},
        "priority": TaskPriority.LOW.value
    }
]

KNOWLEDGE_QUERY_TEMPLATE = [
    {
        "agent_type": "intent_agent",
        "input_data": {"query": "{goal}"},
        "priority": TaskPriority.HIGH.value
    },
    {
        "agent_type": "knowledge_agent",
        "input_data": {"query": "{goal}"},
        "priority": TaskPriority.HIGH.value
    }
]

# Default task templates
DEFAULT_TASK_TEMPLATES = {
    "customer_service": CUSTOMER_SERVICE_TEMPLATE,
    "technical_support": TECHNICAL_SUPPORT_TEMPLATE,
    "knowledge_query": KNOWLEDGE_QUERY_TEMPLATE
}