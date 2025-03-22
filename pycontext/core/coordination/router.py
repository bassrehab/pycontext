"""
pycontext/core/coordination/router.py

Route tasks to appropriate agents based on content and intent.
"""
from typing import Dict, List, Optional, Any, Union, Tuple
import json
import asyncio
import logging
import re

from ..agents.base import BaseAgent
from ..context.manager import ContextManager
from ..mcp.protocol import ContextType
from .orchestrator import TaskPriority

logger = logging.getLogger(__name__)


class AgentRouter:
    """
    Routes tasks to appropriate agents based on content analysis.
    Can use intent recognition to determine the best agent for a task.
    """

    def __init__(
            self,
            agents: Dict[str, BaseAgent],
            default_agent: str,
            intent_agent: Optional[str] = None,
            context_manager: Optional[ContextManager] = None,
            routing_rules: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize the agent router.

        Args:
            agents: Dictionary of available agents
            default_agent: Default agent type to use
            intent_agent: Optional intent analysis agent
            context_manager: Optional context manager
            routing_rules: Optional routing rules mapping intents to agent types
        """
        self.agents = agents
        self.default_agent = default_agent
        self.intent_agent = intent_agent
        self.context_manager = context_manager or ContextManager()
        self.routing_rules = routing_rules or {}

        # Default routing rules if none provided
        if not self.routing_rules:
            self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """Set up default routing rules."""
        self.routing_rules = {
            # Technical intents
            "technical_issue": ["technical_agent"],
            "troubleshooting": ["technical_agent"],
            "error": ["technical_agent"],
            "not_working": ["technical_agent"],

            # Knowledge intents
            "information": ["knowledge_agent"],
            "how_to": ["knowledge_agent"],
            "explanation": ["knowledge_agent"],
            "definition": ["knowledge_agent"],

            # Dialog intents
            "greeting": ["dialog_agent"],
            "farewell": ["dialog_agent"],
            "thanks": ["dialog_agent"],
            "smalltalk": ["dialog_agent"],

            # Customer service intents
            "complaint": ["dialog_agent", "knowledge_agent"],
            "feedback": ["dialog_agent"],
            "request": ["knowledge_agent", "technical_agent"]
        }

    async def route(
            self,
            input_text: str,
            context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, float]:
        """
        Route input to the appropriate agent.

        Args:
            input_text: User input text
            context: Optional additional context

        Returns:
            Tuple of (agent_type, confidence)
        """
        # Use intent agent if available
        if self.intent_agent and self.intent_agent in self.agents:
            intent_result = await self._analyze_intent(input_text)
            if intent_result:
                agent_type = await self._route_by_intent(intent_result)
                confidence = intent_result.get("confidence", 0.0)
                return agent_type, confidence

        # Fall back to pattern-based routing
        agent_type, confidence = self._route_by_pattern(input_text)
        return agent_type, confidence

    async def process_message(
            self,
            input_text: str,
            context: Optional[Dict[str, Any]] = None,
            session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a message by routing it to the appropriate agent.

        Args:
            input_text: User input text
            context: Optional additional context
            session_id: Optional session ID for context continuity

        Returns:
            Processing result
        """
        # Route to appropriate agent
        agent_type, confidence = await self.route(input_text, context)

        # Get the agent
        agent = self.agents.get(agent_type)
        if not agent:
            logger.warning(f"Agent type {agent_type} not found, using default")
            agent = self.agents.get(self.default_agent)

            if not agent:
                return {
                    "error": "No suitable agent found",
                    "input": input_text
                }

        # Initialize agent session if needed
        if not agent.session_id:
            if session_id:
                # Try to import existing session
                if session_id in self.context_manager.sessions:
                    context_data = self.context_manager.export_session(session_id)
                    agent.import_context(context_data)
                else:
                    await agent.initialize_session()
                    agent.session_id = session_id
            else:
                await agent.initialize_session()

        # Process with agent
        try:
            result = await agent.process(input_text)

            # Export updated context if using shared session
            if session_id and session_id != agent.session_id:
                context_data = agent.export_context()
                self.context_manager.import_session(context_data)

            return {
                "agent_type": agent_type,
                "confidence": confidence,
                "result": result
            }

        except Exception as e:
            logger.error(f"Error processing with {agent_type}: {str(e)}", exc_info=True)
            return {
                "error": f"Error processing with {agent_type}: {str(e)}",
                "agent_type": agent_type,
                "input": input_text
            }

    async def _analyze_intent(self, input_text: str) -> Optional[Dict[str, Any]]:
        """
        Analyze intent using intent agent.

        Args:
            input_text: User input text

        Returns:
            Intent analysis result or None on failure
        """
        try:
            intent_agent = self.agents[self.intent_agent]
            intent_result = await intent_agent.process(input_text)
            return intent_result
        except Exception as e:
            logger.error(f"Error analyzing intent: {str(e)}", exc_info=True)
            return None

    async def _route_by_intent(self, intent_result: Dict[str, Any]) -> str:
        """
        Determine appropriate agent based on intent analysis.

        Args:
            intent_result: Intent analysis result

        Returns:
            Appropriate agent type
        """
        # Extract primary intent
        primary_intent = intent_result.get("primary_intent", "").lower()

        # Check confidence threshold
        confidence = intent_result.get("confidence", 0.0)
        if confidence < 0.5:
            logger.info(f"Low intent confidence ({confidence}), using default agent")
            return self.default_agent

        # Check routing rules for primary intent
        for intent_pattern, agent_types in self.routing_rules.items():
            if intent_pattern in primary_intent:
                # Use first available agent in the list
                for agent_type in agent_types:
                    if agent_type in self.agents:
                        logger.info(f"Routing to {agent_type} based on intent '{primary_intent}'")
                        return agent_type

        # Check secondary intents if present
        secondary_intents = intent_result.get("secondary_intents", [])
        for secondary_intent in secondary_intents:
            if isinstance(secondary_intent, str):
                intent_name = secondary_intent.lower()
            elif isinstance(secondary_intent, dict) and "name" in secondary_intent:
                intent_name = secondary_intent["name"].lower()
            else:
                continue

            for intent_pattern, agent_types in self.routing_rules.items():
                if intent_pattern in intent_name:
                    # Use first available agent in the list
                    for agent_type in agent_types:
                        if agent_type in self.agents:
                            logger.info(f"Routing to {agent_type} based on secondary intent '{intent_name}'")
                            return agent_type

        # No matching intent, use default
        logger.info(f"No matching intent rule for '{primary_intent}', using default agent")
        return self.default_agent

    def _route_by_pattern(self, input_text: str) -> Tuple[str, float]:
        """
        Route based on text patterns when intent analysis is not available.

        Args:
            input_text: User input text

        Returns:
            Tuple of (agent_type, confidence)
        """
        input_lower = input_text.lower()

        # Technical patterns
        technical_patterns = [
            r"(not|isn't|doesn't|won't)\s+work(ing)?",
            r"(problem|issue|error|bug|crash|broken)",
            r"(fix|repair|troubleshoot|diagnose)",
            r"(screen|button|keyboard|mouse|click|tap)",
            r"(wifi|internet|network|connection|signal)"
        ]

        for pattern in technical_patterns:
            if re.search(pattern, input_lower):
                if "technical_agent" in self.agents:
                    return "technical_agent", 0.7

        # Knowledge patterns
        knowledge_patterns = [
            r"(what|how|why|when|where|who|which)",
            r"(explain|tell|show|describe|definition|meaning)",
            r"(difference between|compare|versus|vs\.)"
        ]

        for pattern in knowledge_patterns:
            if re.search(pattern, input_lower):
                if "knowledge_agent" in self.agents:
                    return "knowledge_agent", 0.7

        # Dialog patterns
        dialog_patterns = [
            r"^(hello|hi|hey|greetings)",
            r"(thanks|thank you)",
            r"(bye|goodbye|see you)",
            r"(nice|good|great|excellent)"
        ]

        for pattern in dialog_patterns:
            if re.search(pattern, input_lower):
                if "dialog_agent" in self.agents:
                    return "dialog_agent", 0.6

        # Default to the default agent with low confidence
        return self.default_agent, 0.3


class MultiAgentRouter:
    """
    Advanced router that can use multiple agents in sequence.
    Can determine which combination of agents would be best for a task.
    """

    def __init__(
            self,
            agents: Dict[str, BaseAgent],
            default_sequence: List[str],
            intent_agent: Optional[str] = None,
            context_manager: Optional[ContextManager] = None,
            routing_rules: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize the multi-agent router.

        Args:
            agents: Dictionary of available agents
            default_sequence: Default sequence of agent types to use
            intent_agent: Optional intent analysis agent
            context_manager: Optional context manager
            routing_rules: Optional routing rules mapping intents to agent sequences
        """
        self.agents = agents
        self.default_sequence = default_sequence
        self.intent_agent = intent_agent
        self.context_manager = context_manager or ContextManager()
        self.routing_rules = routing_rules or {}

        # Simple router for initial routing
        self.simple_router = AgentRouter(
            agents=agents,
            default_agent=default_sequence[0] if default_sequence else list(agents.keys())[0],
            intent_agent=intent_agent,
            context_manager=context_manager,
            routing_rules=routing_rules
        )

    async def route(
            self,
            input_text: str,
            context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Route input to a sequence of appropriate agents.

        Args:
            input_text: User input text
            context: Optional additional context

        Returns:
            List of agent types to use in sequence
        """
        # Use intent agent if available
        if self.intent_agent and self.intent_agent in self.agents:
            try:
                intent_agent = self.agents[self.intent_agent]
                intent_result = await intent_agent.process(input_text)

                if intent_result:
                    sequence = await self._route_by_intent(intent_result)
                    return sequence
            except Exception as e:
                logger.error(f"Error in multi-agent routing: {str(e)}", exc_info=True)

        # If no intent-based routing, use simple router to get primary agent
        primary_agent, _ = await self.simple_router.route(input_text, context)

        # Build a sequence with the primary agent first
        sequence = [primary_agent]

        # Add other agents based on the content
        remaining_agents = [a for a in self.default_sequence if a != primary_agent]
        sequence.extend(remaining_agents)

        # Ensure sequence only contains available agents
        sequence = [agent_type for agent_type in sequence if agent_type in self.agents]

        # If sequence is empty, use default
        if not sequence:
            sequence = self.default_sequence

        return sequence

    async def process_message(
            self,
            input_text: str,
            context: Optional[Dict[str, Any]] = None,
            session_id: Optional[str] = None,
            max_agents: int = 3
    ) -> Dict[str, Any]:
        """
        Process a message by routing it through a sequence of agents.

        Args:
            input_text: User input text
            context: Optional additional context
            session_id: Optional session ID for context continuity
            max_agents: Maximum number of agents to use

        Returns:
            Processing result
        """
        # Route to appropriate agent sequence
        agent_sequence = await self.route(input_text, context)

        # Limit sequence length
        agent_sequence = agent_sequence[:max_agents]

        results = []
        current_input = input_text

        # Create or get session
        if not session_id:
            session_id = self.context_manager.create_session("multi_agent_router")

        # Process through each agent in sequence
        for i, agent_type in enumerate(agent_sequence):
            # Get the agent
            agent = self.agents.get(agent_type)
            if not agent:
                logger.warning(f"Agent type {agent_type} not found, skipping")
                continue

            # Initialize agent session if needed
            if not agent.session_id:
                # Import existing session
                if session_id and session_id in self.context_manager.sessions:
                    context_data = self.context_manager.export_session(session_id)
                    agent.import_context(context_data)
                else:
                    await agent.initialize_session()

            try:
                # Process with agent
                result = await agent.process(current_input)

                # Add to results
                results.append({
                    "agent_type": agent_type,
                    "result": result
                })

                # For all but the last agent, use the result as input for the next agent
                if i < len(agent_sequence) - 1:
                    # Convert result to string if it's a dict
                    if isinstance(result, dict):
                        current_input = json.dumps(result)
                    else:
                        current_input = str(result)

                # Export context for next agent
                context_data = agent.export_context()
                self.context_manager.import_session(context_data)

            except Exception as e:
                logger.error(f"Error processing with {agent_type}: {str(e)}", exc_info=True)
                results.append({
                    "agent_type": agent_type,
                    "error": str(e)
                })

        return {
            "agent_sequence": agent_sequence,
            "results": results,
            "final_result": results[-1]["result"] if results else None
        }

    async def _route_by_intent(self, intent_result: Dict[str, Any]) -> List[str]:
        """
        Determine appropriate agent sequence based on intent analysis.

        Args:
            intent_result: Intent analysis result

        Returns:
            List of agent types to use in sequence
        """
        # Extract primary intent
        primary_intent = intent_result.get("primary_intent", "").lower()

        # Check routing rules for primary intent
        for intent_pattern, agent_sequence in self.routing_rules.items():
            if intent_pattern in primary_intent:
                # Filter to only available agents
                sequence = [a for a in agent_sequence if a in self.agents]
                if sequence:
                    logger.info(f"Routing to sequence {sequence} based on intent '{primary_intent}'")
                    return sequence

        # No matching rule, use default sequence
        return self.default_sequence
