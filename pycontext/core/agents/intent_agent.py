"""
pycontext/core/agents/intent_agent.py

Intent Analysis Agent for PyContext.
"""
from typing import Dict, List, Optional, Any
import json

from .base import BaseAgent
from ..mcp.protocol import ContextType


class IntentAgent(BaseAgent):
    """
    Intent Analysis Agent that identifies and classifies user intents.
    """

    def __init__(
            self,
            agent_id: str = None,
            llm_client: Any = None,
            confidence_threshold: float = 0.7
    ):
        """
        Initialize the Intent Analysis Agent.

        Args:
            agent_id: Optional unique identifier
            llm_client: LLM client for generating responses
            confidence_threshold: Minimum confidence for intent classification
        """
        super().__init__(
            agent_id=agent_id,
            agent_role="intent_agent"
        )
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold

    async def _load_role_prompt(self) -> str:
        """Load the intent agent's system prompt."""
        return """You are an Intent Analysis Agent in a customer service system.
Your role is to:
1. Precisely identify customer intent from their queries
2. Detect multiple or hidden intents
3. Assess intent confidence
4. Identify required context for resolution

Given a customer query, analyze it using this structure:
1. Primary Intent: Main customer goal
2. Secondary Intents: Additional or implied needs
3. Required Information: What we need to know
4. Confidence Score: How certain are you (0-1)
5. Context Needs: What additional context would help

Return your analysis in JSON format.
"""

    async def process(self, input_text: str) -> Dict:
        """
        Process user input to identify intents.

        Args:
            input_text: User's message

        Returns:
            Dict containing intent analysis
        """
        # Add user input to context
        await self.add_user_context(input_text, {"type": "customer_query"})

        # If we have an LLM client, use it to analyze intent
        if self.llm_client:
            formatted_context = self.context_manager.get_formatted_context(self.session_id)

            analysis_prompt = """Based on the customer query above, provide the following analysis:
1. Primary Intent: The main customer goal
2. Secondary Intents: Additional or implied needs
3. Required Information: What we need to know to resolve this
4. Confidence Score: How certain are you (0-1)
Return your analysis in JSON format.
"""

            # Process with LLM (implementation depends on the specific LLM client)
            response = await self._process_with_llm(formatted_context, analysis_prompt)

            try:
                # Parse response as JSON
                result = json.loads(response)
            except json.JSONDecodeError:
                # Fallback if response is not valid JSON
                result = {
                    "primary_intent": "unknown",
                    "secondary_intents": [],
                    "required_information": [],
                    "confidence": 0.5
                }
        else:
            # Placeholder implementation for when no LLM client is available
            # In a real implementation, this would use a trained classifier
            result = {
                "primary_intent": self._simple_intent_classification(input_text),
                "secondary_intents": [],
                "required_information": [],
                "confidence": 0.6
            }

        # Store the analysis result in context
        self.context_manager.add_context(
            session_id=self.session_id,
            content=json.dumps(result),
            context_type=ContextType.AGENT,
            relevance_score=0.9,
            metadata={"type": "intent_analysis"}
        )

        return result

    async def _process_with_llm(self, context: str, prompt: str) -> str:
        """
        Process context with LLM to generate a response.
        This method should be implemented based on the specific LLM client being used.

        Args:
            context: Formatted context
            prompt: Additional prompt

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
        #         {"role": "system", "content": context},
        #         {"role": "user", "content": prompt}
        #     ]
        # )
        # return response.choices[0].message.content

        return "Placeholder LLM response"

    def _simple_intent_classification(self, text: str) -> str:
        """
        Simple rule-based intent classification.
        This is a placeholder - in a real implementation, use a trained classifier.

        Args:
            text: User input text

        Returns:
            Classified intent
        """
        text = text.lower()

        if any(word in text for word in ["help", "support", "assist"]):
            return "request_assistance"
        elif any(word in text for word in ["price", "cost", "expensive", "cheap"]):
            return "pricing_inquiry"
        elif any(word in text for word in ["broken", "problem", "issue", "error", "not working"]):
            return "report_problem"
        elif any(word in text for word in ["how to", "how do i", "tutorial"]):
            return "how_to_question"
        else:
            return "general_inquiry"