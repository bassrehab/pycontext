"""
pycontext/core/agents/dialog_agent.py

Dialog Agent implementation for PyContext.
"""
from typing import Dict, List, Optional, Any, Tuple, Union
import json
import asyncio
import uuid
import time
import logging
from enum import Enum

from .base import BaseAgent
from ..mcp.protocol import ContextType
from ..memory.working_memory import WorkingMemory

logger = logging.getLogger(__name__)


class DialogState(Enum):
    """States for dialog management."""
    GREETING = "greeting"           # Initial greeting
    INFORMATION_GATHERING = "information_gathering"  # Collecting information
    CLARIFICATION = "clarification"  # Asking for clarification
    CONFIRMATION = "confirmation"    # Confirming understanding
    RESOLUTION = "resolution"        # Providing resolution
    CLOSING = "closing"              # Ending conversation
    HANDOFF = "handoff"              # Handing off to human/other agent


class DialogMove(Enum):
    """Dialog moves that can be made in conversation."""
    GREETING = "greeting"                   # Initial greeting
    FAREWELL = "farewell"                   # Ending the conversation
    INFORM = "inform"                       # Providing information
    REQUEST = "request"                     # Requesting information
    CLARIFY = "clarify"                     # Asking for clarification
    CONFIRM = "confirm"                     # Confirming understanding
    SUGGEST = "suggest"                     # Making a suggestion
    ACKNOWLEDGE = "acknowledge"             # Acknowledging user input
    EMPATHIZE = "empathize"                 # Expressing empathy
    HANDOFF = "handoff"                     # Handing off to another agent/human
    FOLLOWUP = "followup"                   # Following up on previous topic
    META = "meta"                           # Talking about the conversation itself


class DialogStrategy:
    """Strategies for dialog management."""
    
    def __init__(
        self,
        initial_state: DialogState = DialogState.GREETING,
        max_clarification_turns: int = 3,
        empathy_level: float = 0.7  # 0.0 to 1.0
    ):
        """
        Initialize the dialog strategy.
        
        Args:
            initial_state: Initial dialog state
            max_clarification_turns: Maximum number of clarification turns
            empathy_level: Level of empathy to express (0.0 to 1.0)
        """
        self.initial_state = initial_state
        self.max_clarification_turns = max_clarification_turns
        self.empathy_level = empathy_level
        self.state_transitions = self._define_state_transitions()
    
    def _define_state_transitions(self) -> Dict[DialogState, List[DialogState]]:
        """Define allowed state transitions."""
        return {
            DialogState.GREETING: [
                DialogState.INFORMATION_GATHERING,
                DialogState.RESOLUTION,
                DialogState.CLOSING
            ],
            DialogState.INFORMATION_GATHERING: [
                DialogState.CLARIFICATION,
                DialogState.CONFIRMATION,
                DialogState.RESOLUTION,
                DialogState.HANDOFF
            ],
            DialogState.CLARIFICATION: [
                DialogState.INFORMATION_GATHERING,
                DialogState.CONFIRMATION,
                DialogState.RESOLUTION,
                DialogState.HANDOFF
            ],
            DialogState.CONFIRMATION: [
                DialogState.RESOLUTION,
                DialogState.INFORMATION_GATHERING,
                DialogState.CLARIFICATION,
                DialogState.HANDOFF
            ],
            DialogState.RESOLUTION: [
                DialogState.CLOSING,
                DialogState.INFORMATION_GATHERING,
                DialogState.CONFIRMATION
            ],
            DialogState.CLOSING: [],  # Terminal state
            DialogState.HANDOFF: []   # Terminal state
        }
    
    def next_state(
        self,
        current_state: DialogState,
        intent: Dict[str, Any],
        context: Dict[str, Any]
    ) -> DialogState:
        """
        Determine the next dialog state.
        
        Args:
            current_state: Current dialog state
            intent: Intent analysis result
            context: Conversation context
            
        Returns:
            Next dialog state
        """
        # Extract useful information
        primary_intent = intent.get("primary_intent", "").lower()
        confidence = intent.get("confidence", 0.0)
        clarification_count = context.get("clarification_count", 0)
        required_info = intent.get("required_information", [])
        missing_info = context.get("missing_information", [])
        
        # Check for farewell/closing intent
        if any(term in primary_intent for term in ["bye", "goodbye", "farewell", "thank"]):
            return DialogState.CLOSING
        
        # Handle low confidence - ask for clarification
        if confidence < 0.5 and current_state != DialogState.CLARIFICATION:
            return DialogState.CLARIFICATION
        
        # Too many clarification turns - consider handoff
        if (current_state == DialogState.CLARIFICATION and 
            clarification_count >= self.max_clarification_turns):
            return DialogState.HANDOFF
        
        # Missing required information - gather more information
        if required_info and missing_info:
            return DialogState.INFORMATION_GATHERING
        
        # Ready to provide resolution
        if (current_state in [DialogState.INFORMATION_GATHERING, DialogState.CONFIRMATION] and 
            not missing_info):
            return DialogState.RESOLUTION
        
        # After resolution, move to closing
        if current_state == DialogState.RESOLUTION:
            return DialogState.CLOSING
        
        # Default: stay in current state if no obvious transition
        return current_state
    
    def suggest_moves(
        self,
        current_state: DialogState,
        intent: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[DialogMove]:
        """
        Suggest dialog moves based on current state.
        
        Args:
            current_state: Current dialog state
            intent: Intent analysis result
            context: Conversation context
            
        Returns:
            List of suggested dialog moves
        """
        moves = []
        
        if current_state == DialogState.GREETING:
            moves.append(DialogMove.GREETING)
            moves.append(DialogMove.INFORM)
        
        elif current_state == DialogState.INFORMATION_GATHERING:
            moves.append(DialogMove.REQUEST)
            moves.append(DialogMove.INFORM)
            
            # Add empathy if appropriate
            sentiment = context.get("sentiment", "neutral")
            if sentiment in ["negative", "frustrated", "confused"]:
                moves.append(DialogMove.EMPATHIZE)
        
        elif current_state == DialogState.CLARIFICATION:
            moves.append(DialogMove.CLARIFY)
            moves.append(DialogMove.ACKNOWLEDGE)
            
            # Add empathy for multiple clarifications
            clarification_count = context.get("clarification_count", 0)
            if clarification_count > 1:
                moves.append(DialogMove.EMPATHIZE)
        
        elif current_state == DialogState.CONFIRMATION:
            moves.append(DialogMove.CONFIRM)
            moves.append(DialogMove.ACKNOWLEDGE)
        
        elif current_state == DialogState.RESOLUTION:
            moves.append(DialogMove.INFORM)
            moves.append(DialogMove.SUGGEST)
            moves.append(DialogMove.FOLLOWUP)
        
        elif current_state == DialogState.CLOSING:
            moves.append(DialogMove.FAREWELL)
            moves.append(DialogMove.FOLLOWUP)
        
        elif current_state == DialogState.HANDOFF:
            moves.append(DialogMove.HANDOFF)
            moves.append(DialogMove.EMPATHIZE)
        
        return moves


class DialogAgent(BaseAgent):
    """
    Dialog Agent that manages conversational interaction.
    Focuses on dialogue structure, clarifications, and appropriate responses.
    """
    
    def __init__(
        self,
        agent_id: str = None,
        llm_client: Any = None,
        working_memory: Optional[WorkingMemory] = None,
        dialog_strategy: Optional[DialogStrategy] = None
    ):
        """
        Initialize the Dialog Agent.
        
        Args:
            agent_id: Optional unique identifier
            llm_client: LLM client for generating responses
            working_memory: Optional working memory
            dialog_strategy: Optional dialog strategy
        """
        super().__init__(
            agent_id=agent_id,
            agent_role="dialog_agent"
        )
        self.llm_client = llm_client
        self.working_memory = working_memory or WorkingMemory()
        self.dialog_strategy = dialog_strategy or DialogStrategy()
        
        # Track conversation state
        self.current_state = self.dialog_strategy.initial_state
        self.conversation_context = {}
        self.turn_count = 0
    
    async def _load_role_prompt(self) -> str:
        """Load the dialog agent's system prompt."""
        return """You are a Dialog Management Agent specializing in conversational interaction.
Your role is to:
1. Maintain natural, engaging conversations
2. Ask appropriate clarifying questions
3. Express empathy when needed
4. Guide the conversation towards resolution

When interacting with users:
1. Be responsive to user emotions and needs
2. Ask clarifying questions when necessary
3. Confirm understanding before moving forward
4. Provide clear, helpful responses
5. Use appropriate conversational transitions

Your responses should be natural, engaging, and focused on helping the user.
"""
    
    async def process(self, input_text: str) -> Dict[str, Any]:
        """
        Process user input and generate a dialog response.
        
        Args:
            input_text: User's message
            
        Returns:
            Dict containing dialog response and state information
        """
        # Add user input to context
        await self.add_user_context(input_text, {"type": "user_message"})
        
        # Update conversation context
        self._update_conversation_context(input_text)
        
        # Analyze sentiment and intent if LLM client is available
        sentiment = await self._analyze_sentiment(input_text)
        intent = await self._analyze_intent(input_text)
        
        # Update context with sentiment and intent
        self.conversation_context["sentiment"] = sentiment
        if intent:
            self.conversation_context["intent"] = intent
        
        # Determine next state
        next_state = self.dialog_strategy.next_state(
            self.current_state,
            intent or {},
            self.conversation_context
        )
        
        # Update clarification count if needed
        if next_state == DialogState.CLARIFICATION:
            self.conversation_context["clarification_count"] = self.conversation_context.get("clarification_count", 0) + 1
        
        # Get suggested moves
        suggested_moves = self.dialog_strategy.suggest_moves(
            next_state,
            intent or {},
            self.conversation_context
        )
        
        # Generate response
        response = await self._generate_response(input_text, next_state, suggested_moves)
        
        # Update state
        self.current_state = next_state
        self.turn_count += 1
        
        # Add response to context
        self.context_manager.add_context(
            session_id=self.session_id,
            content=json.dumps({
                "response": response,
                "state": next_state.value,
                "moves": [move.value for move in suggested_moves]
            }),
            context_type=ContextType.AGENT,
            relevance_score=0.9,
            metadata={
                "type": "dialog_response",
                "turn": self.turn_count,
                "state": next_state.value
            }
        )
        
        # Store in working memory
        if self.working_memory:
            self.working_memory.add(
                content={
                    "user_input": input_text,
                    "response": response,
                    "state": next_state.value,
                    "sentiment": sentiment,
                    "intent": intent,
                    "turn": self.turn_count
                },
                memory_type="dialog_turn",
                metadata={
                    "state": next_state.value,
                    "sentiment": sentiment,
                    "turn": self.turn_count
                }
            )
        
        # Prepare result
        result = {
            "response": response,
            "state": next_state.value,
            "moves": [move.value for move in suggested_moves],
            "sentiment": sentiment,
            "turn_count": self.turn_count
        }
        
        if intent:
            result["intent"] = intent
        
        return result
    
    async def generate_question(
        self,
        context: Dict[str, Any],
        missing_information: List[str]
    ) -> str:
        """
        Generate a question to ask the user.
        
        Args:
            context: Conversation context
            missing_information: List of missing information items
            
        Returns:
            Question to ask
        """
        # Add context to working memory if available
        if self.working_memory:
            self.working_memory.add(
                content={
                    "context": context,
                    "missing_information": missing_information
                },
                memory_type="question_context",
                metadata={"type": "question_generation"}
            )
        
        # Create prompt for question generation
        prompt = f"""
Based on the conversation context, generate a natural question to ask the user.

Missing information: {', '.join(missing_information)}

Conversation context: {json.dumps(context, indent=2)}

Generate a conversational question that asks for the missing information in a natural way.
"""
        
        # Process with LLM
        if self.llm_client:
            # Get formatted context
            formatted_context = self.context_manager.get_formatted_context(self.session_id)
            
            # Generate question
            question = await self._process_with_llm(formatted_context, prompt)
        else:
            # Default question if no LLM
            if missing_information:
                question = f"Could you please tell me about {missing_information[0]}?"
            else:
                question = "Could you please provide more information?"
        
        return question
    
    async def handle_unclear_input(
        self,
        input_text: str,
        confidence: float
    ) -> str:
        """
        Handle unclear user input by generating a clarification question.
        
        Args:
            input_text: User's unclear message
            confidence: Confidence in understanding
            
        Returns:
            Clarification response
        """
        # Update conversation context
        clarification_count = self.conversation_context.get("clarification_count", 0) + 1
        self.conversation_context["clarification_count"] = clarification_count
        
        # Generate appropriate response based on clarification count
        if clarification_count == 1:
            clarification = f"I'm sorry, but I'm not entirely sure what you're asking. Could you please provide more details about what you need help with?"
        elif clarification_count == 2:
            clarification = f"I'm still having trouble understanding. Could you try rephrasing your request in a different way?"
        else:
            clarification = f"I apologize, but I'm having difficulty understanding your request. Perhaps you could break it down into simpler parts or be more specific?"
        
        # Process with LLM if available for more natural response
        if self.llm_client:
            # Create clarification prompt
            prompt = f"""
The user said: "{input_text}"

I'm having trouble understanding this message (confidence: {confidence:.2f}).
This is clarification attempt #{clarification_count}.

Generate a polite clarification response asking for more information.
"""
            
            # Get formatted context
            formatted_context = self.context_manager.get_formatted_context(self.session_id)
            
            # Generate clarification
            clarification = await self._process_with_llm(formatted_context, prompt)
        
        return clarification
    
    def reset_conversation(self) -> None:
        """Reset the conversation state."""
        self.current_state = self.dialog_strategy.initial_state
        self.conversation_context = {}
        self.turn_count = 0
    
    def _update_conversation_context(self, input_text: str) -> None:
        """
        Update conversation context with user input.
        
        Args:
            input_text: User's message
        """
        # Store user input
        if "user_inputs" not in self.conversation_context:
            self.conversation_context["user_inputs"] = []
        
        self.conversation_context["user_inputs"].append(input_text)
        self.conversation_context["latest_input"] = input_text
        
        # Get conversation history from working memory if available
        if self.working_memory:
            recent_turns = self.working_memory.get_by_type("dialog_turn")
            if recent_turns:
                self.conversation_context["recent_turns"] = recent_turns
    
    async def _analyze_sentiment(self, input_text: str) -> str:
        """
        Analyze sentiment of user input.
        
        Args:
            input_text: User's message
            
        Returns:
            Sentiment label
        """
        if not self.llm_client:
            return "neutral"  # Default without LLM
        
        # Simple rule-based sentiment detection as fallback
        negative_terms = ["angry", "upset", "frustrated", "annoyed", "terrible", "bad", "hate"]
        positive_terms = ["happy", "glad", "pleased", "good", "great", "excellent", "love", "thanks"]
        
        input_lower = input_text.lower()
        
        if any(term in input_lower for term in negative_terms):
            return "negative"
        elif any(term in input_lower for term in positive_terms):
            return "positive"
        
        # Use LLM for more accurate sentiment analysis
        prompt = f"""
Analyze the sentiment of this message: "{input_text}"

Classify the sentiment as one of: positive, negative, neutral, confused, frustrated.
Respond with just the sentiment label.
"""
        
        # Get formatted context
        formatted_context = self.context_manager.get_formatted_context(self.session_id)
        
        # Analyze sentiment
        sentiment = await self._process_with_llm(formatted_context, prompt)
        
        # Clean up response
        sentiment = sentiment.strip().lower()
        
        # Normalize to standard categories
        if sentiment in ["positive", "negative", "neutral", "confused", "frustrated"]:
            return sentiment
        
        # Default fallback
        return "neutral"
    
    async def _analyze_intent(self, input_text: str) -> Optional[Dict[str, Any]]:
        """
        Analyze intent of user input.
        
        Args:
            input_text: User's message
            
        Returns:
            Intent analysis or None if not available
        """
        if not self.llm_client:
            return None  # Can't analyze intent without LLM
        
        # Create intent analysis prompt
        prompt = f"""
Analyze the intent of this message: "{input_text}"

Provide a structured analysis with:
1. Primary intent: The main goal or request
2. Required information: What information is needed to fulfill this intent
3. Confidence: How certain you are about this intent (0-1)

Respond in JSON format.
"""
        
        # Get formatted context
        formatted_context = self.context_manager.get_formatted_context(self.session_id)
        
        # Analyze intent
        response = await self._process_with_llm(formatted_context, prompt)
        
        try:
            # Parse JSON response
            intent = json.loads(response)
            return intent
        except json.JSONDecodeError:
            # Fallback if not valid JSON
            return {
                "primary_intent": "unknown",
                "required_information": [],
                "confidence": 0.5
            }
    
    async def _generate_response(
        self,
        input_text: str,
        dialog_state: DialogState,
        suggested_moves: List[DialogMove]
    ) -> str:
        """
        Generate a response based on dialog state and moves.
        
        Args:
            input_text: User's message
            dialog_state: Current dialog state
            suggested_moves: Suggested dialog moves
            
        Returns:
            Generated response
        """
        if not self.llm_client:
            # Simple rule-based responses if no LLM
            return self._generate_rule_based_response(dialog_state, suggested_moves)
        
        # Create response generation prompt
        prompt = f"""
Generate a natural, conversational response to this user message: "{input_text}"

Current dialog state: {dialog_state.value}
Suggested dialog moves: {', '.join([move.value for move in suggested_moves])}

Your response should incorporate the suggested dialog moves and be appropriate for the current state.
Keep the response conversational, empathetic, and helpful.
"""
        
        # Get formatted context
        formatted_context = self.context_manager.get_formatted_context(self.session_id)
        
        # Generate response
        response = await self._process_with_llm(formatted_context, prompt)
        
        return response
    
    def _generate_rule_based_response(
        self,
        dialog_state: DialogState,
        suggested_moves: List[DialogMove]
    ) -> str:
        """
        Generate a rule-based response when LLM is not available.
        
        Args:
            dialog_state: Current dialog state
            suggested_moves: Suggested dialog moves
            
        Returns:
            Generated response
        """
        # Simple responses based on state
        if dialog_state == DialogState.GREETING:
            return "Hello! How can I help you today?"
        
        elif dialog_state == DialogState.INFORMATION_GATHERING:
            return "Could you please provide more details about what you need?"
        
        elif dialog_state == DialogState.CLARIFICATION:
            return "I'm not sure I understand. Could you please clarify what you mean?"
        
        elif dialog_state == DialogState.CONFIRMATION:
            return "Just to confirm, you're asking about..."
        
        elif dialog_state == DialogState.RESOLUTION:
            return "Based on what you've told me, here's what I can suggest..."
        
        elif dialog_state == DialogState.CLOSING:
            return "Thank you for your time. Is there anything else I can help you with?"
        
        elif dialog_state == DialogState.HANDOFF:
            return "I'll connect you with someone who can better assist you with this."
        
        # Default response
        return "I understand. Please tell me more about how I can help you."
    
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