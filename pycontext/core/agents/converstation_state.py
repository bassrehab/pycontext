"""
pycontext/core/agents/conversation_state.py

Conversation state tracking for dialog management.
"""
from typing import Dict, List, Optional, Any, Set, Union
import json
import time
from dataclasses import dataclass, field
from enum import Enum
import uuid


class ConversationPhase(Enum):
    """Phases of a conversation."""
    OPENING = "opening"            # Beginning of conversation
    PROBLEM_IDENTIFICATION = "problem_identification"  # Identifying the problem
    INFORMATION_EXCHANGE = "information_exchange"  # Exchanging information
    SOLUTION_GENERATION = "solution_generation"  # Generating solutions
    EVALUATION = "evaluation"      # Evaluating solutions
    DECISION = "decision"          # Making a decision
    CLOSING = "closing"            # Ending conversation


class ParticipantRole(Enum):
    """Roles of conversation participants."""
    USER = "user"                  # End user
    AGENT = "agent"                # AI agent
    SYSTEM = "system"              # System message
    HUMAN_AGENT = "human_agent"    # Human agent


@dataclass
class EntityMention:
    """Representation of an entity mentioned in conversation."""
    entity_id: str
    entity_type: str
    name: str
    mentions: List[int] = field(default_factory=list)  # Turn numbers
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    

@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    id: str
    participant: ParticipantRole
    content: str
    timestamp: float
    turn_number: int
    intents: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)  # Entity IDs
    sentiment: str = "neutral"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationState:
    """
    Tracks and manages the state of a conversation.
    Maintains history, entities, and contextual information.
    """
    
    def __init__(self, conversation_id: Optional[str] = None):
        """
        Initialize conversation state.
        
        Args:
            conversation_id: Optional conversation identifier
        """
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.turns: List[ConversationTurn] = []
        self.entities: Dict[str, EntityMention] = {}
        self.active_intents: List[Dict[str, Any]] = []
        self.phase: ConversationPhase = ConversationPhase.OPENING
        self.metadata: Dict[str, Any] = {}
        self.start_time: float = time.time()
        self.last_update_time: float = time.time()
        
        # Track specific conversation information
        self.missing_information: Set[str] = set()
        self.discussed_topics: Dict[str, int] = {}  # Topic -> mention count
        self.action_items: List[Dict[str, Any]] = []
        self.satisfaction_indicators: List[Dict[str, Any]] = []
    
    def add_turn(
        self,
        content: str,
        participant: ParticipantRole,
        intents: Optional[List[Dict[str, Any]]] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
        sentiment: str = "neutral",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a turn to the conversation.
        
        Args:
            content: Turn content
            participant: Participant role
            intents: Optional list of intents
            entities: Optional list of entities
            sentiment: Sentiment of the turn
            metadata: Optional additional metadata
            
        Returns:
            Turn ID
        """
        # Update timestamps
        self.last_update_time = time.time()
        
        # Create turn ID
        turn_id = str(uuid.uuid4())
        turn_number = len(self.turns) + 1
        
        # Process entities
        entity_ids = []
        if entities:
            for entity in entities:
                entity_id = self._process_entity(entity, turn_number)
                entity_ids.append(entity_id)
        
        # Create turn
        turn = ConversationTurn(
            id=turn_id,
            participant=participant,
            content=content,
            timestamp=self.last_update_time,
            turn_number=turn_number,
            intents=intents or [],
            entities=entity_ids,
            sentiment=sentiment,
            metadata=metadata or {}
        )
        
        # Add to turns
        self.turns.append(turn)
        
        # Process intents
        if intents:
            for intent in intents:
                self._process_intent(intent, turn_number)
        
        # Update phase if needed
        self._update_phase(turn)
        
        # Update discussed topics
        self._update_topics(content)
        
        # Update satisfaction indicators
        if participant == ParticipantRole.USER:
            self._update_satisfaction(content, sentiment)
        
        return turn_id
    
    def get_turn(self, turn_id: str) -> Optional[ConversationTurn]:
        """
        Get a turn by ID.
        
        Args:
            turn_id: Turn identifier
            
        Returns:
            Turn if found, None otherwise
        """
        for turn in self.turns:
            if turn.id == turn_id:
                return turn
        return None
    
    def get_recent_turns(self, count: int = 5) -> List[ConversationTurn]:
        """
        Get the most recent turns.
        
        Args:
            count: Maximum number of turns to retrieve
            
        Returns:
            List of recent turns
        """
        return self.turns[-count:] if self.turns else []
    
    def get_entity_mentions(self, entity_id: str) -> List[ConversationTurn]:
        """
        Get all turns where an entity was mentioned.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            List of turns
        """
        if entity_id not in self.entities:
            return []
        
        entity = self.entities[entity_id]
        return [self.turns[turn_num - 1] for turn_num in entity.mentions if 0 < turn_num <= len(self.turns)]
    
    def update_missing_information(
        self,
        missing_info: Optional[List[str]] = None,
        resolved_info: Optional[List[str]] = None
    ) -> None:
        """
        Update the set of missing information.
        
        Args:
            missing_info: New missing information
            resolved_info: Information that has been provided
        """
        if missing_info:
            self.missing_information.update(missing_info)
        
        if resolved_info:
            for info in resolved_info:
                if info in self.missing_information:
                    self.missing_information.remove(info)
    
    def add_action_item(
        self,
        description: str,
        assigned_to: Optional[str] = None,
        deadline: Optional[str] = None,
        status: str = "pending"
    ) -> str:
        """
        Add an action item.
        
        Args:
            description: Action item description
            assigned_to: Who the action is assigned to
            deadline: Optional deadline
            status: Status of the action item
            
        Returns:
            Action item ID
        """
        action_id = str(uuid.uuid4())
        
        self.action_items.append({
            "id": action_id,
            "description": description,
            "assigned_to": assigned_to,
            "deadline": deadline,
            "status": status,
            "created_at": time.time(),
            "updated_at": time.time()
        })
        
        return action_id
    
    def update_action_item(
        self,
        action_id: str,
        status: Optional[str] = None,
        description: Optional[str] = None,
        assigned_to: Optional[str] = None,
        deadline: Optional[str] = None
    ) -> bool:
        """
        Update an action item.
        
        Args:
            action_id: Action item identifier
            status: New status
            description: New description
            assigned_to: New assignee
            deadline: New deadline
            
        Returns:
            Whether the update was successful
        """
        for action in self.action_items:
            if action["id"] == action_id:
                if status is not None:
                    action["status"] = status
                if description is not None:
                    action["description"] = description
                if assigned_to is not None:
                    action["assigned_to"] = assigned_to
                if deadline is not None:
                    action["deadline"] = deadline
                
                action["updated_at"] = time.time()
                return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation state to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "turns": [self._turn_to_dict(turn) for turn in self.turns],
            "entities": {k: self._entity_to_dict(v) for k, v in self.entities.items()},
            "active_intents": self.active_intents,
            "phase": self.phase.value,
            "metadata": self.metadata,
            "start_time": self.start_time,
            "last_update_time": self.last_update_time,
            "missing_information": list(self.missing_information),
            "discussed_topics": self.discussed_topics,
            "action_items": self.action_items,
            "satisfaction_indicators": self.satisfaction_indicators
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationState':
        """Create conversation state from dictionary."""
        state = cls(conversation_id=data["conversation_id"])
        
        # Restore timestamps
        state.start_time = data["start_time"]
        state.last_update_time = data["last_update_time"]
        
        # Restore phase
        state.phase = ConversationPhase(data["phase"])
        
        # Restore metadata
        state.metadata = data["metadata"]
        
        # Restore entities first (they're referenced by turns)
        for entity_id, entity_data in data["entities"].items():
            state.entities[entity_id] = cls._dict_to_entity(entity_data)
        
        # Restore turns
        for turn_data in data["turns"]:
            state.turns.append(cls._dict_to_turn(turn_data))
        
        # Restore active intents
        state.active_intents = data["active_intents"]
        
        # Restore missing information
        state.missing_information = set(data["missing_information"])
        
        # Restore discussed topics
        state.discussed_topics = data["discussed_topics"]
        
        # Restore action items
        state.action_items = data["action_items"]
        
        # Restore satisfaction indicators
        state.satisfaction_indicators = data["satisfaction_indicators"]
        
        return state
    
    def _process_entity(
        self,
        entity: Dict[str, Any],
        turn_number: int
    ) -> str:
        """
        Process an entity, adding or updating it in the state.
        
        Args:
            entity: Entity information
            turn_number: Current turn number
            
        Returns:
            Entity ID
        """
        entity_type = entity.get("type", "unknown")
        entity_name = entity.get("name", "")
        entity_id = entity.get("id", str(uuid.uuid4()))
        attributes = entity.get("attributes", {})
        confidence = entity.get("confidence", 1.0)
        
        # Check if entity already exists
        if entity_id in self.entities:
            # Update existing entity
            existing = self.entities[entity_id]
            
            # Add turn to mentions if not already there
            if turn_number not in existing.mentions:
                existing.mentions.append(turn_number)
            
            # Update attributes
            existing.attributes.update(attributes)
            
            # Update name if confidence is higher
            if confidence > existing.confidence:
                existing.name = entity_name
                existing.confidence = confidence
        else:
            # Create new entity
            self.entities[entity_id] = EntityMention(
                entity_id=entity_id,
                entity_type=entity_type,
                name=entity_name,
                mentions=[turn_number],
                attributes=attributes,
                confidence=confidence
            )
        
        return entity_id
    
    def _process_intent(self, intent: Dict[str, Any], turn_number: int) -> None:
        """
        Process an intent, updating conversation state.
        
        Args:
            intent: Intent information
            turn_number: Current turn number
        """
        # Check if this intent updates missing information
        if "required_information" in intent:
            missing = intent.get("required_information", [])
            self.update_missing_information(missing_info=missing)
        
        # Add to active intents if not already there
        intent_name = intent.get("intent", intent.get("name", ""))
        if intent_name:
            # Check if already active
            active = False
            for active_intent in self.active_intents:
                if active_intent.get("intent", active_intent.get("name", "")) == intent_name:
                    active = True
                    # Update turn references
                    if "turns" not in active_intent:
                        active_intent["turns"] = []
                    if turn_number not in active_intent["turns"]:
                        active_intent["turns"].append(turn_number)
                    break
            
            # Add if not active
            if not active:
                intent_copy = intent.copy()
                if "turns" not in intent_copy:
                    intent_copy["turns"] = [turn_number]
                self.active_intents.append(intent_copy)
    
    def _update_phase(self, turn: ConversationTurn) -> None:
        """
        Update conversation phase based on the latest turn.
        
        Args:
            turn: Latest conversation turn
        """
        # Simple state machine for phase transitions
        current_phase = self.phase
        
        # Check for explicit phase transitions in intents
        for intent in turn.intents:
            intent_name = intent.get("intent", intent.get("name", "")).lower()
            
            if "greeting" in intent_name and current_phase == ConversationPhase.OPENING:
                # Stay in opening
                return
            
            elif any(term in intent_name for term in ["problem", "issue", "help", "question"]):
                if current_phase == ConversationPhase.OPENING:
                    self.phase = ConversationPhase.PROBLEM_IDENTIFICATION
                    return
            
            elif any(term in intent_name for term in ["explain", "describe", "detail", "information"]):
                if current_phase in [ConversationPhase.OPENING, ConversationPhase.PROBLEM_IDENTIFICATION]:
                    self.phase = ConversationPhase.INFORMATION_EXCHANGE
                    return
            
            elif any(term in intent_name for term in ["solve", "solution", "fix", "resolve"]):
                if current_phase in [ConversationPhase.PROBLEM_IDENTIFICATION, ConversationPhase.INFORMATION_EXCHANGE]:
                    self.phase = ConversationPhase.SOLUTION_GENERATION
                    return
            
            elif any(term in intent_name for term in ["evaluate", "compare", "assess", "consider"]):
                if current_phase == ConversationPhase.SOLUTION_GENERATION:
                    self.phase = ConversationPhase.EVALUATION
                    return
            
            elif any(term in intent_name for term in ["decide", "choose", "select", "go with"]):
                if current_phase in [ConversationPhase.SOLUTION_GENERATION, ConversationPhase.EVALUATION]:
                    self.phase = ConversationPhase.DECISION
                    return
            
            elif any(term in intent_name for term in ["thanks", "bye", "goodbye", "end"]):
                self.phase = ConversationPhase.CLOSING
                return
        
        # Default transitions based on turn count
        if current_phase == ConversationPhase.OPENING and len(self.turns) >= 2:
            self.phase = ConversationPhase.PROBLEM_IDENTIFICATION
        elif current_phase == ConversationPhase.PROBLEM_IDENTIFICATION and len(self.turns) >= 4:
            self.phase = ConversationPhase.INFORMATION_EXCHANGE
        elif current_phase == ConversationPhase.SOLUTION_GENERATION and len(self.turns) >= 8:
            self.phase = ConversationPhase.EVALUATION
    
    def _update_topics(self, content: str) -> None:
        """
        Update discussed topics based on content.
        
        Args:
            content: Turn content
        """
        # This is a very simple topic extraction
        # In a real implementation, this would use an LLM or NLP model
        
        # Simple keyword-based topic detection
        topics = {
            "pricing": ["price", "cost", "fee", "payment", "subscription", "plan"],
            "technical": ["problem", "issue", "error", "broken", "not working"],
            "account": ["account", "login", "password", "username", "profile"],
            "product": ["product", "feature", "service", "offering"],
            "support": ["support", "help", "assistance", "contact"]
        }
        
        content_lower = content.lower()
        
        for topic, keywords in topics.items():
            if any(keyword in content_lower for keyword in keywords):
                self.discussed_topics[topic] = self.discussed_topics.get(topic, 0) + 1
    
    def _update_satisfaction(self, content: str, sentiment: str) -> None:
        """
        Update satisfaction indicators based on user content and sentiment.
        
        Args:
            content: Turn content
            sentiment: Turn sentiment
        """
        # Check for satisfaction indicators in content
        positive_indicators = ["thank", "great", "helpful", "excellent", "awesome", "good"]
        negative_indicators = ["unhelpful", "useless", "waste", "frustrated", "disappointed"]
        
        content_lower = content.lower()
        
        # Create indicator
        indicator = {
            "turn": len(self.turns),
            "timestamp": time.time(),
            "sentiment": sentiment,
            "positive_matches": [word for word in positive_indicators if word in content_lower],
            "negative_matches": [word for word in negative_indicators if word in content_lower],
            "overall": "positive" if sentiment in ["positive"] else 
                      "negative" if sentiment in ["negative", "frustrated"] else "neutral"
        }
        
        # Add to indicators
        self.satisfaction_indicators.append(indicator)
    
    @staticmethod
    def _turn_to_dict(turn: ConversationTurn) -> Dict[str, Any]:
        """Convert conversation turn to dictionary."""
        return {
            "id": turn.id,
            "participant": turn.participant.value,
            "content": turn.content,
            "timestamp": turn.timestamp,
            "turn_number": turn.turn_number,
            "intents": turn.intents,
            "entities": turn.entities,
            "sentiment": turn.sentiment,
            "metadata": turn.metadata
        }
    
    @staticmethod
    def _entity_to_dict(entity: EntityMention) -> Dict[str, Any]:
        """Convert entity mention to dictionary."""
        return {
            "entity_id": entity.entity_id,
            "entity_type": entity.entity_type,
            "name": entity.name,
            "mentions": entity.mentions,
            "attributes": entity.attributes,
            "confidence": entity.confidence
        }
    
    @classmethod
    def _dict_to_turn(cls, data: Dict[str, Any]) -> ConversationTurn:
        """Create conversation turn from dictionary."""
        return ConversationTurn(
            id=data["id"],
            participant=ParticipantRole(data["participant"]),
            content=data["content"],
            timestamp=data["timestamp"],
            turn_number=data["turn_number"],
            intents=data["intents"],
            entities=data["entities"],
            sentiment=data["sentiment"],
            metadata=data["metadata"]
        )
    
    @classmethod
    def _dict_to_entity(cls, data: Dict[str, Any]) -> EntityMention:
        """Create entity mention from dictionary."""
        return EntityMention(
            entity_id=data["entity_id"],
            entity_type=data["entity_type"],
            name=data["name"],
            mentions=data["mentions"],
            attributes=data["attributes"],
            confidence=data["confidence"]
        )


class ConversationStateTracker:
    """
    Tracks and manages multiple conversation states.
    Provides persistence and retrieval capabilities.
    """
    
    def __init__(self):
        """Initialize the conversation state tracker."""
        self.conversations: Dict[str, ConversationState] = {}
    
    def create_conversation(self, conversation_id: Optional[str] = None) -> str:
        """
        Create a new conversation.
        
        Args:
            conversation_id: Optional conversation identifier
            
        Returns:
            Conversation ID
        """
        conversation = ConversationState(conversation_id)
        self.conversations[conversation.conversation_id] = conversation
        return conversation.conversation_id
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationState]:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            Conversation state if found, None otherwise
        """
        return self.conversations.get(conversation_id)
    
    def add_turn(
        self,
        conversation_id: str,
        content: str,
        participant: ParticipantRole,
        intents: Optional[List[Dict[str, Any]]] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
        sentiment: str = "neutral",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Add a turn to a conversation.
        
        Args:
            conversation_id: Conversation identifier
            content: Turn content
            participant: Participant role
            intents: Optional list of intents
            entities: Optional list of entities
            sentiment: Sentiment of the turn
            metadata: Optional additional metadata
            
        Returns:
            Turn ID if successful, None otherwise
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return None
        
        return conversation.add_turn(
            content=content,
            participant=participant,
            intents=intents,
            entities=entities,
            sentiment=sentiment,
            metadata=metadata
        )
    
    def save_conversation(self, conversation_id: str, filename: str) -> bool:
        """
        Save a conversation to a file.
        
        Args:
            conversation_id: Conversation identifier
            filename: Output filename
            
        Returns:
            Whether the save was successful
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False
        
        try:
            with open(filename, 'w') as f:
                json.dump(conversation.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving conversation: {e}")
            return False
    
    def load_conversation(self, filename: str) -> Optional[str]:
        """
        Load a conversation from a file.
        
        Args:
            filename: Input filename
            
        Returns:
            Conversation ID if successful, None otherwise
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            conversation = ConversationState.from_dict(data)
            self.conversations[conversation.conversation_id] = conversation
            return conversation.conversation_id
        except Exception as e:
            print(f"Error loading conversation: {e}")
            return None
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """
        List all conversations.
        
        Returns:
            List of conversation summaries
        """
        return [
            {
                "id": conv_id,
                "turns": len(conv.turns),
                "start_time": conv.start_time,
                "last_update_time": conv.last_update_time,
                "phase": conv.phase.value
            }
            for conv_id, conv in self.conversations.items()
        ]
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            Whether the deletion was successful
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            return True
        return False
    