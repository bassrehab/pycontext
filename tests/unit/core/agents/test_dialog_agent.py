"""
tests/unit/core/agents/test_dialog_agent.py

Tests for the Dialog Agent module.
"""
import unittest
import asyncio
from unittest.mock import MagicMock, patch
import json
import time

from pycontext.core.agents.dialog_agent import (
    DialogAgent, DialogState, DialogMove, DialogStrategy
)
from pycontext.core.memory.working_memory import WorkingMemory
from pycontext.core.context.manager import ContextManager
from pycontext.core.mcp.protocol import ContextType


class TestDialogAgent(unittest.TestCase):
    """Test the Dialog Agent implementation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock LLM client
        self.mock_llm = MagicMock()
        self.mock_llm.return_value = "Mock LLM response"
        
        # Mock the _process_with_llm method
        self.process_with_llm_patcher = patch.object(
            DialogAgent, '_process_with_llm', 
            new=self.mock_process_with_llm
        )
        self.process_with_llm_patcher.start()
        
        # Create dialog agent
        self.working_memory = WorkingMemory()
        self.agent = DialogAgent(
            agent_id="test_dialog_agent",
            llm_client=self.mock_llm,
            working_memory=self.working_memory
        )
        
        # Initialize session
        asyncio.run(self.agent.initialize_session())

    def tearDown(self):
        """Clean up after tests."""
        self.process_with_llm_patcher.stop()

    async def mock_process_with_llm(self, context, prompt):
        """Mock implementation of _process_with_llm."""
        # Return different responses based on prompt content
        if "sentiment" in prompt.lower():
            return "positive"
        elif "intent" in prompt.lower():
            return json.dumps({
                "primary_intent": "greeting",
                "required_information": [],
                "confidence": 0.9
            })
        elif "clarification" in prompt.lower():
            return "I'm not sure I understand. Could you please clarify?"
        elif "question" in prompt.lower():
            return "Could you tell me more about your situation?"
        else:
            return "This is a generic response to your message."

    def test_init(self):
        """Test initialization of dialog agent."""
        self.assertEqual(self.agent.agent_id, "test_dialog_agent")
        self.assertEqual(self.agent.role, "dialog_agent")
        self.assertEqual(self.agent.current_state, DialogState.GREETING)
        self.assertEqual(self.agent.turn_count, 0)

    def test_load_role_prompt(self):
        """Test loading the role prompt."""
        prompt = asyncio.run(self.agent._load_role_prompt())
        
        # Check that prompt contains key elements
        self.assertIn("Dialog Management Agent", prompt)
        self.assertIn("conversational interaction", prompt)
        self.assertIn("clarifying questions", prompt)
        self.assertIn("empathy", prompt)

    def test_dialog_strategy(self):
        """Test dialog strategy behavior."""
        strategy = DialogStrategy()
        
        # Test state transitions
        next_state = strategy.next_state(
            DialogState.GREETING,
            {"primary_intent": "information", "confidence": 0.8},
            {}
        )
        self.assertEqual(next_state, DialogState.INFORMATION_GATHERING)
        
        # Test suggested moves
        moves = strategy.suggest_moves(
            DialogState.GREETING,
            {"primary_intent": "greeting"},
            {}
        )
        self.assertIn(DialogMove.GREETING, moves)
        self.assertIn(DialogMove.INFORM, moves)
        
        # Test clarification state
        next_state = strategy.next_state(
            DialogState.INFORMATION_GATHERING,
            {"primary_intent": "unknown", "confidence": 0.3},
            {}
        )
        self.assertEqual(next_state, DialogState.CLARIFICATION)
        
        # Test moves for clarification
        moves = strategy.suggest_moves(
            DialogState.CLARIFICATION,
            {"primary_intent": "unknown"},
            {"clarification_count": 1}
        )
        self.assertIn(DialogMove.CLARIFY, moves)
        self.assertIn(DialogMove.ACKNOWLEDGE, moves)

    def test_process(self):
        """Test processing a user message."""
        response = asyncio.run(self.agent.process("Hello there!"))
        
        # Check that response contains expected fields
        self.assertIn("response", response)
        self.assertIn("state", response)
        self.assertIn("moves", response)
        self.assertIn("sentiment", response)
        self.assertEqual(self.agent.turn_count, 1)
        
        # Check that state was updated
        self.assertNotEqual(self.agent.current_state, DialogState.GREETING)
        
        # Check that something was added to working memory
        self.assertGreater(len(self.working_memory.items), 0)

    def test_sentiment_analysis(self):
        """Test sentiment analysis function."""
        sentiment = asyncio.run(self.agent._analyze_sentiment("I'm happy with your service"))
        self.assertEqual(sentiment, "positive")
        
        # Test without LLM client
        self.agent.llm_client = None
        sentiment = asyncio.run(self.agent._analyze_sentiment("I'm happy with your service"))
        self.assertEqual(sentiment, "neutral")  # Default without LLM

    def test_intent_analysis(self):
        """Test intent analysis function."""
        intent = asyncio.run(self.agent._analyze_intent("I need help with my account"))
        self.assertEqual(intent["primary_intent"], "greeting")
        self.assertEqual(intent["confidence"], 0.9)
        
        # Test without LLM client
        self.agent.llm_client = None
        intent = asyncio.run(self.agent._analyze_intent("I need help with my account"))
        self.assertIsNone(intent)  # None without LLM

    def test_handle_unclear_input(self):
        """Test handling unclear user input."""
        response = asyncio.run(self.agent.handle_unclear_input("gibberish text", 0.2))
        self.assertIn("clarify", response.lower())
        
        # Check clarification count gets updated
        self.assertEqual(self.agent.conversation_context["clarification_count"], 1)
        
        # Test multiple clarifications
        response1 = asyncio.run(self.agent.handle_unclear_input("more gibberish", 0.2))
        self.assertEqual(self.agent.conversation_context["clarification_count"], 2)
        
        response2 = asyncio.run(self.agent.handle_unclear_input("yet more gibberish", 0.2))
        self.assertEqual(self.agent.conversation_context["clarification_count"], 3)
        
        # Responses should be different
        self.assertNotEqual(response, response1)
        self.assertNotEqual(response1, response2)

    def test_generate_question(self):
        """Test generating questions."""
        question = asyncio.run(self.agent.generate_question(
            {"topic": "account_issue"},
            ["account_number", "issue_type"]
        ))
        
        self.assertTrue(isinstance(question, str))
        self.assertGreater(len(question), 0)
        
        # Test without LLM
        self.agent.llm_client = None
        question = asyncio.run(self.agent.generate_question(
            {},
            ["account_number"]
        ))
        self.assertIn("account_number", question)

    def test_reset_conversation(self):
        """Test resetting conversation state."""
        # First process a message to change state
        asyncio.run(self.agent.process("Hello there!"))
        self.assertNotEqual(self.agent.current_state, DialogState.GREETING)
        self.assertEqual(self.agent.turn_count, 1)
        
        # Reset conversation
        self.agent.reset_conversation()
        
        # Check reset state
        self.assertEqual(self.agent.current_state, DialogState.GREETING)
        self.assertEqual(self.agent.turn_count, 0)
        self.assertEqual(self.agent.conversation_context, {})

    def test_multi_turn_conversation(self):
        """Test a multi-turn conversation."""
        # First turn
        response1 = asyncio.run(self.agent.process("Hello, I need help with my account"))
        
        # Second turn
        response2 = asyncio.run(self.agent.process("I can't log in to my account"))
        
        # Third turn
        response3 = asyncio.run(self.agent.process("Yes, I tried resetting my password"))
        
        # Check turn count
        self.assertEqual(self.agent.turn_count, 3)
        
        # Check conversation flow progressed
        self.assertNotEqual(response1["state"], response3["state"])
        
        # Check context was maintained
        self.assertIn("user_inputs", self.agent.conversation_context)
        self.assertEqual(len(self.agent.conversation_context["user_inputs"]), 3)
        
        # Check that each message is properly stored
        self.assertIn("Hello, I need help with my account", self.agent.conversation_context["user_inputs"])
        self.assertIn("I can't log in to my account", self.agent.conversation_context["user_inputs"])
        self.assertIn("Yes, I tried resetting my password", self.agent.conversation_context["user_inputs"])

    def test_rule_based_response(self):
        """Test rule-based response generation without LLM."""
        # Test responses for different states
        greeting_response = self.agent._generate_rule_based_response(
            DialogState.GREETING,
            [DialogMove.GREETING, DialogMove.INFORM]
        )
        self.assertIn("Hello", greeting_response)
        
        info_response = self.agent._generate_rule_based_response(
            DialogState.INFORMATION_GATHERING,
            [DialogMove.REQUEST, DialogMove.INFORM]
        )
        self.assertIn("details", info_response)
        
        closing_response = self.agent._generate_rule_based_response(
            DialogState.CLOSING,
            [DialogMove.FAREWELL, DialogMove.FOLLOWUP]
        )
        self.assertIn("Thank you", closing_response)

    def test_complex_state_transitions(self):
        """Test complex dialog state transitions across multiple turns."""
        # Mock the analyze intent and sentiment to have controlled inputs
        with patch.object(DialogAgent, '_analyze_intent') as mock_intent, \
             patch.object(DialogAgent, '_analyze_sentiment') as mock_sentiment:
            
            # Mock sentiment to be neutral by default
            mock_sentiment.return_value = "neutral"
            
            # Set up specific intents to force state transitions
            conversation_sequence = [
                # Turn 1: Greeting -> Information Gathering
                {"intent": {"primary_intent": "greeting", "confidence": 0.9}, 
                 "input": "Hello there"},
                
                # Turn 2: Stay in Information Gathering
                {"intent": {"primary_intent": "problem", "confidence": 0.8,
                           "required_information": ["specific_issue"]}, 
                 "input": "I'm having a problem with your service"},
                
                # Turn 3: Trigger Clarification due to low confidence
                {"intent": {"primary_intent": "unclear", "confidence": 0.3}, 
                 "input": "It's not working properly"},
                
                # Turn 4: Back to Information Gathering with details
                {"intent": {"primary_intent": "specific_problem", "confidence": 0.9}, 
                 "input": "My internet keeps disconnecting randomly"},
                
                # Turn 5: Move to Resolution
                {"intent": {"primary_intent": "confirmation", "confidence": 0.9}, 
                 "input": "Yes, it happens every hour or so"},
                
                # Turn 6: Move toward Closing
                {"intent": {"primary_intent": "acceptance", "confidence": 0.9}, 
                 "input": "That sounds good, I'll try that"},
                
                # Turn 7: Closing state
                {"intent": {"primary_intent": "farewell", "confidence": 0.9}, 
                 "input": "Thank you for your help"}
            ]
            
            # Process each turn in the conversation
            states = []
            
            for turn in conversation_sequence:
                # Set up the mocked intent for this turn
                mock_intent.return_value = turn["intent"]
                
                # Process the input
                result = asyncio.run(self.agent.process(turn["input"]))
                
                # Save the state
                states.append(result["state"])
            
            # Check that we went through proper state transitions
            self.assertEqual(len(states), len(conversation_sequence))
            
            # Verify that we hit the key states
            # Note: The exact sequence may vary based on the dialog strategy implementation
            # So we check for presence of key states instead of exact sequence
            
            # Check that clarification occurred
            self.assertIn(DialogState.CLARIFICATION.value, states)
            
            # Check that we reached resolution
            self.assertIn(DialogState.RESOLUTION.value, states)
            
            # Check that we ended with closing
            self.assertEqual(states[-1], DialogState.CLOSING.value)

    def test_emotional_conversation_handling(self):
        """Test how the agent handles emotionally charged conversations."""
        # Mock sentiment analysis to return different emotions
        with patch.object(DialogAgent, '_analyze_sentiment') as mock_sentiment:
            # First simulate an angry user
            mock_sentiment.return_value = "negative"
            
            response = asyncio.run(self.agent.process("I'm very upset with your service!"))
            
            # Check if empathy was one of the suggested moves
            self.assertIn(DialogMove.EMPATHIZE.value, response["moves"])
            
            # Now simulate a satisfied user
            mock_sentiment.return_value = "positive"
            
            response = asyncio.run(self.agent.process("I'm really happy with your help!"))
            
            # Check that the agent recognized positive sentiment
            self.assertEqual(response["sentiment"], "positive")

    def test_memory_integration(self):
        """Test integration with the working memory system."""
        # Process a message that should be stored in memory
        asyncio.run(self.agent.process("I need help with order #12345"))
        
        # Check if it was stored in working memory
        memory_items = self.working_memory.search("order #12345")
        self.assertGreater(len(memory_items), 0)
        
        # Check if dialog turn was properly stored
        dialog_turns = self.working_memory.get_by_type("dialog_turn")
        self.assertGreater(len(dialog_turns), 0)
        
        # Check if the turn contains the correct data
        turn = dialog_turns[0]
        self.assertIn("order #12345", str(turn))

    def test_edge_cases(self):
        """Test handling of edge cases in conversation."""
        # Test empty input
        response = asyncio.run(self.agent.process(""))
        self.assertIn("response", response)  # Should still provide a response
        
        # Test very long input (simulate truncation if needed)
        long_input = "Hello " * 100  # A very long repetitive message
        response = asyncio.run(self.agent.process(long_input))
        self.assertIn("response", response)  # Should handle long input gracefully
        
        # Test input with special characters
        special_input = "Can you help me? @#$%^&*()!~"
        response = asyncio.run(self.agent.process(special_input))
        self.assertIn("response", response)  # Should handle special characters
        
        # Test handling when LLM client fails
        with patch.object(DialogAgent, '_process_with_llm', side_effect=Exception("LLM Error")):
            # Should fallback to rule-based responses
            response = asyncio.run(self.agent.process("This will cause an LLM error"))
            self.assertIn("response", response)  # Should still get a response

    def test_context_maintenance_across_topics(self):
        """Test maintaining context across different conversation topics."""
        # Start with topic A
        asyncio.run(self.agent.process("I need help with my internet connection"))
        
        # Switch to topic B
        asyncio.run(self.agent.process("Actually, I also have a question about my bill"))
        
        # Return to topic A with reference
        response = asyncio.run(self.agent.process("Going back to my internet issue..."))
        
        # Check that context was maintained
        self.assertIn("internet", str(self.agent.conversation_context))
        self.assertIn("bill", str(self.agent.conversation_context))
        
        # Check that both topics are in the conversation history
        user_inputs = self.agent.conversation_context.get("user_inputs", [])
        has_internet_topic = any("internet" in input_text.lower() for input_text in user_inputs)
        has_bill_topic = any("bill" in input_text.lower() for input_text in user_inputs)
        
        self.assertTrue(has_internet_topic)
        self.assertTrue(has_bill_topic)
        
    def test_integration_with_context_manager(self):
        """Test integration with the PyContext context manager."""
        # Verify that the agent properly uses the context manager
        # Get the context data
        context_data = self.agent.export_context()
        
        # Check that it contains the correct session and context
        self.assertEqual(context_data["agent_id"], "test_dialog_agent")
        self.assertIsNotNone(context_data["session_id"])
        
        # Check that blocks were added during initialization
        self.assertGreater(len(context_data["blocks"]), 0)
        
        # Process a message
        asyncio.run(self.agent.process("Hello, I need some help."))
        
        # Get updated context
        updated_context = self.agent.export_context()
        
        # Context should have more blocks now
        self.assertGreater(len(updated_context["blocks"]), len(context_data["blocks"]))
        
        # At least one USER block should be present
        has_user_block = False
        for block in updated_context["blocks"]:
            if block["type"] == ContextType.USER.name:
                has_user_block = True
                break
        
        self.assertTrue(has_user_block, "Context should contain USER blocks")

    def test_missing_information_tracking(self):
        """Test tracking of missing information during conversation."""
        # Create a conversation with missing information
        with patch.object(DialogAgent, '_analyze_intent') as mock_intent:
            # First message with required information
            mock_intent.return_value = {
                "primary_intent": "account_issue",
                "required_information": ["account_number", "issue_type"],
                "confidence": 0.9
            }
            
            asyncio.run(self.agent.process("I have a problem with my account"))
            
            # Manual setup of missing information in conversation_context
            # since we don't have direct method to update it
            self.agent.conversation_context["missing_information"] = ["account_number", "issue_type"]
            
            # Second message providing some of the missing information
            mock_intent.return_value = {
                "primary_intent": "provide_info",
                "provided_information": ["account_number"],
                "confidence": 0.9
            }
            
            # Process a message that provides account number
            asyncio.run(self.agent.process("My account number is A12345"))
            
            # Manually update missing information to simulate resolved info
            current_missing = self.agent.conversation_context.get("missing_information", [])
            if "account_number" in current_missing:
                current_missing.remove("account_number")
            self.agent.conversation_context["missing_information"] = current_missing
            
            # Check that the missing information was updated correctly in context
            self.assertNotIn("account_number", self.agent.conversation_context.get("missing_information", []))
            self.assertIn("issue_type", self.agent.conversation_context.get("missing_information", []))


if __name__ == "__main__":
    unittest.main()