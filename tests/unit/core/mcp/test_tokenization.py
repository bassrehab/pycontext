"""
tests/unit/core/mcp/test_tokenization.py

Tests for the tokenization module.
"""
import unittest
from unittest.mock import patch, MagicMock
from pycontext.core.mcp.tokenization import (
    default_tokenizer,
    TokenizerRegistry,
    count_tokens,
    registry
)


class TestTokenization(unittest.TestCase):
    """Test the tokenization module."""

    def test_default_tokenizer(self):
        """Test the default tokenizer."""
        text = "Hello, world! This is a test."
        tokens = default_tokenizer(text)

        # Check the tokenization
        expected_tokens = [
            'Hello', ',', 'world', '!', 'This', 'is', 'a', 'test', '.'
        ]
        self.assertEqual(tokens, expected_tokens)
        self.assertEqual(len(tokens), 9)

    def test_tokenizer_registry(self):
        """Test the tokenizer registry."""
        # Create a new registry for testing
        test_registry = TokenizerRegistry()

        # Register a custom tokenizer
        def custom_tokenizer(text):
            return text.split()

        test_registry.register("custom-model", custom_tokenizer)

        # Check retrieval
        self.assertEqual(test_registry.get("custom-model"), custom_tokenizer)
        self.assertEqual(test_registry.get("non-existent-model"), test_registry.default)

    def test_count_tokens(self):
        """Test counting tokens."""
        text = "Hello world this is a test"

        # Count with default tokenizer
        count = count_tokens(text)
        self.assertEqual(count, 6)

        # Register a mock tokenizer and count again
        def mock_tokenizer(text):
            return ["token1", "token2", "token3"]  # Always returns 3 tokens

        # Use a registry mock to avoid affecting the global registry
        with patch('pycontext.core.mcp.tokenization.registry') as mock_registry:
            mock_registry.get.return_value = mock_tokenizer
            count = count_tokens(text, "mock-model")
            self.assertEqual(count, 3)
            mock_registry.get.assert_called_once_with("mock-model")

    @unittest.skipIf("tiktoken" not in globals(), "tiktoken not installed")
    def test_openai_tokenizer(self):
        """Test OpenAI tokenizer if available."""
        # This test will only run if tiktoken is installed
        text = "Hello world"

        # Should use OpenAI's tokenizer if available
        tokens = count_tokens(text, "gpt-4")
        # We can't assert the exact count since it depends on the tiktoken implementation
        # But we can check that it's reasonable
        self.assertGreater(tokens, 0)
        self.assertLess(tokens, 10)  # "Hello world" should be less than 10 tokens


class TestTokenizationIntegration(unittest.TestCase):
    """Integration tests for the tokenization module with mocked dependencies."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mocks for the external tokenizers
        self.tiktoken_mock = MagicMock()
        self.transformers_mock = MagicMock()

        # Setup OpenAI tokenizer mock
        self.encoded_tokens = [15496, 995]  # Example token IDs for "Hello world"
        self.tiktoken_encoder = MagicMock()
        self.tiktoken_encoder.encode.return_value = self.encoded_tokens
        self.tiktoken_mock.get_encoding.return_value = self.tiktoken_encoder

        # Setup Hugging Face tokenizer mock
        self.hf_tokenizer = MagicMock()
        self.hf_tokenizer.tokenize.return_value = ["Hello", "world"]
        self.transformers_mock.AutoTokenizer.from_pretrained.return_value = self.hf_tokenizer

    @patch.dict('sys.modules', {'tiktoken': MagicMock()})
    def test_openai_tokenizer_with_mock(self):
        """Test OpenAI tokenizer with mocked tiktoken."""
        import sys
        # Replace the tiktoken module with our mock
        sys.modules['tiktoken'] = self.tiktoken_mock

        # Create a registry with mocked modules
        from pycontext.core.mcp.tokenization import TokenizerRegistry
        test_registry = TokenizerRegistry()

        # Define a test text
        text = "Hello world"

        # Register a tokenizer function that uses our mock
        def test_openai_tokenizer(text):
            enc = self.tiktoken_mock.get_encoding("cl100k_base")
            return enc.encode(text)

        test_registry.register("gpt-4", test_openai_tokenizer)

        # Count tokens
        tokenizer = test_registry.get("gpt-4")
        tokens = tokenizer(text)

        # Check results
        self.assertEqual(tokens, self.encoded_tokens)
        self.tiktoken_mock.get_encoding.assert_called_once_with("cl100k_base")
        self.tiktoken_encoder.encode.assert_called_once_with(text)


if __name__ == "__main__":
    unittest.main()
