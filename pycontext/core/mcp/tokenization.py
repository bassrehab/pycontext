"""
pycontext/core/mcp/tokenization.py

Tokenization utilities for the Model Context Protocol.
"""
from typing import List, Dict, Optional, Union, Callable
import re


# Default fallback tokenizer
def default_tokenizer(text: str) -> List[str]:
    """
    Simple whitespace and punctuation tokenizer as fallback.
    Not accurate for LLM token counting but works as fallback.
    """
    # Split on whitespace and punctuation, keeping punctuation as tokens
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return tokens


class TokenizerRegistry:
    """Registry of tokenizers for different models."""

    def __init__(self):
        self.tokenizers = {}
        self.default = default_tokenizer

    def register(self, model_name: str, tokenizer_fn: Callable):
        """Register a tokenizer for a model."""
        self.tokenizers[model_name] = tokenizer_fn

    def get(self, model_name: Optional[str] = None) -> Callable:
        """Get a tokenizer for a model or the default tokenizer."""
        if model_name is None:
            return self.default
        return self.tokenizers.get(model_name, self.default)


# Global registry
registry = TokenizerRegistry()


def count_tokens(text: str, model_name: Optional[str] = None) -> int:
    """
    Count tokens in text using the appropriate tokenizer.

    Args:
        text: Text to tokenize
        model_name: Optional model name to use specific tokenizer

    Returns:
        Number of tokens
    """
    tokenizer = registry.get(model_name)
    tokens = tokenizer(text)
    return len(tokens)


# Register OpenAI tokenizers if available
try:
    import tiktoken


    def openai_tokenizer(text: str) -> List[str]:
        """Tokenize text using tiktoken for OpenAI models."""
        enc = tiktoken.get_encoding("cl100k_base")  # Default for recent models
        return enc.encode(text)


    # Register for various OpenAI models
    registry.register("gpt-3.5-turbo", openai_tokenizer)
    registry.register("gpt-4", openai_tokenizer)
    registry.register("text-embedding-ada-002", openai_tokenizer)
except ImportError:
    # tiktoken not available, will use default tokenizer
    pass

# Register Anthropic tokenizers if available
try:
    from anthropic import Anthropic


    # This is a simplified approach - in practice would need to use
    # Anthropic's tokenizer more directly
    def anthropic_tokenizer(text: str) -> List[str]:
        """Approximate tokenization for Anthropic models."""
        # This is a very rough approximation
        # In practice, would use Anthropic's tokenizer
        return text.split()


    # Register for Anthropic models
    registry.register("claude-2", anthropic_tokenizer)
    registry.register("claude-instant-1", anthropic_tokenizer)
except ImportError:
    # anthropic not available, will use default tokenizer
    pass

# Register Hugging Face tokenizers if available
try:
    from transformers import AutoTokenizer


    def get_hf_tokenizer(model_name: str):
        """Get a tokenizer function for a Hugging Face model."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize(text: str) -> List[str]:
            return tokenizer.tokenize(text)

        return tokenize

    # Could register specific HF models here
    # For example:
    # registry.register("gpt2", get_hf_tokenizer("gpt2"))
except ImportError:
    # transformers not available, will use default tokenizer
    pass