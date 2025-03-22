"""
tests/unit/core/mcp/test_protocol.py

Tests for the MCP protocol module.
"""
import unittest
import time
from pycontext.core.mcp.protocol import (
    ContextType,
    ContextBlock,
    ContextMetrics,
    ContextPackage,
    create_context_block,
    compress_context_block
)
from pycontext.core.mcp.tokenization import count_tokens


class TestMCPProtocol(unittest.TestCase):
    """Test the MCP protocol components."""

    def test_context_block_init(self):
        """Test context block initialization."""
        block = ContextBlock(
            id="test-id",
            content="Test content",
            relevance_score=0.8,
            type=ContextType.USER
        )

        self.assertEqual(block.id, "test-id")
        self.assertEqual(block.content, "Test content")
        self.assertEqual(block.relevance_score, 0.8)
        self.assertEqual(block.type, ContextType.USER)
        self.assertIsNotNone(block.timestamp)
        self.assertIsNotNone(block.metadata)
        self.assertIsNotNone(block.references)
        self.assertIsNotNone(block.token_count)

        # Check token counting
        self.assertEqual(block.token_count, count_tokens(block.content))

    def test_context_package_metrics(self):
        """Test context package metrics calculation."""
        # Create a package with two blocks
        blocks = [
            ContextBlock(
                id="block1",
                content="First block content",
                relevance_score=0.9,
                type=ContextType.SYSTEM,
                token_count=4
            ),
            ContextBlock(
                id="block2",
                content="Second block content with more words",
                relevance_score=0.8,
                type=ContextType.USER,
                token_count=7
            )
        ]

        package = ContextPackage(
            session_id="test-session",
            agent_id="test-agent",
            blocks=blocks,
            metrics=ContextMetrics(
                total_tokens=0,
                context_saturation=0,
                type_distribution={}
            )
        )

        # Calculate metrics
        package.calculate_metrics()

        # Check results
        self.assertEqual(package.metrics.total_tokens, 11)
        self.assertAlmostEqual(package.metrics.context_saturation, 11 / 8192)
        self.assertEqual(len(package.metrics.type_distribution), 2)
        self.assertIn(ContextType.SYSTEM.name, package.metrics.type_distribution)
        self.assertIn(ContextType.USER.name, package.metrics.type_distribution)

        # Check type distribution percentages
        self.assertAlmostEqual(package.metrics.type_distribution[ContextType.SYSTEM.name], 4 / 11)
        self.assertAlmostEqual(package.metrics.type_distribution[ContextType.USER.name], 7 / 11)

    def test_create_context_block_helper(self):
        """Test the create_context_block helper function."""
        block = create_context_block(
            content="Helper function test",
            context_type=ContextType.AGENT,
            relevance_score=0.75,
            metadata={"test": "value"},
            model_name="gpt-4"
        )

        self.assertEqual(block.content, "Helper function test")
        self.assertEqual(block.type, ContextType.AGENT)
        self.assertEqual(block.relevance_score, 0.75)
        self.assertEqual(block.metadata["test"], "value")
        self.assertEqual(block.metadata["model"], "gpt-4")
        self.assertIsNotNone(block.id)

        # Check token count
        self.assertEqual(block.token_count, count_tokens("Helper function test", "gpt-4"))

    def test_context_package_serialization(self):
        """Test context package serialization to dict."""
        # Create a package
        blocks = [
            ContextBlock(
                id="block1",
                content="System prompt",
                relevance_score=1.0,
                type=ContextType.SYSTEM,
                token_count=2
            ),
            ContextBlock(
                id="block2",
                content="User query",
                relevance_score=0.9,
                type=ContextType.USER,
                token_count=2
            )
        ]

        package = ContextPackage(
            session_id="test-session",
            agent_id="test-agent",
            blocks=blocks,
            metrics=ContextMetrics(
                total_tokens=4,
                context_saturation=4 / 8192,
                type_distribution={
                    ContextType.SYSTEM.name: 0.5,
                    ContextType.USER.name: 0.5
                }
            ),
            trace_id="test-trace"
        )

        # Convert to dict
        package_dict = package.to_dict()

        # Check dict contents
        self.assertEqual(package_dict["session_id"], "test-session")
        self.assertEqual(package_dict["agent_id"], "test-agent")
        self.assertEqual(package_dict["trace_id"], "test-trace")
        self.assertEqual(package_dict["version"], 1)
        self.assertEqual(len(package_dict["blocks"]), 2)
        self.assertEqual(package_dict["metrics"]["total_tokens"], 4)
        self.assertEqual(package_dict["max_tokens"], 8192)

    def test_compress_context_block(self):
        """Test context block compression."""
        # Create a block with somewhat lengthy content
        original_block = ContextBlock(
            id="test-id",
            content="This is a somewhat lengthy piece of content that we will compress. " * 5,
            relevance_score=0.8,
            type=ContextType.MEMORY,
            token_count=None  # Will be auto-calculated
        )

        # Compress the block
        compressed_block = compress_context_block(
            block=original_block,
            compression_ratio=0.5
        )

        # Check that compression occurred
        self.assertLess(compressed_block.token_count, original_block.token_count)
        self.assertIn("compressed", compressed_block.metadata)
        self.assertEqual(compressed_block.metadata["compressed"], "true")
        self.assertEqual(compressed_block.metadata["original_tokens"], str(original_block.token_count))

        # The compressed block should be roughly half the size
        self.assertLess(
            compressed_block.token_count,
            original_block.token_count * 0.6  # Allow some overhead
        )

        # Check that very small blocks aren't compressed
        small_block = ContextBlock(
            id="small-id",
            content="Small text",
            relevance_score=0.8,
            type=ContextType.MEMORY,
            token_count=2
        )

        compressed_small = compress_context_block(
            block=small_block,
            compression_ratio=0.5
        )

        # Should be the same block
        self.assertEqual(compressed_small.token_count, small_block.token_count)
        self.assertNotIn("compressed", compressed_small.metadata)


if __name__ == "__main__":
    unittest.main()