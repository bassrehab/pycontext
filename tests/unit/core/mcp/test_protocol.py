"""
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
)


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
        self.assertAlmostEqual(package.metrics.context_saturation, 11/8192)
        self.assertEqual(len(package.metrics.type_distribution), 2)
        self.assertIn(ContextType.SYSTEM.name, package.metrics.type_distribution)
        self.assertIn(ContextType.USER.name, package.metrics.type_distribution)
    
    def test_create_context_block_helper(self):
        """Test the create_context_block helper function."""
        block = create_context_block(
            content="Helper function test",
            context_type=ContextType.AGENT,
            relevance_score=0.75,
            metadata={"test": "value"}
        )
        
        self.assertEqual(block.content, "Helper function test")
        self.assertEqual(block.type, ContextType.AGENT)
        self.assertEqual(block.relevance_score, 0.75)
        self.assertEqual(block.metadata["test"], "value")
        self.assertIsNotNone(block.id)


if __name__ == "__main__":
    unittest.main()
