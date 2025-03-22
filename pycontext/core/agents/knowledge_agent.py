"""
pycontext/core/agents/knowledge_agent.py

Knowledge Agent implementation for PyContext.
"""
from typing import Dict, List, Optional, Any, Union, Tuple
import json
import asyncio

from .base import BaseAgent
from ..mcp.protocol import ContextType
from ..memory.semantic_memory import SemanticMemory, SemanticEntry


class KnowledgeAgent(BaseAgent):
    """
    Knowledge Agent that retrieves and synthesizes information from semantic memory.
    """

    def __init__(
            self,
            agent_id: str = None,
            llm_client: Any = None,
            semantic_memory: Optional[SemanticMemory] = None
    ):
        """
        Initialize the Knowledge Agent.

        Args:
            agent_id: Optional unique identifier
            llm_client: LLM client for generating responses
            semantic_memory: Semantic memory for knowledge retrieval
        """
        super().__init__(
            agent_id=agent_id,
            agent_role="knowledge_agent"
        )
        self.llm_client = llm_client
        self.semantic_memory = semantic_memory

    async def _load_role_prompt(self) -> str:
        """Load the knowledge agent's system prompt."""
        return """You are a Knowledge Agent specializing in retrieving and synthesizing information.
Your role is to:
1. Retrieve relevant information from knowledge sources
2. Synthesize information into coherent responses
3. Provide accurate and up-to-date knowledge
4. Acknowledge gaps in knowledge when they exist

When answering questions:
1. Consider the most relevant information first
2. Synthesize multiple sources when needed
3. Maintain citation information where appropriate
4. Structure information in a clear, logical manner
5. Explain complex concepts in understandable terms

Always provide sources for your information when available, and acknowledge when information
might be incomplete or uncertain.
"""

    async def process(self, input_text: str) -> Dict:
        """
        Process user input to retrieve and synthesize knowledge.

        Args:
            input_text: User's question or query

        Returns:
            Dict containing knowledge response
        """
        # Add user input to context
        await self.add_user_context(input_text, {"type": "knowledge_query"})

        # Retrieve relevant knowledge
        knowledge_results = await self._retrieve_knowledge(input_text)

        # Add knowledge to context
        for entry, score in knowledge_results:
            await self.add_memory_context(
                content=json.dumps({
                    "content": entry.content,
                    "type": entry.entry_type,
                    "confidence": entry.confidence,
                    "relevance": score
                }),
                relevance_score=score,
                metadata={
                    "type": "knowledge_entry",
                    "entry_id": entry.id,
                    "entry_type": entry.entry_type
                }
            )

        # Synthesize knowledge
        synthesis = await self._synthesize_knowledge(input_text, knowledge_results)

        # Add synthesis to context
        self.context_manager.add_context(
            session_id=self.session_id,
            content=json.dumps(synthesis),
            context_type=ContextType.AGENT,
            relevance_score=0.9,
            metadata={"type": "knowledge_synthesis"}
        )

        return synthesis

    async def add_knowledge(
            self,
            content: str,
            entry_type: str,
            metadata: Optional[Dict[str, Any]] = None,
            confidence: float = 1.0
    ) -> Dict:
        """
        Add knowledge to semantic memory.

        Args:
            content: Knowledge content
            entry_type: Type of knowledge entry
            metadata: Additional metadata
            confidence: Confidence in the knowledge (0-1)

        Returns:
            Dict with entry ID and status
        """
        if not self.semantic_memory:
            return {
                "success": False,
                "error": "No semantic memory available"
            }

        # Add to semantic memory
        entry_id = await self.semantic_memory.add(
            content=content,
            entry_type=entry_type,
            metadata=metadata or {},
            confidence=confidence
        )

        return {
            "success": True,
            "entry_id": entry_id,
            "entry_type": entry_type
        }

    async def retrieve_knowledge(
            self,
            query: str,
            limit: int = 5,
            entry_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve knowledge entries based on a query.

        Args:
            query: Search query
            limit: Maximum number of entries to retrieve
            entry_type: Optional entry type filter

        Returns:
            List of knowledge entries with relevance scores
        """
        if not self.semantic_memory:
            return []

        # Search semantic memory
        results = await self.semantic_memory.search(
            query=query,
            limit=limit,
            entry_type=entry_type,
            include_related=True
        )

        # Format results
        formatted_results = []
        for entry, score in results:
            formatted_results.append({
                "id": entry.id,
                "content": entry.content,
                "type": entry.entry_type,
                "relevance": score,
                "confidence": entry.confidence,
                "metadata": entry.metadata
            })

        return formatted_results

    async def _retrieve_knowledge(
            self,
            query: str,
            limit: int = 5
    ) -> List[Tuple[SemanticEntry, float]]:
        """
        Retrieve knowledge from semantic memory.

        Args:
            query: Search query
            limit: Maximum number of entries to retrieve

        Returns:
            List of (entry, score) tuples
        """
        if not self.semantic_memory:
            return []

        return await self.semantic_memory.search(
            query=query,
            limit=limit,
            include_related=True
        )

    async def _synthesize_knowledge(
            self,
            query: str,
            knowledge_results: List[Tuple[SemanticEntry, float]]
    ) -> Dict:
        """
        Synthesize knowledge entries into a coherent response.

        Args:
            query: Original query
            knowledge_results: List of (entry, score) tuples

        Returns:
            Synthesis dictionary
        """
        if not self.llm_client:
            # Create a basic synthesis without LLM
            sources = []
            for entry, score in knowledge_results:
                sources.append({
                    "id": entry.id,
                    "content": self._truncate_content(entry.content),
                    "relevance": score,
                    "type": entry.entry_type
                })

            return {
                "answer": "No LLM client available to synthesize knowledge.",
                "sources": sources,
                "confidence": 0.0,
                "query": query
            }

        # Get formatted context
        formatted_context = self.context_manager.get_formatted_context(self.session_id)

        # Create synthesis prompt
        synthesis_prompt = self._create_synthesis_prompt(query, knowledge_results)

        # Process with LLM
        response = await self._process_with_llm(formatted_context, synthesis_prompt)

        try:
            # Parse JSON response
            result = json.loads(response)
        except json.JSONDecodeError:
            # Fallback if response is not valid JSON
            sources = []
            for entry, score in knowledge_results:
                sources.append({
                    "id": entry.id,
                    "content": self._truncate_content(entry.content),
                    "relevance": score,
                    "type": entry.entry_type
                })

            result = {
                "answer": response,  # Use raw response as answer
                "sources": sources,
                "confidence": 0.5,
                "query": query
            }

        return result

    def _create_synthesis_prompt(
            self,
            query: str,
            knowledge_results: List[Tuple[SemanticEntry, float]]
    ) -> str:
        """
        Create a prompt for knowledge synthesis.

        Args:
            query: Original query
            knowledge_results: List of (entry, score) tuples

        Returns:
            Synthesis prompt
        """
        prompt = f"""
Question: {query}

Relevant knowledge entries:
"""

        for i, (entry, score) in enumerate(knowledge_results):
            content = entry.content
            if isinstance(content, dict) and "text" in content:
                content = content["text"]

            prompt += f"""
Entry {i + 1} (relevance: {score:.2f}, type: {entry.entry_type}):
{content}
"""

        prompt += """
Based on the knowledge entries above, provide a comprehensive answer to the question.
Return your response in JSON format with the following structure:
{
    "answer": "Your detailed answer to the question",
    "sources": [
        {
            "id": 1,
            "summary": "Brief summary of the source",
            "relevance": 0.95
        },
        ...
    ],
    "confidence": 0.9,  // Your confidence in the answer (0-1)
    "missing_information": "Any information you needed but wasn't provided"
}
"""

        return prompt

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

    def _truncate_content(self, content: Any) -> Any:
        """
        Truncate content for inclusion in responses.

        Args:
            content: Content to truncate

        Returns:
            Truncated content
        """
        if isinstance(content, str):
            if len(content) > 200:
                return content[:197] + "..."
            return content
        elif isinstance(content, dict) and "text" in content:
            text = content["text"]
            if len(text) > 200:
                content["text"] = text[:197] + "..."
            return content
        return content
