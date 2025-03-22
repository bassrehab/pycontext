"""
pycontext/core/agents/technical_agent.py

Technical Agent implementation for PyContext.
"""
from typing import Dict, List, Optional, Any
import json

from .base import BaseAgent
from ..mcp.protocol import ContextType


class TechnicalAgent(BaseAgent):
    """
    Technical Agent that diagnoses and resolves technical issues.
    """

    def __init__(
            self,
            agent_id: str = None,
            llm_client: Any = None,
            technical_tools: Dict[str, Any] = None
    ):
        """
        Initialize the Technical Agent.

        Args:
            agent_id: Optional unique identifier
            llm_client: LLM client for generating responses
            technical_tools: Dictionary of technical tools available to the agent
        """
        super().__init__(
            agent_id=agent_id,
            agent_role="technical_agent"
        )
        self.llm_client = llm_client
        self.technical_tools = technical_tools or {}

    async def _load_role_prompt(self) -> str:
        """Load the technical agent's system prompt."""
        return """You are a Technical Support Agent specializing in troubleshooting and resolving technical issues.
Your role is to:
1. Diagnose technical issues from symptoms
2. Design step-by-step troubleshooting plans
3. Interpret diagnostic results
4. Recommend solutions

When helping with technical issues:
1. Be precise in technical language but explain concepts clearly
2. Provide step-by-step instructions
3. Consider the user's technical expertise level
4. Document all diagnostic steps
5. Suggest preventive measures for future issues

Return your analysis in JSON format when requested.
"""

    async def process(self, input_text: str) -> Dict:
        """
        Process user input to diagnose technical issues.

        Args:
            input_text: User's message describing the technical issue

        Returns:
            Dict containing diagnostic information
        """
        # Add user input to context
        await self.add_user_context(input_text, {"type": "technical_issue"})

        # Analyze the issue to determine what tools might help
        tools_to_use = await self._determine_tools_needed(input_text)

        # Run diagnostic tools and add results to context
        for tool_name in tools_to_use:
            if tool_name in self.technical_tools:
                try:
                    tool_result = await self._run_diagnostic_tool(tool_name, input_text)
                    await self.add_tool_context(
                        content=json.dumps(tool_result),
                        tool_name=tool_name,
                        metadata={"type": "diagnostic_result"}
                    )
                except Exception as e:
                    # Log tool error but continue with other tools
                    print(f"Error running diagnostic tool {tool_name}: {e}")

        # Process with LLM to diagnose the issue
        diagnosis = await self._diagnose_issue(input_text)

        # Add diagnosis to context
        self.context_manager.add_context(
            session_id=self.session_id,
            content=json.dumps(diagnosis),
            context_type=ContextType.AGENT,
            relevance_score=0.9,
            metadata={"type": "technical_diagnosis"}
        )

        return diagnosis

    async def diagnose_issue(
            self,
            issue_description: str,
            system_info: Dict = None,
            customer_id: str = None
    ) -> Dict:
        """
        Diagnose a technical issue with additional context.

        Args:
            issue_description: Description of the issue
            system_info: System information
            customer_id: Optional customer identifier

        Returns:
            Dict containing diagnostic information
        """
        # Add issue to context
        metadata = {"type": "technical_issue"}
        if customer_id:
            metadata["customer_id"] = customer_id

        await self.add_user_context(issue_description, metadata)

        # Add system info to context if provided
        if system_info:
            await self.add_tool_context(
                content=json.dumps(system_info),
                tool_name="system_info",
                metadata={"type": "system_information"}
            )

        # Analyze the issue to determine what tools might help
        tools_to_use = await self._determine_tools_needed(issue_description)

        # Run diagnostic tools and add results to context
        for tool_name in tools_to_use:
            if tool_name in self.technical_tools:
                try:
                    tool_result = await self._run_diagnostic_tool(
                        tool_name, issue_description, system_info
                    )
                    await self.add_tool_context(
                        content=json.dumps(tool_result),
                        tool_name=tool_name,
                        metadata={"type": "diagnostic_result"}
                    )
                except Exception as e:
                    # Log tool error but continue with other tools
                    print(f"Error running diagnostic tool {tool_name}: {e}")

        # Process with LLM to diagnose the issue
        diagnosis = await self._diagnose_issue(issue_description)

        # Add diagnosis to context
        self.context_manager.add_context(
            session_id=self.session_id,
            content=json.dumps(diagnosis),
            context_type=ContextType.AGENT,
            relevance_score=0.9,
            metadata={"type": "technical_diagnosis"}
        )

        return diagnosis

    async def suggest_solution(self, diagnosis_id: str) -> Dict:
        """
        Suggest a solution based on a previous diagnosis.

        Args:
            diagnosis_id: ID of the diagnosis block in context

        Returns:
            Dict containing solution information
        """
        # Get the diagnosis from context
        diagnosis = None
        for block in self.context_manager.sessions[self.session_id].blocks:
            if block.id == diagnosis_id:
                try:
                    diagnosis = json.loads(block.content)
                except:
                    # Not a JSON block
                    pass

        if not diagnosis:
            return {
                "error": "Diagnosis not found",
                "solution": "Unable to suggest a solution without a diagnosis"
            }

        # Process with LLM to suggest a solution
        solution = await self._suggest_solution(diagnosis)

        # Add solution to context
        self.context_manager.add_context(
            session_id=self.session_id,
            content=json.dumps(solution),
            context_type=ContextType.AGENT,
            relevance_score=0.9,
            metadata={"type": "technical_solution"}
        )

        return solution

    async def _determine_tools_needed(self, issue_description: str) -> List[str]:
        """
        Determine which diagnostic tools might be helpful for this issue.

        Args:
            issue_description: Description of the issue

        Returns:
            List of tool names that might be helpful
        """
        # In a real implementation, this would use the LLM to determine which tools to use
        # For now, return a simple set of tools based on keywords
        tools = []

        issue_lower = issue_description.lower()

        if any(word in issue_lower for word in ["network", "internet", "wifi", "connection"]):
            tools.append("network_diagnostic")

        if any(word in issue_lower for word in ["hardware", "device", "computer", "laptop"]):
            tools.append("hardware_diagnostic")

        if any(word in issue_lower for word in ["software", "application", "program", "app"]):
            tools.append("software_diagnostic")

        if any(word in issue_lower for word in ["security", "virus", "malware", "hack"]):
            tools.append("security_diagnostic")

        # If no specific tools matched, add a general diagnostic tool
        if not tools and "system_check" in self.technical_tools:
            tools.append("system_check")

        return tools

    async def _run_diagnostic_tool(
            self,
            tool_name: str,
            issue_description: str,
            system_info: Dict = None
    ) -> Dict:
        """
        Run a diagnostic tool.

        Args:
            tool_name: Name of the tool to run
            issue_description: Description of the issue
            system_info: Optional system information

        Returns:
            Tool results as a dictionary
        """
        if tool_name not in self.technical_tools:
            return {"error": f"Tool {tool_name} not available"}

        tool = self.technical_tools[tool_name]

        try:
            # Assuming tool is a callable that accepts issue_description and system_info
            result = await tool(issue_description, system_info)
            return result
        except Exception as e:
            return {
                "error": f"Error running tool {tool_name}: {str(e)}",
                "tool_name": tool_name
            }

    async def _diagnose_issue(self, issue_description: str) -> Dict:
        """
        Diagnose an issue using the LLM and available context.

        Args:
            issue_description: Description of the issue

        Returns:
            Diagnosis dictionary
        """
        if not self.llm_client:
            # Provide a placeholder diagnosis if no LLM is available
            return {
                "root_cause": "Unable to determine without an LLM",
                "confidence": 0.0,
                "recommended_steps": [
                    "Please provide an LLM client to enable diagnosis"
                ],
                "potential_issues": []
            }

        # Get formatted context
        formatted_context = self.context_manager.get_formatted_context(self.session_id)

        # Create diagnosis prompt
        diagnosis_prompt = """Based on the issue description and any diagnostic results, provide a technical diagnosis with:
1. Root Cause: The most likely cause of the issue
2. Confidence: How certain you are about this diagnosis (0-1)
3. Recommended Steps: Step-by-step troubleshooting plan
4. Potential Issues: Other possible causes if the primary diagnosis is incorrect

Return your analysis in JSON format with the following structure:
{
    "root_cause": "string",
    "confidence": float,
    "recommended_steps": ["step1", "step2", ...],
    "potential_issues": ["issue1", "issue2", ...]
}
"""

        # Process with LLM (implementation depends on the specific LLM client)
        response = await self._process_with_llm(formatted_context, diagnosis_prompt)

        try:
            # Parse JSON response
            result = json.loads(response)
        except json.JSONDecodeError:
            # Fallback if response is not valid JSON
            result = {
                "root_cause": "Unable to parse diagnosis",
                "confidence": 0.5,
                "recommended_steps": ["Retry the diagnosis"],
                "potential_issues": ["LLM response format error"]
            }

        return result

    async def _suggest_solution(self, diagnosis: Dict) -> Dict:
        """
        Suggest a solution based on a diagnosis.

        Args:
            diagnosis: Diagnosis dictionary

        Returns:
            Solution dictionary
        """
        if not self.llm_client:
            # Provide a placeholder solution if no LLM is available
            return {
                "solution_description": "Unable to suggest a solution without an LLM",
                "steps": [
                    "Please provide an LLM client to enable solution generation"
                ],
                "alternative_solutions": []
            }

        # Get formatted context
        formatted_context = self.context_manager.get_formatted_context(self.session_id)

        # Create solution prompt
        solution_prompt = f"""Based on the diagnosis:
{json.dumps(diagnosis, indent=2)}

Provide a comprehensive solution with:
1. Solution Description: A clear explanation of the solution
2. Steps: Detailed step-by-step instructions to implement the solution
3. Alternative Solutions: Other approaches if the primary solution doesn't work

Return your solution in JSON format with the following structure:
{{
    "solution_description": "string",
    "steps": ["step1", "step2", ...],
    "alternative_solutions": [
        {{
            "description": "string",
            "steps": ["step1", "step2", ...]
        }},
        ...
    ]
}}
"""

        # Process with LLM
        response = await self._process_with_llm(formatted_context, solution_prompt)

        try:
            # Parse JSON response
            result = json.loads(response)
        except json.JSONDecodeError:
            # Fallback if response is not valid JSON
            result = {
                "solution_description": "Unable to parse solution",
                "steps": ["Retry the solution generation"],
                "alternative_solutions": []
            }

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
