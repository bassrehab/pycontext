"""
tests/unit/core/memory/test_procedural_memory.py

Tests for the Procedural Memory module.
"""
import unittest
import asyncio
import time
from unittest.mock import MagicMock, patch
import json

from pycontext.core.memory.procedural_memory import (
    ProceduralMemory, Procedure, ProcedureStep, ProcedureStatus, StepStatus,
    ProcedureExecutor, ProcedureBuilder
)


class TestProcedureStep(unittest.TestCase):
    """Test the ProcedureStep class."""

    def test_creation(self):
        """Test step creation."""
        step = ProcedureStep(
            id="test-step",
            name="Test Step",
            description="A test step",
            action={"type": "test_action", "params": {"key": "value"}}
        )

        self.assertEqual(step.id, "test-step")
        self.assertEqual(step.name, "Test Step")
        self.assertEqual(step.description, "A test step")
        self.assertEqual(step.action["type"], "test_action")
        self.assertEqual(step.status, StepStatus.PENDING)
        self.assertIsNone(step.result)
        self.assertIsNone(step.error)

    def test_duration_property(self):
        """Test duration calculation."""
        step = ProcedureStep(
            id="test-step",
            name="Test Step",
            description="A test step",
            action={"type": "test_action"}
        )

        # Should be 0 if not started
        self.assertEqual(step.duration, 0)

        # Set started time
        step.started_at = time.time() - 5  # 5 seconds ago
        
        # Duration should be approximately 5 seconds
        self.assertAlmostEqual(step.duration, 5, delta=0.1)

        # Set completed time
        step.completed_at = step.started_at + 3  # 3 seconds after start
        
        # Duration should now be exactly 3 seconds
        self.assertEqual(step.duration, 3)


class TestProcedure(unittest.TestCase):
    """Test the Procedure class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a test procedure with steps
        step1 = ProcedureStep(
            id="step1",
            name="Step 1",
            description="First step",
            action={"type": "test_action"}
        )
        
        step2 = ProcedureStep(
            id="step2",
            name="Step 2",
            description="Second step",
            action={"type": "test_action"},
            dependencies=["step1"]
        )
        
        step3 = ProcedureStep(
            id="step3",
            name="Step 3",
            description="Third step",
            action={"type": "test_action"},
            dependencies=["step2"]
        )
        
        self.procedure = Procedure(
            id="test-procedure",
            name="Test Procedure",
            description="A test procedure",
            steps={"step1": step1, "step2": step2, "step3": step3},
            execution_order=["step1", "step2", "step3"],
            step_dependencies={"step1": ["step2"], "step2": ["step3"]}
        )

    def test_creation(self):
        """Test procedure creation."""
        self.assertEqual(self.procedure.id, "test-procedure")
        self.assertEqual(self.procedure.name, "Test Procedure")
        self.assertEqual(self.procedure.description, "A test procedure")
        self.assertEqual(len(self.procedure.steps), 3)
        self.assertEqual(self.procedure.status, ProcedureStatus.PENDING)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        procedure_dict = self.procedure.to_dict()
        
        self.assertEqual(procedure_dict["id"], "test-procedure")
        self.assertEqual(procedure_dict["name"], "Test Procedure")
        self.assertEqual(procedure_dict["status"], "pending")
        self.assertEqual(len(procedure_dict["steps"]), 3)
        self.assertIn("step1", procedure_dict["steps"])
        self.assertEqual(procedure_dict["steps"]["step1"]["name"], "Step 1")

    def test_from_dict(self):
        """Test creation from dictionary."""
        procedure_dict = self.procedure.to_dict()
        new_procedure = Procedure.from_dict(procedure_dict)
        
        self.assertEqual(new_procedure.id, self.procedure.id)
        self.assertEqual(new_procedure.name, self.procedure.name)
        self.assertEqual(new_procedure.status, self.procedure.status)
        self.assertEqual(len(new_procedure.steps), len(self.procedure.steps))
        self.assertEqual(new_procedure.steps["step1"].name, "Step 1")

    def test_get_next_steps(self):
        """Test getting next steps to execute."""
        # Initially only step1 should be ready since it has no dependencies
        next_steps = self.procedure.get_next_steps()
        self.assertEqual(next_steps, ["step1"])
        
        # Mark step1 as completed
        self.procedure.steps["step1"].status = StepStatus.COMPLETED
        
        # Now step2 should be ready
        next_steps = self.procedure.get_next_steps()
        self.assertEqual(next_steps, ["step2"])
        
        # Mark step2 as completed
        self.procedure.steps["step2"].status = StepStatus.COMPLETED
        
        # Now step3 should be ready
        next_steps = self.procedure.get_next_steps()
        self.assertEqual(next_steps, ["step3"])
        
        # Mark step3 as completed
        self.procedure.steps["step3"].status = StepStatus.COMPLETED
        
        # Now no steps should be ready
        next_steps = self.procedure.get_next_steps()
        self.assertEqual(next_steps, [])


class TestProcedureExecutor(unittest.TestCase):
    """Test the ProcedureExecutor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock action handlers
        self.mock_handlers = {
            "test_action": self.mock_test_action,
            "fail_action": self.mock_fail_action,
            "slow_action": self.mock_slow_action
        }
        
        # Create executor with mock handlers
        self.executor = ProcedureExecutor(self.mock_handlers)
        
        # Create a simple procedure
        step1 = ProcedureStep(
            id="step1",
            name="Step 1",
            description="First step",
            action={"type": "test_action", "value": 10}
        )
        
        step2 = ProcedureStep(
            id="step2",
            name="Step 2",
            description="Second step",
            action={"type": "test_action", "value": 20},
            dependencies=["step1"]
        )
        
        self.procedure = Procedure(
            id="test-procedure",
            name="Test Procedure",
            description="A test procedure",
            steps={"step1": step1, "step2": step2},
            execution_order=["step1", "step2"],
            step_dependencies={"step1": ["step2"]}
        )
    
    async def mock_test_action(self, action, inputs):
        """Mock test action handler."""
        value = action.get("value", 0)
        input_value = inputs.get("input_value", 0)
        result = value + input_value
        return {"status": "success", "outputs": {"result": result}}
    
    async def mock_fail_action(self, action, inputs):
        """Mock action that fails."""
        raise Exception("Action failed")
    
    async def mock_slow_action(self, action, inputs):
        """Mock action that takes time."""
        delay = action.get("delay", 1)
        await asyncio.sleep(delay)
        return {"status": "success", "outputs": {"result": "done"}}
    
    def test_execute_procedure(self):
        """Test executing a procedure."""
        # Execute the procedure
        result = asyncio.run(self.executor.execute_procedure(
            self.procedure,
            inputs={"input_value": 5}
        ))
        
        # Check procedure status
        self.assertEqual(result.status, ProcedureStatus.COMPLETED)
        
        # Check step statuses
        self.assertEqual(result.steps["step1"].status, StepStatus.COMPLETED)
        self.assertEqual(result.steps["step2"].status, StepStatus.COMPLETED)
        
        # Check step results
        self.assertEqual(result.steps["step1"].result["outputs"]["result"], 15)  # 10 + 5
        self.assertEqual(result.steps["step2"].result["outputs"]["result"], 25)  # 20 + 5
        
        # Check procedure outputs (should have the result from step2)
        self.assertEqual(result.outputs.get("result"), 25)
    
    def test_execute_procedure_with_failure(self):
        """Test executing a procedure with a failing step."""
        # Modify the procedure to use a failing action
        self.procedure.steps["step2"].action["type"] = "fail_action"
        
        # Execute the procedure
        result = asyncio.run(self.executor.execute_procedure(self.procedure))
        
        # Check procedure status
        self.assertEqual(result.status, ProcedureStatus.FAILED)
        
        # Check step statuses
        self.assertEqual(result.steps["step1"].status, StepStatus.COMPLETED)
        self.assertEqual(result.steps["step2"].status, StepStatus.FAILED)
        
        # Check error
        self.assertIsNotNone(result.steps["step2"].error)
        self.assertIn("Action failed", result.steps["step2"].error)
    
    def test_execute_procedure_with_retry(self):
        """Test executing a procedure with retry logic."""
        # Modify step2 to use fail_action but with retry
        self.procedure.steps["step2"].action["type"] = "fail_action"
        self.procedure.steps["step2"].max_retries = 1
        self.procedure.steps["step2"].retry_delay = 0.1
        
        # Create a mock handler that fails once then succeeds
        retry_count = [0]
        
        async def mock_retry_action(action, inputs):
            retry_count[0] += 1
            if retry_count[0] == 1:
                raise Exception("First attempt failed")
            return {"status": "success", "outputs": {"result": "retry succeeded"}}
        
        # Replace the fail_action handler
        old_handler = self.executor.action_handlers["fail_action"]
        self.executor.action_handlers["fail_action"] = mock_retry_action
        
        try:
            # Execute the procedure
            result = asyncio.run(self.executor.execute_procedure(self.procedure))
            
            # Check procedure status
            self.assertEqual(result.status, ProcedureStatus.COMPLETED)
            
            # Check step statuses
            self.assertEqual(result.steps["step1"].status, StepStatus.COMPLETED)
            self.assertEqual(result.steps["step2"].status, StepStatus.COMPLETED)
            
            # Check retry count
            self.assertEqual(retry_count[0], 2)
            
            # Check result
            self.assertEqual(result.steps["step2"].result["outputs"]["result"], "retry succeeded")
        finally:
            # Restore original handler
            self.executor.action_handlers["fail_action"] = old_handler
    
    def test_step_timeout(self):
        """Test step timeout functionality."""
        # Modify step2 to use slow_action but with a short timeout
        self.procedure.steps["step2"].action = {"type": "slow_action", "delay": 0.5}
        self.procedure.steps["step2"].timeout = 0.1  # 100ms timeout
        
        # Execute the procedure
        result = asyncio.run(self.executor.execute_procedure(self.procedure))
        
        # Check procedure status
        self.assertEqual(result.status, ProcedureStatus.FAILED)
        
        # Check step statuses
        self.assertEqual(result.steps["step1"].status, StepStatus.COMPLETED)
        self.assertEqual(result.steps["step2"].status, StepStatus.FAILED)
        
        # Check timeout error
        self.assertIsNotNone(result.steps["step2"].error)
        self.assertIn("timed out", result.steps["step2"].error.lower())


class TestProceduralMemory(unittest.TestCase):
    """Test the ProceduralMemory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create procedural memory
        self.memory = ProceduralMemory()
        
        # Register mock action handlers
        self.memory.register_action_handler("test_action", self.mock_test_action)
        
        # Create a simple procedure using the builder
        builder = self.memory.create_procedure_builder()
        self.procedure_id = builder\
            .set_name("Test Procedure")\
            .set_description("A test procedure")\
            .add_tag("test")\
            .add_input("input_value", 5)\
            .add_step(
                name="Step 1",
                description="First step",
                action={"type": "test_action", "value": 10}
            )\
            .add_step(
                name="Step 2",
                description="Second step",
                action={"type": "test_action", "value": 20},
                dependencies=["step1"]
            )\
            .build()
    
    async def mock_test_action(self, action, inputs):
        """Mock test action handler."""
        value = action.get("value", 0)
        input_value = inputs.get("input_value", 0)
        result = value + input_value
        return {"status": "success", "outputs": {"result": result}}
    
    def test_add_and_get_procedure(self):
        """Test adding and retrieving procedures."""
        # Get the procedure
        procedure = self.memory.get_procedure(self.procedure_id)
        
        # Check procedure
        self.assertIsNotNone(procedure)
        self.assertEqual(procedure.name, "Test Procedure")
        self.assertEqual(procedure.description, "A test procedure")
        self.assertEqual(len(procedure.steps), 2)
        
        # Check tags index
        procedures_with_tag = self.memory.get_procedures_by_tag("test")
        self.assertEqual(len(procedures_with_tag), 1)
        self.assertEqual(procedures_with_tag[0].id, self.procedure_id)
    
    def test_update_procedure(self):
        """Test updating a procedure."""
        # Get the procedure
        procedure = self.memory.get_procedure(self.procedure_id)
        
        # Update procedure
        procedure.name = "Updated Procedure"
        procedure.tags = ["test", "updated"]
        
        # Save changes
        success = self.memory.update_procedure(procedure)
        
        # Check update was successful
        self.assertTrue(success)
        
        # Get the updated procedure
        updated = self.memory.get_procedure(self.procedure_id)
        
        # Check updates were applied
        self.assertEqual(updated.name, "Updated Procedure")
        self.assertIn("updated", updated.tags)
        
        # Check tag indexing was updated
        procedures_with_tag = self.memory.get_procedures_by_tag("updated")
        self.assertEqual(len(procedures_with_tag), 1)
        self.assertEqual(procedures_with_tag[0].id, self.procedure_id)
    
    def test_remove_procedure(self):
        """Test removing a procedure."""
        # Remove the procedure
        success = self.memory.remove_procedure(self.procedure_id)
        
        # Check removal was successful
        self.assertTrue(success)
        
        # Check procedure is gone
        procedure = self.memory.get_procedure(self.procedure_id)
        self.assertIsNone(procedure)
        
        # Check tag index was updated
        procedures_with_tag = self.memory.get_procedures_by_tag("test")
        self.assertEqual(len(procedures_with_tag), 0)
    
    def test_execute_procedure(self):
        """Test executing a procedure."""
        # Execute the procedure
        result = asyncio.run(self.memory.execute_procedure(
            self.procedure_id,
            inputs={"input_value": 15}
        ))
        
        # Check procedure status
        self.assertEqual(result.status, ProcedureStatus.COMPLETED)
        
        # Check step results
        self.assertEqual(result.steps["step1"].result["outputs"]["result"], 25)  # 10 + 15
        self.assertEqual(result.steps["step2"].result["outputs"]["result"], 35)  # 20 + 15
        
        # Check the stored procedure was updated
        stored = self.memory.get_procedure(self.procedure_id)
        self.assertEqual(stored.status, ProcedureStatus.COMPLETED)
    
    def test_serialization(self):
        """Test serialization to and from dictionary."""
        # Convert to dictionary
        memory_dict = self.memory.to_dict()
        
        # Create new memory from dictionary
        new_memory = ProceduralMemory.from_dict(memory_dict)
        
        # Check procedure was loaded correctly
        procedure = new_memory.get_procedure(self.procedure_id)
        self.assertIsNotNone(procedure)
        self.assertEqual(procedure.name, "Test Procedure")
        
        # Check tag indexing was preserved
        procedures_with_tag = new_memory.get_procedures_by_tag("test")
        self.assertEqual(len(procedures_with_tag), 1)
        self.assertEqual(procedures_with_tag[0].id, self.procedure_id)


class TestProcedureBuilder(unittest.TestCase):
    """Test the ProcedureBuilder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.memory = ProceduralMemory()
        self.builder = self.memory.create_procedure_builder()
    
    def test_builder_creates_valid_procedure(self):
        """Test the builder creates valid procedures."""
        # Build a procedure with the builder
        procedure_id = self.builder\
            .set_name("Builder Test")\
            .set_description("Testing the builder")\
            .set_version("1.1.0")\
            .add_tag("builder")\
            .add_tag("test")\
            .set_timeout(30)\
            .add_input("input1", "value1")\
            .set_metadata("category", "test")\
            .set_retry_policy(max_retries=2, retry_delay=0.5)\
            .add_step(
                name="First Step",
                description="First test step",
                action={"type": "test_action"},
                inputs={"step_input": "value"}
            )\
            .add_step(
                name="Second Step",
                description="Second test step",
                action={"type": "another_action"},
                dependencies=["step1"]
            )\
            .build()
        
        # Get the built procedure
        procedure = self.memory.get_procedure(procedure_id)
        
        # Check procedure attributes
        self.assertEqual(procedure.name, "Builder Test")
        self.assertEqual(procedure.description, "Testing the builder")
        self.assertEqual(procedure.version, "1.1.0")
        self.assertEqual(set(procedure.tags), {"builder", "test"})
        self.assertEqual(procedure.timeout, 30)
        self.assertEqual(procedure.inputs["input1"], "value1")
        self.assertEqual(procedure.metadata["category"], "test")
        self.assertEqual(procedure.retry_policy["max_retries"], 2)
        
        # Check steps
        self.assertEqual(len(procedure.steps), 2)
        
        # Verify step inputs
        first_step_id = procedure.execution_order[0]
        self.assertEqual(procedure.steps[first_step_id].inputs["step_input"], "value")
        
        # Verify dependencies
        second_step_id = procedure.execution_order[1]
        self.assertIn(first_step_id, procedure.steps[second_step_id].dependencies)


if __name__ == "__main__":
    unittest.main()