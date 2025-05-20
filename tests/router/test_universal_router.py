"""
Tests for Universal Router.
"""
import os
import sys
import unittest
import asyncio
from typing import Dict, List, Any, Optional

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.router.universal_router import UniversalRouter, Task, SolverInterface


class MockSolver(SolverInterface):
    """Mock solver for testing."""
    
    def __init__(self, name: str, domains: List[str], config: Dict[str, Any] = None):
        """Initialize the mock solver."""
        super().__init__(name, config)
        self.domains = domains
        self.solve_count = 0
        self.solved_tasks = []
        self.delay = config.get("delay", 0.0) if config else 0.0
        self.success = config.get("success", True) if config else True
    
    async def solve(self, task: Task) -> Dict[str, Any]:
        """Solve a task."""
        self.solve_count += 1
        self.solved_tasks.append(task)
        
        # Simulate processing time
        await asyncio.sleep(self.delay)
        
        if not self.success:
            raise Exception("Mock solver failure")
        
        return {
            "status": "success",
            "result": {"completed": True},
            "metrics": {
                "execution_time": self.delay,
                "efficiency": 0.9
            }
        }
    
    def can_solve(self, task: Task) -> bool:
        """Check if this solver can solve the given task."""
        return task.domain in self.domains
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get solver capabilities."""
        return {
            "domains": self.domains,
            "max_complexity": 10
        }


class TestUniversalRouter(unittest.TestCase):
    """Tests for the Universal Router."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create router
        self.router = UniversalRouter()
        
        # Create solvers
        self.quantum_solver = MockSolver("quantum_solver", ["quantum"], {"delay": 0.05})
        self.maze_solver = MockSolver("maze_solver", ["maze"], {"delay": 0.03})
        self.multi_solver = MockSolver("multi_solver", ["sorting", "general"], {"delay": 0.02})
        self.failing_solver = MockSolver("failing_solver", ["failure"], {"success": False})
        
        # Register solvers
        self.router.register_solver(self.quantum_solver)
        self.router.register_solver(self.maze_solver)
        self.router.register_solver(self.multi_solver)
        self.router.register_solver(self.failing_solver)
        
        # Create tasks
        self.quantum_task = Task(
            task_id="quantum_1",
            domain="quantum",
            description="Test quantum task",
            parameters={"num_qubits": 4}
        )
        
        self.maze_task = Task(
            task_id="maze_1",
            domain="maze",
            description="Test maze task",
            parameters={"size": [10, 10]}
        )
        
        self.sorting_task = Task(
            task_id="sorting_1",
            domain="sorting",
            description="Test sorting task",
            parameters={"array_size": 1000}
        )
        
        self.failure_task = Task(
            task_id="failure_1",
            domain="failure",
            description="Test failure task",
            parameters={}
        )
        
        self.unknown_task = Task(
            task_id="unknown_1",
            domain="unknown",
            description="Test unknown task",
            parameters={}
        )
    
    async def async_setUp(self):
        """Async setup."""
        # Start router
        await self.router.start(num_workers=2)
    
    async def async_tearDown(self):
        """Async teardown."""
        # Stop router
        await self.router.stop()
    
    def test_task(self):
        """Test Task class."""
        # Test to_dict
        task_dict = self.quantum_task.to_dict()
        self.assertEqual(task_dict["task_id"], "quantum_1")
        self.assertEqual(task_dict["domain"], "quantum")
        self.assertEqual(task_dict["description"], "Test quantum task")
        self.assertEqual(task_dict["parameters"]["num_qubits"], 4)
        
        # Test from_dict
        task = Task.from_dict(task_dict)
        self.assertEqual(task.task_id, "quantum_1")
        self.assertEqual(task.domain, "quantum")
        self.assertEqual(task.description, "Test quantum task")
        self.assertEqual(task.parameters["num_qubits"], 4)
    
    def test_solver_registration(self):
        """Test solver registration."""
        # Check solvers are registered
        self.assertIn("quantum_solver", self.router.solvers)
        self.assertIn("maze_solver", self.router.solvers)
        self.assertIn("multi_solver", self.router.solvers)
        
        # Check domain mappings
        self.assertIn("quantum", self.router.domain_mappings)
        self.assertIn("maze", self.router.domain_mappings)
        self.assertIn("sorting", self.router.domain_mappings)
        self.assertIn("general", self.router.domain_mappings)
        self.assertIn("failure", self.router.domain_mappings)
        
        # Check solver in domain mapping
        self.assertIn("quantum_solver", self.router.domain_mappings["quantum"])
        self.assertIn("maze_solver", self.router.domain_mappings["maze"])
        self.assertIn("multi_solver", self.router.domain_mappings["sorting"])
    
    async def test_execute_task(self):
        """Test executing a task."""
        # Execute quantum task
        result = await self.router.execute(self.quantum_task)
        
        # Check result
        self.assertEqual(result["task_id"], "quantum_1")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["solver"], "quantum_solver")
        self.assertTrue(result["result"]["completed"])
        
        # Check solver was called
        self.assertEqual(self.quantum_solver.solve_count, 1)
        self.assertEqual(self.quantum_solver.solved_tasks[0].task_id, "quantum_1")
    
    async def test_execute_multiple_tasks(self):
        """Test executing multiple tasks."""
        # Execute tasks
        results = await asyncio.gather(
            self.router.execute(self.quantum_task),
            self.router.execute(self.maze_task),
            self.router.execute(self.sorting_task)
        )
        
        # Check results
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["task_id"], "quantum_1")
        self.assertEqual(results[0]["solver"], "quantum_solver")
        self.assertEqual(results[1]["task_id"], "maze_1")
        self.assertEqual(results[1]["solver"], "maze_solver")
        self.assertEqual(results[2]["task_id"], "sorting_1")
        self.assertEqual(results[2]["solver"], "multi_solver")
        
        # Check solvers were called
        self.assertEqual(self.quantum_solver.solve_count, 1)
        self.assertEqual(self.maze_solver.solve_count, 1)
        self.assertEqual(self.multi_solver.solve_count, 1)
    
    async def test_task_queue(self):
        """Test task queue processing."""
        # Start router
        await self.router.start(num_workers=2)
        
        # Add tasks to queue
        self.router.add_task(self.quantum_task)
        self.router.add_task(self.maze_task)
        self.router.add_task(self.sorting_task)
        
        # Wait for tasks to be processed
        await asyncio.sleep(0.2)
        
        # Check solvers were called
        self.assertEqual(self.quantum_solver.solve_count, 1)
        self.assertEqual(self.maze_solver.solve_count, 1)
        self.assertEqual(self.multi_solver.solve_count, 1)
        
        # Stop router
        await self.router.stop()
    
    async def test_failing_task(self):
        """Test failing task."""
        # Execute failing task
        result = await self.router.execute(self.failure_task)
        
        # Check result
        self.assertEqual(result["task_id"], "failure_1")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["solver"], "failing_solver")
        self.assertIn("error", result)
        
        # Check solver was called
        self.assertEqual(self.failing_solver.solve_count, 1)
    
    async def test_unknown_domain(self):
        """Test task with unknown domain."""
        # Execute unknown task
        result = await self.router.execute(self.unknown_task)
        
        # Check result
        self.assertEqual(result["task_id"], "unknown_1")
        self.assertEqual(result["status"], "error")
        self.assertIn("error", result)
        self.assertIn("No suitable solver found", result["error"])
    
    async def test_stats(self):
        """Test router statistics."""
        # Execute tasks
        await self.router.execute(self.quantum_task)
        await self.router.execute(self.maze_task)
        await self.router.execute(self.sorting_task)
        await self.router.execute(self.failure_task)
        await self.router.execute(self.unknown_task)
        
        # Get stats
        stats = self.router.get_stats()
        
        # Check stats
        self.assertEqual(stats["total_tasks"], 5)
        self.assertEqual(stats["successful_tasks"], 3)
        self.assertEqual(stats["failed_tasks"], 2)
        self.assertGreaterEqual(len(stats["task_durations"]), 3)  # At least 3 successful tasks
        
        # Check domain stats
        self.assertEqual(stats["domains"]["quantum"]["total_tasks"], 1)
        self.assertEqual(stats["domains"]["quantum"]["successful_tasks"], 1)
        self.assertEqual(stats["domains"]["maze"]["total_tasks"], 1)
        self.assertEqual(stats["domains"]["maze"]["successful_tasks"], 1)
        self.assertEqual(stats["domains"]["sorting"]["total_tasks"], 1)
        self.assertEqual(stats["domains"]["sorting"]["successful_tasks"], 1)
        self.assertEqual(stats["domains"]["failure"]["total_tasks"], 1)
        self.assertEqual(stats["domains"]["failure"]["failed_tasks"], 1)
        self.assertEqual(stats["domains"]["unknown"]["total_tasks"], 1)
        self.assertEqual(stats["domains"]["unknown"]["failed_tasks"], 1)
        
        # Check resources
        self.assertIn("resources", stats)
        self.assertEqual(stats["resources"]["active_tasks"], 0)


# Run tests asynchronously
def run_async_tests():
    """Run async tests."""
    import unittest
    import asyncio
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add async tests
    for method_name in dir(TestUniversalRouter):
        if method_name.startswith("test_"):
            suite.addTest(TestUniversalRouter(method_name))
    
    # Run tests
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == "__main__":
    # Run synchronous tests
    unittest.main()