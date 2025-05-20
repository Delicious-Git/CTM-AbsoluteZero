"""
Tests for reward system components.
"""
import os
import sys
import unittest
import numpy as np

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.rewards.novelty import SemanticNoveltyTracker
from src.rewards.progress import SkillPyramid
from src.rewards.composite import CompositeRewardSystem
from src.transfer.phase import PhaseController


class TestNoveltyTracker(unittest.TestCase):
    """Tests for the SemanticNoveltyTracker class."""
    
    def setUp(self):
        """Set up test environment."""
        self.tracker = SemanticNoveltyTracker(
            embedding_dim=3,
            novelty_threshold=0.3
        )
    
    def test_compute_novelty(self):
        """Test novelty computation."""
        # Add initial tasks
        task1 = {"description": "Task 1", "embedding": np.array([1.0, 0.0, 0.0])}
        task2 = {"description": "Task 2", "embedding": np.array([0.0, 1.0, 0.0])}
        
        # Get novelty for a completely new task
        new_task = {"description": "New Task", "embedding": np.array([0.0, 0.0, 1.0])}
        novelty = self.tracker.compute_novelty(new_task)
        self.assertEqual(novelty, 1.0)  # Should be maximum novelty since memory is empty
        
        # Add tasks to memory
        self.tracker.add_to_memory(task1)
        self.tracker.add_to_memory(task2)
        
        # Test novelty for similar task
        similar_task = {"description": "Similar Task", "embedding": np.array([0.9, 0.1, 0.0])}
        novelty = self.tracker.compute_novelty(similar_task)
        self.assertLess(novelty, self.tracker.novelty_threshold)  # Should be below threshold
        
        # Test novelty for different task
        different_task = {"description": "Different Task", "embedding": np.array([0.0, 0.0, 1.0])}
        novelty = self.tracker.compute_novelty(different_task)
        self.assertGreater(novelty, self.tracker.novelty_threshold)  # Should be above threshold
    
    def test_compute_embedding(self):
        """Test embedding computation."""
        # Create a task without embedding
        task = {"description": "Test task"}
        
        # Compute embedding
        embedding = self.tracker.compute_embedding(task)
        
        # Check embedding shape
        self.assertEqual(embedding.shape, (self.tracker.embedding_dim,))
    
    def test_add_to_memory(self):
        """Test adding tasks to memory."""
        # Create test tasks
        task1 = {"description": "Task 1", "embedding": np.array([1.0, 0.0, 0.0])}
        task2 = {"description": "Task 2", "embedding": np.array([0.0, 1.0, 0.0])}
        
        # Add to memory
        self.tracker.add_to_memory(task1)
        self.tracker.add_to_memory(task2)
        
        # Check memory size
        self.assertEqual(len(self.tracker.memory), 2)
        
        # Check memory contents
        self.assertTrue(np.array_equal(self.tracker.memory[0]["embedding"], task1["embedding"]))
        self.assertTrue(np.array_equal(self.tracker.memory[1]["embedding"], task2["embedding"]))


class TestSkillPyramid(unittest.TestCase):
    """Tests for the SkillPyramid class."""
    
    def setUp(self):
        """Set up test environment."""
        self.domains = ["math", "language", "science"]
        self.pyramid = SkillPyramid(domains=self.domains, levels=5)
    
    def test_calculate_skill_level(self):
        """Test skill level calculation."""
        # Test initial levels
        for domain in self.domains:
            self.assertEqual(self.pyramid.get_skill_level(domain), 1)
        
        # Update some skill levels
        self.pyramid.update_skill_level("math", 2)
        self.pyramid.update_skill_level("language", 3)
        
        # Test updated levels
        self.assertEqual(self.pyramid.get_skill_level("math"), 2)
        self.assertEqual(self.pyramid.get_skill_level("language"), 3)
        self.assertEqual(self.pyramid.get_skill_level("science"), 1)
    
    def test_reward_scaling(self):
        """Test reward scaling based on skill level."""
        # Set up different skill levels
        self.pyramid.update_skill_level("math", 1)
        self.pyramid.update_skill_level("language", 3)
        self.pyramid.update_skill_level("science", 5)
        
        # Test reward scaling for different domains
        math_reward = self.pyramid.scale_reward("math", 1.0)
        language_reward = self.pyramid.scale_reward("language", 1.0)
        science_reward = self.pyramid.scale_reward("science", 1.0)
        
        # Higher level should give lower reward for same achievement
        self.assertGreater(math_reward, language_reward)
        self.assertGreater(language_reward, science_reward)
    
    def test_challenge_factor(self):
        """Test challenge factor calculation."""
        # Set a skill level
        self.pyramid.update_skill_level("math", 2)
        
        # Test challenge factors
        low_challenge = self.pyramid.calculate_challenge_factor("math", 1)
        medium_challenge = self.pyramid.calculate_challenge_factor("math", 2)
        high_challenge = self.pyramid.calculate_challenge_factor("math", 4)
        
        # Higher challenge should give higher factor
        self.assertLess(low_challenge, medium_challenge)
        self.assertLess(medium_challenge, high_challenge)


class TestCompositeRewardSystem(unittest.TestCase):
    """Tests for the CompositeRewardSystem class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create components
        self.novelty_tracker = SemanticNoveltyTracker(embedding_dim=3)
        self.skill_pyramid = SkillPyramid(domains=["math", "language", "science"])
        self.phase_controller = PhaseController()
        
        # Create reward system
        self.reward_system = CompositeRewardSystem(
            novelty_tracker=self.novelty_tracker,
            skill_pyramid=self.skill_pyramid,
            phase_controller=self.phase_controller
        )
    
    def test_calculate_reward(self):
        """Test reward calculation."""
        # Create a test task
        task = {
            "domain": "math",
            "description": "Solve equation",
            "embedding": np.array([1.0, 0.0, 0.0]),
            "challenge_level": 2,
            "success": True,
            "execution_time": 5.0
        }
        
        # Calculate reward
        reward = self.reward_system.calculate_reward(task)
        
        # Reward should be positive for successful task
        self.assertGreater(reward, 0)
        
        # Test unsuccessful task
        fail_task = {**task, "success": False}
        fail_reward = self.reward_system.calculate_reward(fail_task)
        
        # Reward should be lower for unsuccessful task
        self.assertLess(fail_reward, reward)
    
    def test_update_weights(self):
        """Test weight updating."""
        # Get initial weights
        initial_weights = self.reward_system.get_weights()
        
        # Update phase
        self.phase_controller.set_phase("exploitation")
        
        # Update weights
        self.reward_system.update_weights()
        
        # Get updated weights
        updated_weights = self.reward_system.get_weights()
        
        # Weights should be different after phase change
        self.assertNotEqual(initial_weights, updated_weights)
    
    def test_component_contributions(self):
        """Test individual component contributions to reward."""
        # Create a test task
        task = {
            "domain": "math",
            "description": "Solve equation",
            "embedding": np.array([1.0, 0.0, 0.0]),
            "challenge_level": 2,
            "success": True,
            "execution_time": 5.0
        }
        
        # Get component rewards
        rewards = self.reward_system.get_component_rewards(task)
        
        # All components should contribute
        self.assertIn("novelty", rewards)
        self.assertIn("progress", rewards)
        self.assertIn("success", rewards)
        self.assertIn("efficiency", rewards)
        
        # Calculate total reward
        total_reward = self.reward_system.calculate_reward(task)
        
        # Total reward should be a weighted sum of components
        weights = self.reward_system.get_weights()
        weighted_sum = sum(rewards[k] * weights[k] for k in rewards)
        self.assertAlmostEqual(total_reward, weighted_sum, places=6)


if __name__ == "__main__":
    unittest.main()