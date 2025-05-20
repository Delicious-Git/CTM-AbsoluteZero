#!/bin/bash
# Setup script for integration tests

set -e

# Create directories
mkdir -p tests/integration/data
mkdir -p tests/integration/logs
mkdir -p tests/integration/models
mkdir -p tests/integration/configs

# Create test config
cat > tests/integration/configs/test_config.yaml << EOL
# Test configuration for CTM-AbsoluteZero

agent:
  proposer_model_path: "tests/integration/models/test_proposer"
  solver_model_path: "tests/integration/models/test_solver"
  max_tokens: 256
  temperature: 0.7
  top_p: 0.95
  task_history_size: 10
  max_attempts: 2
  timeout: 30

ctm:
  interface_type: "mock"
  connection_timeout: 5
  retry_count: 1
  components:
    - "maze_solver"
    - "quantum_sim"
  metrics_enabled: true

rewards:
  embedding_dim: 64
  novelty_threshold: 0.2
  skill_levels: 3
  hyperparams:
    novelty_weight: 0.3
    progress_weight: 0.3
    success_weight: 0.3
    efficiency_weight: 0.1
    max_reward: 10.0
    min_reward: -5.0

domains:
  - "maze"
  - "quantum"
  - "test"

logging:
  level: "debug"
  file: "tests/integration/logs/test.log"
EOL

# Create mock models
mkdir -p tests/integration/models/test_proposer
mkdir -p tests/integration/models/test_solver

# Create basic model files for testing
cat > tests/integration/models/test_proposer/config.json << EOL
{
    "model_type": "gpt2",
    "architectures": ["GPT2LMHeadModel"],
    "vocab_size": 1000,
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_hidden_layers": 2
}
EOL

cp tests/integration/models/test_proposer/config.json tests/integration/models/test_solver/config.json

# Create an integration test config for DFZ
mkdir -p tests/integration/dfz
cat > tests/integration/dfz/test_dfz_config.yaml << EOL
# Test DFZ integration configuration

dfz:
  dfz_path: "tests/integration/dfz/mock_dfz"
  plugin_enabled: true
  model_sync_interval: 10
  conversation_history_size: 5
  task_registry_size: 10

agent:
  proposer_model_path: "tests/integration/models/test_proposer"
  solver_model_path: "tests/integration/models/test_solver"
  max_tokens: 256
  temperature: 0.7
EOL

# Create mock DFZ structure
mkdir -p tests/integration/dfz/mock_dfz/evolution
touch tests/integration/dfz/mock_dfz/evolution/__init__.py

# Create a simple test database for test data
cat > tests/integration/data/test_database.json << EOL
{
  "tasks": [
    {
      "id": "test_task_1",
      "domain": "maze",
      "description": "Test maze task",
      "parameters": {
        "size": [5, 5],
        "difficulty": "easy"
      }
    },
    {
      "id": "test_task_2",
      "domain": "quantum",
      "description": "Test quantum task",
      "parameters": {
        "algorithm": "grover",
        "num_qubits": 2
      }
    }
  ],
  "execution_results": [
    {
      "task_id": "test_task_1",
      "success": true,
      "execution_time": 0.5,
      "solution": {"path": [[0, 0], [0, 1], [1, 1], [2, 1], [2, 2]]}
    }
  ]
}
EOL

# Create basic integration test 
cat > tests/integration/test_basic_integration.py << EOL
"""
Basic integration tests for CTM-AbsoluteZero.
"""
import os
import sys
import unittest
import json
import yaml
import asyncio
import pytest
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.utils.config import ConfigManager
from src.ctm.interface import RealCTMInterface
from src.ctm_az_agent import AbsoluteZeroAgent

class TestBasicIntegration(unittest.TestCase):
    """Basic integration tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Load test configuration
        cls.config_path = os.path.join(os.path.dirname(__file__), "configs", "test_config.yaml")
        cls.config_manager = ConfigManager(cls.config_path)
        cls.config = cls.config_manager.to_dict()
        
        # Mock setup would go here in a real test
        # We're keeping this simple for the example
    
    def test_config_loading(self):
        """Test configuration loading."""
        self.assertIsNotNone(self.config)
        self.assertIn("agent", self.config)
        self.assertIn("ctm", self.config)
        self.assertIn("rewards", self.config)
        self.assertIn("domains", self.config)
    
    def test_data_access(self):
        """Test data access."""
        data_path = os.path.join(os.path.dirname(__file__), "data", "test_database.json")
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self.assertIn("tasks", data)
        self.assertIn("execution_results", data)
        self.assertEqual(len(data["tasks"]), 2)
        self.assertEqual(data["tasks"][0]["domain"], "maze")
    
    @pytest.mark.asyncio
    async def test_async_operation(self):
        """Test async operation."""
        # Just a simple async test to verify the setup
        await asyncio.sleep(0.1)
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
EOL

# Create DFZ integration test
cat > tests/integration/test_dfz_integration.py << EOL
"""
DFZ integration tests for CTM-AbsoluteZero.
"""
import os
import sys
import unittest
import json
import yaml
import asyncio
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.utils.config import ConfigManager
from src.integration.dfz import DFZAdapter, CTMAbsoluteZeroPlugin

class TestDFZIntegration(unittest.TestCase):
    """DFZ integration tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Load test configuration
        cls.config_path = os.path.join(os.path.dirname(__file__), "dfz", "test_dfz_config.yaml")
        cls.config_manager = ConfigManager(cls.config_path)
        cls.config = cls.config_manager.to_dict()
    
    @pytest.mark.asyncio
    @patch("src.integration.dfz.DFZAdapter.initialize")
    async def test_dfz_adapter_creation(self, mock_initialize):
        """Test DFZ adapter creation."""
        mock_initialize.return_value = asyncio.Future()
        mock_initialize.return_value.set_result(True)
        
        adapter = DFZAdapter(
            dfz_path=self.config["dfz"]["dfz_path"],
            config=self.config
        )
        success = await adapter.initialize()
        
        self.assertTrue(success)
        mock_initialize.assert_called_once()
    
    def test_plugin_creation(self):
        """Test plugin creation."""
        plugin = CTMAbsoluteZeroPlugin(
            config_path=self.config_path,
            standalone=True
        )
        
        self.assertEqual(plugin.name, "ctm_az")
        self.assertEqual(plugin.version, "1.0.0")
        self.assertIsNotNone(plugin.description)


if __name__ == "__main__":
    unittest.main()
EOL

# Make the script executable
chmod +x tests/integration/setup_integration_test.sh

echo "Integration test setup complete!"
echo "Run the tests with: pytest tests/integration/"