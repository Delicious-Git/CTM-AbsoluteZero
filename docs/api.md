# CTM-AbsoluteZero API Reference

This document provides a comprehensive reference for the CTM-AbsoluteZero API, allowing you to integrate with the system programmatically.

## Core Classes

### AbsoluteZeroAgent

The main agent class that combines the Proposer and Solver components.

```python
from src.ctm_az_agent import AbsoluteZeroAgent

agent = AbsoluteZeroAgent(
    proposer_model_path="models/proposer",
    solver_model_path="models/solver",
    reward_system=reward_system,
    transfer_adapter=transfer_adapter,
    ctm_interface=ctm_interface,
    config=config
)
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `generate_tasks` | Generates new tasks using the Proposer | `domain`: str, `count`: int, `context`: dict (optional) | List of task dictionaries |
| `solve_task` | Solves a task using the Solver | `task`: dict | Task solution (dict) |
| `train` | Trains the agent on a domain | `domain`: str, `max_iterations`: int, `eval_interval`: int | Training metrics (dict) |
| `evaluate` | Evaluates the agent on a domain | `domain`: str, `num_tasks`: int | Evaluation metrics (dict) |
| `save_state` | Saves agent state to a file | `path`: str | None |
| `load_state` | Loads agent state from a file | `path`: str | None |

### RealCTMInterface

Interface to the CTM (Continuous Thought Machine) components.

```python
from src.ctm.interface import RealCTMInterface

ctm_interface = RealCTMInterface(config={
    "components": ["maze_solver", "image_classifier"]
})
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `execute_task` | Executes a task using appropriate CTM component | `task`: dict | Execution result (dict) |
| `get_component` | Gets a specific CTM component | `component_name`: str | Component instance |
| `list_components` | Lists available components | None | List of component names |
| `health_check` | Checks the health of all components | None | Health status (dict) |

### CompositeRewardSystem

Calculates rewards for tasks based on multiple factors.

```python
from src.rewards.composite import CompositeRewardSystem

reward_system = CompositeRewardSystem(
    novelty_tracker=novelty_tracker,
    skill_pyramid=skill_pyramid,
    phase_controller=phase_controller
)
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `calculate_reward` | Calculates reward for a task | `task`: dict | Reward value (float) |
| `get_component_rewards` | Gets individual component rewards | `task`: dict | Component rewards (dict) |
| `update_weights` | Updates reward component weights | None | New weights (dict) |
| `get_weights` | Gets current reward weights | None | Weights (dict) |

### NeuralTransferAdapter

Adapts knowledge between different domains.

```python
from src.transfer.adapter import NeuralTransferAdapter

adapter = NeuralTransferAdapter(domains=["maze", "vision"])
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `adapt_task` | Adapts a task for a target domain | `task`: dict, `source_domain`: str, `target_domain`: str | Adapted task (dict) |
| `calculate_similarity` | Calculates similarity between domains | `domain1`: str, `domain2`: str | Similarity score (float) |
| `update_mapping` | Updates domain mapping | `source_domain`: str, `target_domain`: str, `mapping`: dict | None |

## DFZ Integration

### CTMAbsoluteZeroPlugin

Plugin for integrating with DFZ.

```python
from src.integration.dfz import CTMAbsoluteZeroPlugin

plugin = CTMAbsoluteZeroPlugin(config_path="configs/dfz.yaml")
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `initialize` | Initializes the plugin | `manager`: DFZ manager (optional) | Success status (bool) |
| `shutdown` | Cleans up resources | None | None |
| `register_hooks` | Registers hooks with DFZ | `manager`: DFZ manager | None |
| `execute_task` | Executes a task | `task`: dict, `context`: dict (optional) | Task execution results (dict) |

### DFZAdapter

Adapter for connecting CTM-AbsoluteZero to DFZ.

```python
from src.integration.dfz import DFZAdapter

adapter = DFZAdapter(dfz_path="/path/to/dfz")
await adapter.initialize()
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `initialize` | Initializes the adapter | None | Success status (bool) |
| `send_message` | Sends a message to DFZ | `message`: str, `context`: dict (optional) | Response (dict) |
| `generate_task` | Generates tasks using DFZ | `domain`: str, `context`: dict (optional) | List of tasks |
| `execute_task` | Executes a task | `task`: dict, `context`: dict (optional) | Task execution results (dict) |

## Utility Classes

### ConfigManager

Manages configuration for the system.

```python
from src.utils.config import ConfigManager

config = ConfigManager(config_path="configs/production.yaml")
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `load_config` | Loads configuration from a file | `config_path`: str | Loaded config (dict) |
| `save_config` | Saves configuration to a file | `config_path`: str (optional) | None |
| `get` | Gets a configuration value | `key`: str, `default`: Any (optional) | Config value |
| `set` | Sets a configuration value | `key`: str, `value`: Any | None |

### DataManager

Manages data storage and retrieval.

```python
from src.utils.data import DataManager

data_manager = DataManager(data_dir="./data")
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `save_json` | Saves data to a JSON file | `data`: dict, `filename`: str | File path (str) |
| `load_json` | Loads data from a JSON file | `filename`: str | Loaded data (dict) |
| `save_pickle` | Saves data to a pickle file | `data`: object, `filename`: str | File path (str) |
| `load_pickle` | Loads data from a pickle file | `filename`: str | Loaded data (object) |

### VectorDatabase

Stores and retrieves vector embeddings.

```python
from src.utils.data import VectorDatabase

vector_db = VectorDatabase(data_dir="./embeddings")
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `add` | Adds a vector to the database | `key`: str, `vector`: numpy.ndarray, `metadata`: dict (optional) | None |
| `get` | Gets a vector from the database | `key`: str | Tuple of (vector, metadata) |
| `search` | Searches for similar vectors | `query_vector`: numpy.ndarray, `top_k`: int, `threshold`: float | List of (key, similarity, metadata) tuples |
| `save` | Saves database to disk | None | None |
| `load` | Loads database from disk | None | None |

## REST API

When running the CTM-AbsoluteZero API server, the following endpoints are available:

### Tasks

#### Generate Tasks

```
POST /api/tasks/generate
```

Request:
```json
{
  "domain": "maze",
  "count": 3,
  "context": {
    "user_id": "user123",
    "difficulty": "medium"
  }
}
```

Response:
```json
{
  "tasks": [
    {
      "id": "task_001",
      "domain": "maze",
      "description": "Solve a 10x10 maze with multiple paths",
      "parameters": {
        "size": [10, 10],
        "difficulty": "medium"
      }
    },
    ...
  ]
}
```

#### Solve Task

```
POST /api/tasks/solve
```

Request:
```json
{
  "task": {
    "id": "task_001",
    "domain": "maze",
    "description": "Solve a 10x10 maze with multiple paths",
    "parameters": {
      "size": [10, 10],
      "difficulty": "medium"
    }
  },
  "context": {
    "user_id": "user123"
  }
}
```

Response:
```json
{
  "task_id": "task_001",
  "status": "success",
  "result": {
    "path": [[0, 0], [1, 0], ...],
    "execution_time": 0.342,
    "metrics": {
      "path_length": 24,
      "efficiency": 0.87
    }
  },
  "duration": 1.254
}
```

### DFZ Integration

#### Send Message

```
POST /api/dfz/message
```

Request:
```json
{
  "message": "I need a task to practice sorting algorithms",
  "context": {
    "user_id": "user123",
    "session_id": "session_456"
  }
}
```

Response:
```json
{
  "text": "I've created a sorting task for you. Would you like to try implementing a merge sort algorithm?",
  "task_id": "task_002",
  "suggestions": [
    "Yes, show me the task details",
    "No, I want a different algorithm"
  ]
}
```

#### Get Conversation History

```
GET /api/dfz/history
```

Response:
```json
{
  "history": [
    {
      "role": "user",
      "content": "I need a task to practice sorting algorithms",
      "timestamp": 1620000000
    },
    {
      "role": "assistant",
      "content": "I've created a sorting task for you. Would you like to try implementing a merge sort algorithm?",
      "timestamp": 1620000010
    }
  ]
}
```

### System

#### Health Check

```
GET /api/health
```

Response:
```json
{
  "status": "ok",
  "components": {
    "proposer": "ok",
    "solver": "ok",
    "ctm_interface": "ok",
    "dfz_integration": "ok"
  },
  "version": "1.0.0",
  "uptime": 3600
}
```

#### Performance Metrics

```
GET /api/metrics
```

Response:
```json
{
  "total_tasks": 156,
  "successful_tasks": 142,
  "failed_tasks": 14,
  "success_rate": 0.91,
  "avg_duration": 2.34,
  "memory_usage": {
    "total": 4096,
    "used": 2048
  },
  "domains": {
    "maze": {
      "tasks": 45,
      "success_rate": 0.93
    },
    "vision": {
      "tasks": 32,
      "success_rate": 0.88
    }
  }
}
```

## Python SDK Usage Examples

### Basic Usage

```python
import asyncio
from src.ctm_az_agent import AbsoluteZeroAgent
from src.utils.config import ConfigManager

# Load configuration
config_manager = ConfigManager("configs/production.yaml")
config = config_manager.to_dict()

# Create the agent (setup omitted for brevity)
agent = AbsoluteZeroAgent(...)

# Generate tasks
tasks = agent.generate_tasks(domain="maze", count=3)

# Solve a task
result = agent.solve_task(tasks[0])

print(f"Task: {tasks[0]['description']}")
print(f"Result: {result}")
```

### DFZ Integration

```python
import asyncio
from src.integration.dfz import DFZAdapter

async def main():
    # Create DFZ adapter
    adapter = DFZAdapter(dfz_path="/path/to/dfz")
    await adapter.initialize()
    
    # Send a message to DFZ
    response = await adapter.send_message(
        "Generate a task for learning quantum algorithms",
        context={"user_id": "user123"}
    )
    
    print(f"Response: {response}")
    
    # Generate tasks
    tasks = await adapter.generate_task(domain="quantum")
    
    print(f"Generated tasks: {tasks}")

# Run the async function
asyncio.run(main())
```

### Custom Component Integration

```python
from src.ctm.interface import RealCTMInterface
from src.ctm.custom_component import CustomComponent

# Create custom component
custom_component = CustomComponent(params={"key": "value"})

# Create CTM interface
ctm_interface = RealCTMInterface(config={})

# Register custom component
ctm_interface.register_component("custom", custom_component)

# Use the component
result = ctm_interface.execute_task({
    "domain": "custom",
    "description": "Test task",
    "parameters": {"test": True}
})

print(f"Result: {result}")
```

## Error Handling

All API methods return structured error responses with the following format:

```json
{
  "error": {
    "code": "invalid_input", 
    "message": "Invalid task format", 
    "details": "Required field 'description' is missing"
  }
}
```

Common error codes include:

- `invalid_input`: Input validation error
- `task_execution_error`: Error during task execution
- `model_error`: Error in language model
- `component_not_found`: Requested component not found
- `integration_error`: Error in DFZ integration
- `internal_error`: Unspecified internal error