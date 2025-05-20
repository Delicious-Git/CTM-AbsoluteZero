# Claude Code Agentic DeepSeek Brain Framework

This document describes the Claude Code Agentic DeepSeek Brain Framework, an extension to CTM-AbsoluteZero that integrates multiple AI models (Anthropic Claude and DeepSeek) to create a cost-effective, high-performance agentic system.

## Overview

The Agentic Brain Framework provides:

1. **Multi-LLM Support**: Use different LLMs (Claude, DeepSeek) based on cost-performance needs
2. **Unified Interface**: Consistent API across different model providers
3. **DFZ Integration**: Seamless connection with the DFZ conversational intelligence system
4. **Cost Optimization**: DeepSeek integration reduces cost by up to 80x compared to Claude
5. **Comparative Analysis**: Tools to compare performance and cost-effectiveness between models

## Architecture

The framework consists of:

```
┌───────────────────────────────────┐
│        BrainFramework             │
│                                   │
│  ┌───────────┐    ┌───────────┐   │
│  │  Claude   │    │ DeepSeek  │   │
│  │  Agent    │    │  Agent    │   │
│  └───────────┘    └───────────┘   │
│          │              │         │
│          └──────┬───────┘         │
│                 ▼                 │
│        ┌────────────────┐         │
│        │ CTM-AbsoluteZero│        │
│        └────────────────┘         │
│                 │                 │
│                 ▼                 │
│        ┌────────────────┐         │
│        │  DFZ Interface │         │
│        └────────────────┘         │
└───────────────────────────────────┘
```

### Components

1. **AgenController**: Manages different agent types and provides a unified interface
2. **DeepSeekAgentic/ClaudeAgentic**: Model-specific implementations of agentic functionality
3. **BrainFramework**: Top-level orchestrator that combines all components

## Cost Comparison

The DeepSeek integration provides significant cost savings:

| Model | Cost per 1K tokens | Cost for typical task cycle | Relative Cost |
|-------|-------------------|----------------------------|---------------|
| Claude-3-Opus | $0.008 | $0.064 (8K tokens) | 80x |
| DeepSeek-Chat | $0.0001 | $0.0008 (8K tokens) | 1x |

While Claude typically offers better quality, the DeepSeek integration provides excellent value for many use cases, especially during development or for non-critical tasks.

## Installation & Setup

### Prerequisites

- CTM-AbsoluteZero installed and configured
- API keys for Anthropic Claude and/or DeepSeek (optional)
- DFZ system (optional)

### Configuration

Create or modify the configuration file at `configs/agentic_brain.yaml`:

```yaml
default_agent: "claude"  # or "deepseek" for lower cost

agents:
  claude:
    type: "claude"
    model: "claude-3-opus-20240229"
    # Set ANTHROPIC_API_KEY environment variable
    
  deepseek:
    type: "deepseek"
    model: "deepseek-chat-32k"
    # Set DEEPSEEK_API_KEY environment variable
```

### Environment Variables

Set API keys as environment variables:

```bash
export ANTHROPIC_API_KEY="your_claude_api_key"
export DEEPSEEK_API_KEY="your_deepseek_api_key"
```

## Usage

### Basic Example

```python
from src.agentic.framework import BrainFramework

# Create framework
framework = BrainFramework(config_path="configs/agentic_brain.yaml")

# Run a cycle with default agent (Claude)
result = await framework.run_cycle(
    domain="quantum",
    difficulty="medium"
)

# Run with DeepSeek for cost savings
result = await framework.run_cycle(
    domain="sorting",
    difficulty="hard",
    agent_name="deepseek"
)
```

### Command Line Usage

Use the example script to test different features:

```bash
# Run a single cycle
python examples/agentic_brain_example.py cycle --domain quantum --difficulty medium

# Compare Claude and DeepSeek
python examples/agentic_brain_example.py compare --domain maze

# Test DFZ integration
python examples/agentic_brain_example.py dfz --dfz-path /path/to/dfz
```

## DFZ Integration

The framework integrates with the DFZ conversational intelligence system:

```python
# Send a message to DFZ
response = await framework.send_to_dfz(
    "I need a task related to sorting algorithms",
    context={"difficulty": "medium"}
)

# Generate tasks from DFZ
tasks = await framework.generate_dfz_task(domain="sorting")

# Execute a task through DFZ
result = await framework.execute_dfz_task(tasks[0])
```

## Computational Advantage Framework

The Agentic Brain Framework implements the Computational Advantage Framework principles:

1. **Compute Efficiency**: Optimize for performance per compute dollar
2. **Agent Augmentation**: Agents enhance developer productivity
3. **Task Auto-generation**: Self-learning through generated tasks
4. **Cross-domain Transfer**: Knowledge sharing between domains
5. **Hot-swappable Components**: Modular design with pluggable components

## Integration with CTM-AZ Ultimate Fusion

The framework is designed to integrate with the full CTM-AZ Ultimate Fusion system:

1. **Trinity Stack**: AI_Docs, Specs, and .claude directories for persistent memory
2. **Universal Router**: Task routing to appropriate modules
3. **X-Trust Protocol**: Security features for sensitive operations

## Advanced Configuration

### Custom Agents

You can create custom agent implementations:

```python
class MyCustomAgent:
    def __init__(self, config, ctm_interface, reward_system):
        self.config = config
        # ...
        
    async def run_cycle(self, domain, difficulty, constraints):
        # Custom implementation
        return result

# In config:
agents:
  custom:
    type: "custom"
    agent_class: MyCustomAgent
    # Custom parameters
```

### Performance Optimization

Configure performance settings:

```yaml
performance:
  cache_embeddings: true
  cache_duration: 3600  # 1 hour
  parallel_execution: true
  max_workers: 4
```

## Conclusion

The Claude Code Agentic DeepSeek Brain Framework creates a flexible, cost-effective system that combines the strengths of multiple AI models. By leveraging DeepSeek's price advantage while maintaining access to Claude's capabilities, you can optimize for both cost and performance.

The framework's integration with CTM-AbsoluteZero and DFZ provides a comprehensive solution for agentic AI that can continuously learn, adapt, and improve across different domains.

For detailed API documentation, see the source code and inline documentation in the framework modules.