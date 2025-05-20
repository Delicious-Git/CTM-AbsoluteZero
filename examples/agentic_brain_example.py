"""
Example of using the Agentic Brain Framework with CTM-AbsoluteZero and DFZ.
"""
import os
import sys
import asyncio
import json
import argparse
import logging
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.agentic.framework import BrainFramework
from src.ctm.interface import RealCTMInterface
from src.rewards.composite import CompositeRewardSystem
from src.integration.dfz import DFZAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("agentic_brain_example")

async def run_example(args):
    """Run the example."""
    logger.info("Starting Agentic Brain Framework example")
    
    # Set up CTM components
    logger.info("Setting up CTM components")
    
    # Create CTM interface
    ctm_interface = RealCTMInterface({
        "components": ["maze_solver", "image_classifier", "quantum_sim", "sorter"],
        "metrics_enabled": True
    })
    
    # Set up reward system
    reward_system = CompositeRewardSystem(
        novelty_tracker=None,  # Will be created internally
        skill_pyramid=None,    # Will be created internally
        phase_controller=None  # Will be created internally
    )
    
    # Set up DFZ adapter if needed
    dfz_adapter = None
    if args.dfz_path and os.path.exists(args.dfz_path):
        logger.info(f"Setting up DFZ adapter with path: {args.dfz_path}")
        dfz_adapter = DFZAdapter(
            dfz_path=args.dfz_path,
            config={}
        )
        await dfz_adapter.initialize()
    else:
        logger.warning("DFZ path not provided or does not exist, DFZ integration will be disabled")
    
    # Create Brain Framework
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              "configs", "agentic_brain.yaml")
    
    logger.info(f"Creating Brain Framework with config: {config_path}")
    framework = BrainFramework(
        config_path=config_path,
        ctm_interface=ctm_interface,
        reward_system=reward_system,
        dfz_adapter=dfz_adapter
    )
    
    # Run different scenarios based on the command
    if args.command == "cycle":
        await run_cycle(framework, args)
    elif args.command == "compare":
        await run_comparison(framework, args)
    elif args.command == "dfz":
        await run_dfz_integration(framework, args)
    else:
        logger.error(f"Unknown command: {args.command}")

async def run_cycle(framework, args):
    """Run a single agentic cycle."""
    logger.info(f"Running agentic cycle with domain: {args.domain}, difficulty: {args.difficulty}")
    
    result = await framework.run_cycle(
        domain=args.domain,
        difficulty=args.difficulty,
        constraints=None,
        agent_name=args.agent
    )
    
    if "error" in result:
        logger.error(f"Cycle failed: {result['error']}")
        return
    
    logger.info(f"Cycle completed with reward: {result.get('reward', 0)}")
    
    # Print detailed results
    if args.verbose:
        print(json.dumps(result, indent=2))
    else:
        print("\nTask:")
        print(f"Domain: {result['task']['domain']}")
        print(f"Description: {result['task']['description']}")
        
        print("\nSolution Analysis:")
        analysis = result["analysis"]
        print(f"Effectiveness: {analysis.get('effectiveness', 0):.2f}")
        print(f"Efficiency: {analysis.get('efficiency', 0):.2f}")
        print(f"Correctness: {analysis.get('correctness', 0):.2f}")
        print(f"Overall Rating: {analysis.get('overall_rating', 0):.2f}")
        
        print("\nReward:", result.get('reward', 0))

async def run_comparison(framework, args):
    """Run a comparison between Claude and DeepSeek."""
    logger.info("Running agent comparison")
    
    print("\n=== Agent Comparison ===")
    print(f"Domain: {args.domain}, Difficulty: {args.difficulty}")
    print("=======================")
    
    # Run with Claude
    print("\n>> Running with Claude agent")
    claude_start = asyncio.get_event_loop().time()
    claude_result = await framework.run_cycle(
        domain=args.domain,
        difficulty=args.difficulty,
        constraints=None,
        agent_name="claude"
    )
    claude_time = asyncio.get_event_loop().time() - claude_start
    
    # Run with DeepSeek
    print("\n>> Running with DeepSeek agent")
    deepseek_start = asyncio.get_event_loop().time()
    deepseek_result = await framework.run_cycle(
        domain=args.domain,
        difficulty=args.difficulty,
        constraints=None,
        agent_name="deepseek"
    )
    deepseek_time = asyncio.get_event_loop().time() - deepseek_start
    
    # Print comparison
    print("\n=== Comparison Results ===")
    
    print("\nClaude Agent:")
    print(f"Time: {claude_time:.2f} seconds")
    if "error" in claude_result:
        print(f"Error: {claude_result['error']}")
    else:
        print(f"Task: {claude_result['task']['description']}")
        analysis = claude_result["analysis"]
        print(f"Overall Rating: {analysis.get('overall_rating', 0):.2f}")
        print(f"Reward: {claude_result.get('reward', 0)}")
    
    print("\nDeepSeek Agent:")
    print(f"Time: {deepseek_time:.2f} seconds")
    if "error" in deepseek_result:
        print(f"Error: {deepseek_result['error']}")
    else:
        print(f"Task: {deepseek_result['task']['description']}")
        analysis = deepseek_result["analysis"]
        print(f"Overall Rating: {analysis.get('overall_rating', 0):.2f}")
        print(f"Reward: {deepseek_result.get('reward', 0)}")
    
    # Cost comparison (approximate)
    claude_tokens = 4000  # Estimated
    deepseek_tokens = 4000  # Estimated
    claude_cost = claude_tokens * 0.008 / 1000  # $0.008 per 1K tokens
    deepseek_cost = deepseek_tokens * 0.0001 / 1000  # $0.0001 per 1K tokens
    
    print("\nCost Comparison (estimated):")
    print(f"Claude: ${claude_cost:.6f} (approx. {claude_tokens} tokens)")
    print(f"DeepSeek: ${deepseek_cost:.6f} (approx. {deepseek_tokens} tokens)")
    print(f"Cost Ratio: Claude is {claude_cost/deepseek_cost:.1f}x more expensive")
    
    # Performance comparison
    print("\nPerformance Comparison:")
    if "error" not in claude_result and "error" not in deepseek_result:
        claude_rating = claude_result["analysis"].get("overall_rating", 0)
        deepseek_rating = deepseek_result["analysis"].get("overall_rating", 0)
        
        if claude_rating > deepseek_rating:
            print(f"Claude rated higher by {claude_rating - deepseek_rating:.2f} points")
        elif deepseek_rating > claude_rating:
            print(f"DeepSeek rated higher by {deepseek_rating - claude_rating:.2f} points")
        else:
            print("Both agents rated equally")
            
        # Performance per dollar
        claude_perf_per_dollar = claude_rating / claude_cost if claude_cost > 0 else 0
        deepseek_perf_per_dollar = deepseek_rating / deepseek_cost if deepseek_cost > 0 else 0
        
        print("\nPerformance per Dollar:")
        print(f"Claude: {claude_perf_per_dollar:.2f}")
        print(f"DeepSeek: {deepseek_perf_per_dollar:.2f}")
        
        if deepseek_perf_per_dollar > claude_perf_per_dollar:
            print(f"DeepSeek has {deepseek_perf_per_dollar/claude_perf_per_dollar:.1f}x better performance per dollar")
        else:
            print(f"Claude has {claude_perf_per_dollar/deepseek_perf_per_dollar:.1f}x better performance per dollar")

async def run_dfz_integration(framework, args):
    """Run DFZ integration test."""
    if not framework.dfz_adapter:
        logger.error("DFZ integration is not available")
        print("DFZ integration is not available. Please provide a valid DFZ path.")
        return
    
    logger.info("Running DFZ integration test")
    
    # Step 1: Send a message to DFZ
    print("\n=== DFZ Integration Test ===")
    
    print("\n>> Sending message to DFZ")
    message = "I'd like to create a new task related to quantum computing"
    response = await framework.send_to_dfz(message)
    
    if "error" in response:
        logger.error(f"Failed to send message to DFZ: {response['error']}")
        print(f"Error: {response['error']}")
        return
    
    print(f"DFZ Response: {response.get('text', 'No response')}")
    
    # Step 2: Generate tasks from DFZ
    print("\n>> Generating tasks from DFZ")
    tasks = await framework.generate_dfz_task(domain=args.domain)
    
    if not tasks or "error" in tasks[0]:
        error = tasks[0].get("error", "Unknown error") if tasks else "No tasks generated"
        logger.error(f"Failed to generate tasks from DFZ: {error}")
        print(f"Error: {error}")
        return
    
    print(f"Generated {len(tasks)} tasks from DFZ")
    for i, task in enumerate(tasks):
        print(f"\nTask {i+1}:")
        print(f"ID: {task.get('id', 'unknown')}")
        print(f"Description: {task.get('description', 'No description')}")
    
    # Step 3: Execute the first task
    if tasks:
        print("\n>> Executing first task from DFZ")
        task = tasks[0]
        result = await framework.execute_dfz_task(task)
        
        if "error" in result:
            logger.error(f"Failed to execute task: {result['error']}")
            print(f"Error: {result['error']}")
        else:
            print("Task executed successfully")
            print(f"Status: {result.get('status', 'unknown')}")
            print(f"Result: {json.dumps(result.get('result', {}), indent=2)}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Agentic Brain Framework Example")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Cycle command
    cycle_parser = subparsers.add_parser("cycle", help="Run a single agentic cycle")
    cycle_parser.add_argument("--domain", type=str, default="quantum", 
                             help="Task domain (quantum, maze, sorting)")
    cycle_parser.add_argument("--difficulty", type=str, default="medium",
                             help="Task difficulty (easy, medium, hard)")
    cycle_parser.add_argument("--agent", type=str, default=None,
                             help="Agent to use (claude, deepseek)")
    cycle_parser.add_argument("--verbose", "-v", action="store_true",
                             help="Print detailed results")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare Claude and DeepSeek")
    compare_parser.add_argument("--domain", type=str, default="quantum",
                               help="Task domain (quantum, maze, sorting)")
    compare_parser.add_argument("--difficulty", type=str, default="medium",
                               help="Task difficulty (easy, medium, hard)")
    
    # DFZ integration command
    dfz_parser = subparsers.add_parser("dfz", help="Test DFZ integration")
    dfz_parser.add_argument("--domain", type=str, default="quantum",
                           help="Task domain (quantum, maze, sorting)")
    
    # Common arguments
    parser.add_argument("--dfz-path", type=str, default=None,
                       help="Path to DFZ installation")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_example(args))