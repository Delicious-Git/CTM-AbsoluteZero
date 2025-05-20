"""
Command-line interface for CTM-AbsoluteZero.
"""
import argparse
import logging
import os
import sys
import json
import yaml
import asyncio
import time
from typing import Dict, List, Any, Optional, Union

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.ctm_az_agent import AbsoluteZeroAgent
from src.ctm.interface import RealCTMInterface
from src.rewards.composite import CompositeRewardSystem
from src.rewards.novelty import SemanticNoveltyTracker
from src.rewards.progress import SkillPyramid
from src.transfer.adapter import NeuralTransferAdapter
from src.transfer.phase import PhaseController
from src.utils.logging import CTMLogger, configure_logging
from src.utils.config import ConfigManager, load_config, merge_configs
from src.integration.dfz import DFZAdapter, CTMAbsoluteZeroPlugin


def setup_logging(args: argparse.Namespace) -> logging.Logger:
    """
    Set up logging based on command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Logger instance
    """
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_file = args.log_file
    
    configure_logging(
        log_level=log_level,
        log_file=log_file,
        console_output=not args.quiet
    )
    
    logger = logging.getLogger("ctm-az.cli")
    logger.info(f"Logging initialized at level {args.log_level.upper()}")
    
    return logger


def load_configuration(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Load configuration from file and command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Merged configuration
    """
    config = {}
    
    # Load from file if specified
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading configuration from {args.config}: {e}")
            sys.exit(1)
    
    # Override with environment variables
    if args.env_prefix:
        for key, value in os.environ.items():
            if key.startswith(args.env_prefix):
                config_key = key[len(args.env_prefix):].lower()
                try:
                    # Try to parse as JSON for complex values
                    config[config_key] = json.loads(value)
                except json.JSONDecodeError:
                    # Fall back to string
                    config[config_key] = value
    
    # Override with command-line arguments
    if args.set:
        for kv in args.set:
            if '=' in kv:
                key, value = kv.split('=', 1)
                try:
                    # Try to parse as JSON for complex values
                    config[key] = json.loads(value)
                except json.JSONDecodeError:
                    # Fall back to string
                    config[key] = value
    
    return config


def setup_agent(config: Dict[str, Any]) -> AbsoluteZeroAgent:
    """
    Set up the AbsoluteZero agent from configuration.
    
    Args:
        config: Agent configuration
        
    Returns:
        Configured agent
    """
    logger = logging.getLogger("ctm-az.cli.setup")
    
    # Set up phase controller
    phase_controller = PhaseController(
        phase_duration=config.get("phase_duration", 600),
        initial_phase=config.get("initial_phase", "exploration")
    )
    
    # Set up novelty tracker
    reward_config = config.get("rewards", {})
    novelty_tracker = SemanticNoveltyTracker(
        embedding_dim=reward_config.get("embedding_dim", 768),
        novelty_threshold=reward_config.get("novelty_threshold", 0.2)
    )
    
    # Set up skill pyramid
    domains = config.get("domains", ["general"])
    skill_pyramid = SkillPyramid(
        domains=domains,
        levels=reward_config.get("skill_levels", 5)
    )
    
    # Set up reward system
    reward_system = CompositeRewardSystem(
        novelty_tracker=novelty_tracker,
        skill_pyramid=skill_pyramid,
        phase_controller=phase_controller,
        hyperparams=reward_config.get("hyperparams")
    )
    
    # Set up transfer adapter
    transfer_adapter = NeuralTransferAdapter(domains)
    
    # Set up CTM interface
    ctm_config = config.get("ctm", {})
    ctm_interface = RealCTMInterface(ctm_config)
    
    # Get model paths
    agent_config = config.get("agent", {})
    proposer_model_path = agent_config.get("proposer_model_path", "")
    solver_model_path = agent_config.get("solver_model_path", "")
    
    # Create agent
    agent = AbsoluteZeroAgent(
        proposer_model_path=proposer_model_path,
        proposer_tokenizer_path=agent_config.get("proposer_tokenizer_path", proposer_model_path),
        solver_model_path=solver_model_path,
        solver_tokenizer_path=agent_config.get("solver_tokenizer_path", solver_model_path),
        reward_system=reward_system,
        transfer_adapter=transfer_adapter,
        ctm_interface=ctm_interface,
        config=agent_config
    )
    
    logger.info("AbsoluteZero agent set up successfully")
    return agent


async def run_dfz_integration(
    args: argparse.Namespace,
    config: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """
    Run DFZ integration.
    
    Args:
        args: Command-line arguments
        config: Configuration
        logger: Logger instance
    """
    logger.info("Starting DFZ integration")
    
    # Create DFZ adapter
    dfz_adapter = DFZAdapter(
        dfz_path=args.dfz_path or config.get("dfz_path"),
        config=config
    )
    
    # Initialize adapter
    success = await dfz_adapter.initialize()
    if not success:
        logger.error("Failed to initialize DFZ adapter")
        return
    
    logger.info("DFZ adapter initialized successfully")
    
    # Run in interactive mode if requested
    if args.interactive:
        logger.info("Starting interactive mode")
        
        print("CTM-AbsoluteZero DFZ Integration Shell")
        print("Type 'exit' or 'quit' to exit")
        print("Type 'help' for available commands")
        
        while True:
            try:
                command = input("> ")
                command = command.strip()
                
                if command.lower() in ["exit", "quit"]:
                    break
                elif command.lower() == "help":
                    print("Available commands:")
                    print("  task <domain> - Generate tasks for a domain")
                    print("  msg <message> - Send a message to DFZ")
                    print("  execute <task_id> - Execute a task")
                    print("  history - Show conversation history")
                    print("  stats - Show performance statistics")
                    print("  exit/quit - Exit the shell")
                    print("  help - Show this help message")
                elif command.lower().startswith("task "):
                    domain = command[5:].strip() or "general"
                    print(f"Generating tasks for domain: {domain}")
                    tasks = await dfz_adapter.generate_task(domain)
                    print(f"Generated {len(tasks)} tasks:")
                    for task in tasks:
                        print(f"  {task.get('id', 'unknown')}: {task.get('description', '')}")
                elif command.lower().startswith("msg "):
                    message = command[4:].strip()
                    print(f"Sending message: {message}")
                    response = await dfz_adapter.send_message(message)
                    print(f"Response: {response.get('text', 'No response')}")
                elif command.lower().startswith("execute "):
                    task_id = command[8:].strip()
                    task = dfz_adapter.get_task(task_id)
                    if task:
                        print(f"Executing task: {task.get('description', '')}")
                        result = await dfz_adapter.execute_task(task)
                        print(f"Result: {result}")
                    else:
                        print(f"Task not found: {task_id}")
                elif command.lower() == "history":
                    history = dfz_adapter.get_conversation_history()
                    print("Conversation history:")
                    for entry in history:
                        print(f"  {entry['role']}: {entry['content']}")
                elif command.lower() == "stats":
                    metrics = dfz_adapter.plugin.get_performance_metrics()
                    print("Performance metrics:")
                    for key, value in metrics.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"Unknown command: {command}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("Exiting interactive mode")
    else:
        # Run in non-interactive mode (service mode)
        logger.info("Running in service mode")
        
        # Keep the process running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Service interrupted")


def run_train(
    args: argparse.Namespace,
    config: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """
    Run training mode.
    
    Args:
        args: Command-line arguments
        config: Configuration
        logger: Logger instance
    """
    logger.info("Starting training mode")
    
    # Setup agent
    agent = setup_agent(config)
    
    # Get parameters from arguments
    domain = args.domain or config.get("train_domain", "general")
    max_iterations = args.iterations or config.get("train_iterations", 100)
    eval_interval = args.eval_interval or config.get("eval_interval", 10)
    
    logger.info(f"Training on domain: {domain}")
    logger.info(f"Max iterations: {max_iterations}")
    logger.info(f"Evaluation interval: {eval_interval}")
    
    # Run training
    try:
        agent.train(
            domain=domain,
            max_iterations=max_iterations,
            eval_interval=eval_interval
        )
        
        # Save trained agent
        if args.save_path:
            logger.info(f"Saving trained agent to {args.save_path}")
            agent.save_state(args.save_path)
    except KeyboardInterrupt:
        logger.info("Training interrupted")
        
        # Save if interrupted
        if args.save_path:
            logger.info(f"Saving intermediate agent state to {args.save_path}")
            agent.save_state(args.save_path)
    except Exception as e:
        logger.error(f"Training failed: {e}")


def run_evaluate(
    args: argparse.Namespace,
    config: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """
    Run evaluation mode.
    
    Args:
        args: Command-line arguments
        config: Configuration
        logger: Logger instance
    """
    logger.info("Starting evaluation mode")
    
    # Setup agent
    agent = setup_agent(config)
    
    # Get parameters from arguments
    domain = args.domain or config.get("eval_domain", "general")
    num_tasks = args.num_tasks or config.get("eval_tasks", 10)
    
    logger.info(f"Evaluating on domain: {domain}")
    logger.info(f"Number of tasks: {num_tasks}")
    
    # Run evaluation
    try:
        results = agent.evaluate(domain=domain, num_tasks=num_tasks)
        
        # Display results
        logger.info("Evaluation results:")
        
        print("\nEvaluation Results:")
        print(f"Domain: {domain}")
        print(f"Tasks attempted: {results['total_tasks']}")
        print(f"Success rate: {results['success_rate'] * 100:.2f}%")
        print(f"Average reward: {results['avg_reward']:.4f}")
        print(f"Average solution time: {results['avg_time']:.2f}s")
        
        # Save results if requested
        if args.output:
            logger.info(f"Saving evaluation results to {args.output}")
            
            try:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save evaluation results: {e}")
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")


def run_solve(
    args: argparse.Namespace,
    config: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """
    Run solve mode to solve a single task.
    
    Args:
        args: Command-line arguments
        config: Configuration
        logger: Logger instance
    """
    logger.info("Starting solve mode")
    
    # Check for task
    task_description = args.task
    if not task_description:
        logger.error("No task specified")
        print("Error: No task specified. Use --task to specify a task.")
        return
    
    # Setup agent
    agent = setup_agent(config)
    
    # Create task configuration
    domain = args.domain or config.get("default_domain", "general")
    task_config = {
        "domain": domain,
        "description": task_description,
        "parameters": {}
    }
    
    # Add parameters from arguments
    if args.params:
        for param in args.params:
            if '=' in param:
                key, value = param.split('=', 1)
                try:
                    # Try to parse as JSON
                    task_config["parameters"][key] = json.loads(value)
                except json.JSONDecodeError:
                    # Fall back to string
                    task_config["parameters"][key] = value
    
    logger.info(f"Solving task in domain: {domain}")
    logger.info(f"Task: {task_description}")
    
    # Solve task
    try:
        print(f"Solving task: {task_description}")
        start_time = time.time()
        
        result = agent.solve_task(task_config)
        
        duration = time.time() - start_time
        
        print(f"\nTask solved in {duration:.2f}s")
        print(f"Result: {result}")
        
        # Save results if requested
        if args.output:
            logger.info(f"Saving solution to {args.output}")
            
            output_data = {
                "task": task_config,
                "result": result,
                "duration": duration
            }
            
            try:
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save solution: {e}")
    except KeyboardInterrupt:
        logger.info("Task solving interrupted")
    except Exception as e:
        logger.error(f"Task solving failed: {e}")
        print(f"Failed to solve task: {e}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="CTM-AbsoluteZero: Continuous Thought Machine with Absolute Zero Reasoner"
    )
    
    # Common arguments
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file",
        type=str
    )
    parser.add_argument(
        "--log-level",
        help="Logging level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info"
    )
    parser.add_argument(
        "--log-file",
        help="Path to log file",
        type=str
    )
    parser.add_argument(
        "--quiet", "-q",
        help="Suppress console output",
        action="store_true"
    )
    parser.add_argument(
        "--set",
        help="Set configuration values (format: key=value)",
        action="append",
        default=[]
    )
    parser.add_argument(
        "--env-prefix",
        help="Environment variable prefix for configuration",
        default="CTM_AZ_"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the agent")
    train_parser.add_argument(
        "--domain",
        help="Domain to train on",
        type=str
    )
    train_parser.add_argument(
        "--iterations", "-i",
        help="Maximum number of training iterations",
        type=int
    )
    train_parser.add_argument(
        "--eval-interval",
        help="Evaluation interval (iterations)",
        type=int
    )
    train_parser.add_argument(
        "--save-path",
        help="Path to save trained agent",
        type=str
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the agent")
    eval_parser.add_argument(
        "--domain",
        help="Domain to evaluate on",
        type=str
    )
    eval_parser.add_argument(
        "--num-tasks", "-n",
        help="Number of tasks to evaluate",
        type=int
    )
    eval_parser.add_argument(
        "--output", "-o",
        help="Path to save evaluation results",
        type=str
    )
    
    # Solve command
    solve_parser = subparsers.add_parser("solve", help="Solve a task")
    solve_parser.add_argument(
        "--task", "-t",
        help="Task description",
        type=str
    )
    solve_parser.add_argument(
        "--domain",
        help="Task domain",
        type=str
    )
    solve_parser.add_argument(
        "--params", "-p",
        help="Task parameters (format: key=value)",
        action="append",
        default=[]
    )
    solve_parser.add_argument(
        "--output", "-o",
        help="Path to save solution",
        type=str
    )
    
    # DFZ integration command
    dfz_parser = subparsers.add_parser("dfz", help="Run DFZ integration")
    dfz_parser.add_argument(
        "--dfz-path",
        help="Path to DFZ installation",
        type=str
    )
    dfz_parser.add_argument(
        "--interactive", "-i",
        help="Run in interactive mode",
        action="store_true"
    )
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate tasks")
    generate_parser.add_argument(
        "--domain",
        help="Domain to generate tasks for",
        type=str
    )
    generate_parser.add_argument(
        "--count", "-n",
        help="Number of tasks to generate",
        type=int,
        default=3
    )
    generate_parser.add_argument(
        "--output", "-o",
        help="Path to save generated tasks",
        type=str
    )
    
    return parser.parse_args()


def run_generate(
    args: argparse.Namespace,
    config: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """
    Run generate mode to generate tasks.
    
    Args:
        args: Command-line arguments
        config: Configuration
        logger: Logger instance
    """
    logger.info("Starting generate mode")
    
    # Setup agent
    agent = setup_agent(config)
    
    # Get parameters from arguments
    domain = args.domain or config.get("default_domain", "general")
    count = args.count or 3
    
    logger.info(f"Generating tasks for domain: {domain}")
    logger.info(f"Count: {count}")
    
    # Generate tasks
    try:
        print(f"Generating {count} tasks for domain: {domain}")
        
        tasks = agent.generate_tasks(domain=domain, count=count)
        
        print(f"\nGenerated {len(tasks)} tasks:")
        for i, task in enumerate(tasks, 1):
            print(f"\n{i}. {task.get('description', '')}")
            
            parameters = task.get("parameters", {})
            if parameters:
                print("   Parameters:")
                for key, value in parameters.items():
                    print(f"     {key}: {value}")
        
        # Save results if requested
        if args.output:
            logger.info(f"Saving generated tasks to {args.output}")
            
            try:
                with open(args.output, 'w') as f:
                    json.dump(tasks, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save generated tasks: {e}")
    except KeyboardInterrupt:
        logger.info("Task generation interrupted")
    except Exception as e:
        logger.error(f"Task generation failed: {e}")
        print(f"Failed to generate tasks: {e}")


def main() -> None:
    """Main entry point for the CLI."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args)
    
    # Load configuration
    config = load_configuration(args)
    
    # Handle commands
    if args.command == "train":
        run_train(args, config, logger)
    elif args.command == "evaluate":
        run_evaluate(args, config, logger)
    elif args.command == "solve":
        run_solve(args, config, logger)
    elif args.command == "dfz":
        asyncio.run(run_dfz_integration(args, config, logger))
    elif args.command == "generate":
        run_generate(args, config, logger)
    else:
        print("No command specified. Use --help for help.")


if __name__ == "__main__":
    main()