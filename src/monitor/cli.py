"""
Command-line interface for CTM-AbsoluteZero monitor.
"""
import os
import sys
import argparse
import logging
from typing import Dict, Any

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import utilities
from src.utils.logging import configure_logging, get_logger
from src.utils.config import ConfigManager, load_config

# Initialize logger
logger = get_logger("ctm-az.monitor.cli")

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
    
    logger = get_logger("ctm-az.monitor.cli")
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
        config_manager = ConfigManager(args.config)
        config = config_manager.to_dict()
    
    # Override with environment variables
    for key, value in os.environ.items():
        if key.startswith("CTM_AZ_"):
            config_key = key[7:].lower()
            config[config_key] = value
    
    return config

def run_dashboard(args: argparse.Namespace, config: Dict[str, Any]):
    """
    Run the monitoring dashboard.
    
    Args:
        args: Command-line arguments
        config: Configuration dictionary
    """
    # Import the dashboard app
    from src.monitor.app import app as dashboard_app
    
    # Set environment variables from config
    if "core_api_url" in config:
        os.environ["CORE_API_URL"] = config["core_api_url"]
    
    if "log_level" in config:
        os.environ["LOG_LEVEL"] = config["log_level"]
    
    if "dash_debug" in config:
        os.environ["DASH_DEBUG"] = str(config["dash_debug"]).lower()
    
    # Determine port
    port = args.port or config.get("port", 8080)
    
    # Run the dashboard
    logger.info(f"Starting monitor dashboard on port {port}")
    dashboard_app.run_server(
        debug=args.debug or config.get("dash_debug", False),
        host="0.0.0.0",
        port=port
    )

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="CTM-AbsoluteZero Monitor: Real-time monitoring dashboard"
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
    
    # Dashboard-specific arguments
    parser.add_argument(
        "--port", "-p",
        help="Port to run the dashboard on",
        type=int
    )
    parser.add_argument(
        "--debug", "-d",
        help="Run in debug mode",
        action="store_true"
    )
    parser.add_argument(
        "--api-url",
        help="URL of the core API",
        type=str
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the monitor CLI."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args)
    
    # Load configuration
    config = load_configuration(args)
    
    # Set API URL from args if provided
    if args.api_url:
        config["core_api_url"] = args.api_url
        os.environ["CORE_API_URL"] = args.api_url
    
    # Run dashboard
    try:
        run_dashboard(args, config)
    except KeyboardInterrupt:
        logger.info("Monitor dashboard stopped by user")
    except Exception as e:
        logger.error(f"Error running monitor dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()