#!/usr/bin/env python3
"""
Wake script for CTM-AbsoluteZero.

This script wakes up the system from standby mode, performs necessary
initialization, and starts required services.
"""
import os
import sys
import argparse
import logging
import json
import time
import subprocess
from typing import Dict, List, Any, Optional

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.utils.logging import get_logger, configure_logging
from src.utils.config import ConfigManager

# Setup logger
logger = get_logger("ctm-az.wake")

class SystemWake:
    """Handles waking up the CTM-AbsoluteZero system."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        state_dir: str = "./state",
        triggers: Optional[List[str]] = None
    ):
        """
        Initialize the system wake.
        
        Args:
            config_path: Path to configuration file
            state_dir: Directory for state information
            triggers: List of triggered wake IDs
        """
        self.config_path = config_path
        self.state_dir = state_dir
        self.triggers = triggers or []
        
        # Create state directory
        os.makedirs(state_dir, exist_ok=True)
        
        # Load configuration
        self.config_manager = ConfigManager(config_path) if config_path else ConfigManager()
        self.config = self.config_manager.to_dict()
        
        # Load wake state
        self.state = self._load_state()
        
        logger.info(f"System wake initialized with triggers: {', '.join(self.triggers)}")
    
    def _load_state(self) -> Dict[str, Any]:
        """
        Load wake state from file.
        
        Returns:
            State dictionary
        """
        state_path = os.path.join(self.state_dir, "system_state.json")
        
        # Default state
        default_state = {
            "last_wake": 0,
            "last_standby": 0,
            "status": "standby",
            "components": {}
        }
        
        # Load from file if it exists
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                    
                return state
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        
        return default_state
    
    def _save_state(self) -> None:
        """Save system state to file."""
        state_path = os.path.join(self.state_dir, "system_state.json")
        
        try:
            with open(state_path, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def wake(self) -> bool:
        """
        Wake up the system.
        
        Returns:
            True if the system was woken up successfully, False otherwise
        """
        logger.info("Waking up system")
        
        # Check if we're already awake
        if self.state.get("status") == "awake":
            logger.info("System is already awake")
            return True
        
        # Update state
        self.state["last_wake"] = int(time.time())
        self.state["status"] = "waking"
        self.state["wake_triggers"] = self.triggers
        
        # Save state
        self._save_state()
        
        # Run initialization steps
        try:
            # Run tests first to verify system integrity
            success = self._run_tests()
            
            if not success:
                logger.error("System tests failed, aborting wake")
                
                # Update state
                self.state["status"] = "error"
                self._save_state()
                
                return False
            
            # Start required services
            success = self._start_services()
            
            if not success:
                logger.error("Failed to start services, aborting wake")
                
                # Update state
                self.state["status"] = "error"
                self._save_state()
                
                return False
            
            # Update state
            self.state["status"] = "awake"
            self._save_state()
            
            logger.info("System wake completed successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Wake failed: {e}")
            
            # Update state
            self.state["status"] = "error"
            self._save_state()
            
            return False
    
    def _run_tests(self) -> bool:
        """
        Run system tests.
        
        Returns:
            True if tests pass, False otherwise
        """
        logger.info("Running system tests")
        
        # Run test suite
        test_script = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "tests", "integration", "automated_test_suite.py"
        )
        
        if not os.path.exists(test_script):
            logger.error(f"Test script not found: {test_script}")
            return False
        
        # Create test output directory
        test_dir = os.path.join(self.state_dir, "tests")
        os.makedirs(test_dir, exist_ok=True)
        
        # Skip slow tests during wake
        cmd = [
            sys.executable,
            test_script,
            "--output-dir", test_dir,
            "--skip-slow"
        ]
        
        if self.config_path:
            cmd.extend(["--config", self.config_path])
        
        try:
            logger.info(f"Running test command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Tests failed with return code {result.returncode}")
                logger.error(f"Test output: {result.stdout}")
                logger.error(f"Test errors: {result.stderr}")
                
                return False
            
            # Check for latest.txt in test output
            latest_path = os.path.join(test_dir, "latest.txt")
            
            if os.path.exists(latest_path):
                with open(latest_path, 'r') as f:
                    latest = f.read()
                    logger.info(f"Test results: {latest}")
                    
                    # Parse success rate
                    for line in latest.splitlines():
                        if line.startswith("Success rate:"):
                            rate_str = line.split(":")[1].strip()
                            rate = float(rate_str.rstrip("%")) / 100
                            
                            if rate < 0.7:
                                logger.error(f"Test success rate too low: {rate_str}")
                                return False
            
            logger.info("System tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return False
    
    def _start_services(self) -> bool:
        """
        Start required services.
        
        Returns:
            True if services start successfully, False otherwise
        """
        logger.info("Starting services")
        
        # Get services to start based on triggers
        services = self._get_services_for_triggers()
        
        for service_name, service in services.items():
            logger.info(f"Starting service: {service_name}")
            
            cmd = service.get("cmd")
            if not cmd:
                logger.warning(f"No command defined for service {service_name}")
                continue
            
            # Create logs directory
            logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            # Run service
            try:
                service_log = os.path.join(logs_dir, f"{service_name}.log")
                
                with open(service_log, 'a') as log_file:
                    # Start service as subprocess and detach
                    if service.get("detach", True):
                        subprocess.Popen(
                            cmd,
                            stdout=log_file,
                            stderr=log_file,
                            start_new_session=True
                        )
                    else:
                        # Run and wait for completion
                        result = subprocess.run(
                            cmd,
                            stdout=log_file,
                            stderr=log_file
                        )
                        
                        if result.returncode != 0:
                            logger.error(f"Service {service_name} failed with return code {result.returncode}")
                            return False
                    
                # Update service state
                if "components" not in self.state:
                    self.state["components"] = {}
                
                if "services" not in self.state["components"]:
                    self.state["components"]["services"] = {}
                
                self.state["components"]["services"][service_name] = {
                    "status": "running",
                    "start_time": int(time.time()),
                    "log": service_log
                }
                
                logger.info(f"Service {service_name} started")
                
            except Exception as e:
                logger.error(f"Failed to start service {service_name}: {e}")
                return False
        
        return True
    
    def _get_services_for_triggers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get services to start based on triggers.
        
        Returns:
            Dictionary of services to start
        """
        # Default services (always start)
        services = {
            "daemon": {
                "cmd": [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "wake_triggers.py"),
                    "--daemon"
                ],
                "detach": True,
                "priority": "high",
                "description": "Wake trigger daemon"
            }
        }
        
        # Add additional services based on triggers
        if "data" in self.triggers:
            # Start data processing service
            services["data_processing"] = {
                "cmd": [
                    sys.executable,
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "cli.py"),
                    "process-data"
                ],
                "detach": False,
                "priority": "high",
                "description": "Data processing service"
            }
        
        if "dependencies" in self.triggers:
            # Update dependencies
            services["dependency_update"] = {
                "cmd": [
                    "pip", "install", "-r",
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "requirements.txt")
                ],
                "detach": False,
                "priority": "medium",
                "description": "Dependency update service"
            }
        
        # Load additional services from configuration
        config_services = self.config.get("services", {})
        
        for service_name, service in config_services.items():
            if not service.get("enabled", True):
                continue
                
            # Check if service should be started for these triggers
            service_triggers = service.get("triggers", [])
            
            if not service_triggers or any(trigger in self.triggers for trigger in service_triggers):
                services[service_name] = service
        
        return services

def main():
    """Run system wake from command line."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="CTM-AbsoluteZero Wake")
    
    # Add arguments
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file",
        type=str
    )
    parser.add_argument(
        "--state-dir", "-s",
        help="Directory for state information",
        type=str,
        default="./state"
    )
    parser.add_argument(
        "--triggers", "-t",
        help="Comma-separated list of trigger IDs",
        type=str
    )
    parser.add_argument(
        "--log-level",
        help="Logging level",
        choices=["debug", "info", "warning", "error"],
        default="info"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging
    log_level = getattr(logging, args.log_level.upper())
    configure_logging(log_level=log_level)
    
    # Parse triggers
    triggers = args.triggers.split(",") if args.triggers else []
    
    # Create system wake
    system_wake = SystemWake(
        config_path=args.config,
        state_dir=args.state_dir,
        triggers=triggers
    )
    
    # Wake system
    success = system_wake.wake()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()