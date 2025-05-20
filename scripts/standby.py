#!/usr/bin/env python3
"""
Standby script for CTM-AbsoluteZero.

This script puts the system into standby mode, cleaning up resources and
shutting down services.
"""
import os
import sys
import argparse
import logging
import json
import time
import subprocess
import signal
from typing import Dict, List, Any, Optional

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.utils.logging import get_logger, configure_logging
from src.utils.config import ConfigManager

# Setup logger
logger = get_logger("ctm-az.standby")

class SystemStandby:
    """Handles putting the CTM-AbsoluteZero system into standby mode."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        state_dir: str = "./state",
        generate_report: bool = True
    ):
        """
        Initialize the system standby.
        
        Args:
            config_path: Path to configuration file
            state_dir: Directory for state information
            generate_report: Whether to generate a standby report
        """
        self.config_path = config_path
        self.state_dir = state_dir
        self.generate_report = generate_report
        
        # Create state directory
        os.makedirs(state_dir, exist_ok=True)
        
        # Load configuration
        self.config_manager = ConfigManager(config_path) if config_path else ConfigManager()
        self.config = self.config_manager.to_dict()
        
        # Load system state
        self.state = self._load_state()
        
        logger.info("System standby initialized")
    
    def _load_state(self) -> Dict[str, Any]:
        """
        Load system state from file.
        
        Returns:
            State dictionary
        """
        state_path = os.path.join(self.state_dir, "system_state.json")
        
        # Default state
        default_state = {
            "last_wake": 0,
            "last_standby": 0,
            "status": "unknown",
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
    
    def standby(self) -> bool:
        """
        Put the system into standby mode.
        
        Returns:
            True if the system was put into standby successfully, False otherwise
        """
        logger.info("Putting system into standby mode")
        
        # Check if we're already in standby
        if self.state.get("status") == "standby":
            logger.info("System is already in standby mode")
            return True
        
        # Update state
        self.state["last_standby"] = int(time.time())
        self.state["status"] = "entering_standby"
        
        # Save state
        self._save_state()
        
        # Run standby steps
        try:
            # Stop running services
            success = self._stop_services()
            
            if not success:
                logger.error("Failed to stop all services")
                
                # Continue with standby even if some services failed to stop
            
            # Run cleanup
            self._cleanup()
            
            # Generate standby report
            if self.generate_report:
                self._generate_report()
            
            # Setup wake triggers
            self._setup_wake_triggers()
            
            # Update state
            self.state["status"] = "standby"
            self._save_state()
            
            logger.info("System is now in standby mode")
            
            return True
            
        except Exception as e:
            logger.error(f"Standby failed: {e}")
            
            # Update state
            self.state["status"] = "error"
            self._save_state()
            
            return False
    
    def _stop_services(self) -> bool:
        """
        Stop running services.
        
        Returns:
            True if all services stopped successfully, False otherwise
        """
        logger.info("Stopping services")
        
        # Get running services
        services = self.state.get("components", {}).get("services", {})
        
        all_stopped = True
        
        for service_name, service in services.items():
            if service.get("status") != "running":
                continue
                
            logger.info(f"Stopping service: {service_name}")
            
            # Get service PID
            pid = service.get("pid")
            
            if pid:
                try:
                    # Send SIGTERM to process
                    os.kill(pid, signal.SIGTERM)
                    
                    # Wait for process to terminate
                    for _ in range(10):  # Wait up to 10 seconds
                        try:
                            # Check if process is still running
                            os.kill(pid, 0)
                            time.sleep(1)
                        except OSError:
                            # Process is no longer running
                            break
                    else:
                        # Process didn't terminate, send SIGKILL
                        logger.warning(f"Service {service_name} didn't terminate, sending SIGKILL")
                        os.kill(pid, signal.SIGKILL)
                    
                    logger.info(f"Service {service_name} stopped")
                    
                except OSError as e:
                    logger.warning(f"Failed to stop service {service_name}: {e}")
                    all_stopped = False
            else:
                # No PID, use service stop command if available
                stop_cmd = service.get("stop_cmd")
                
                if stop_cmd:
                    try:
                        subprocess.run(stop_cmd, check=True)
                        logger.info(f"Service {service_name} stopped")
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"Failed to stop service {service_name}: {e}")
                        all_stopped = False
                else:
                    logger.warning(f"No PID or stop command for service {service_name}")
                    all_stopped = False
            
            # Update service state
            service["status"] = "stopped"
            service["stop_time"] = int(time.time())
        
        return all_stopped
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up resources")
        
        # Clean up temporary files
        tmp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp")
        
        if os.path.exists(tmp_dir):
            try:
                for root, dirs, files in os.walk(tmp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            logger.debug(f"Removed temporary file: {file_path}")
                        except OSError as e:
                            logger.warning(f"Failed to remove temporary file {file_path}: {e}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {e}")
        
        # Clean up old logs
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        
        if os.path.exists(logs_dir):
            try:
                # Keep only the latest 10 log files for each service
                for root, dirs, files in os.walk(logs_dir):
                    # Group log files by service
                    log_groups = {}
                    
                    for file in files:
                        # Skip if not a log file
                        if not file.endswith(".log"):
                            continue
                            
                        # Get service name from file name
                        service_name = file.split(".")[0]
                        
                        if service_name not in log_groups:
                            log_groups[service_name] = []
                            
                        log_groups[service_name].append(file)
                    
                    # Keep only the latest 10 log files for each service
                    for service_name, logs in log_groups.items():
                        if len(logs) <= 10:
                            continue
                            
                        # Sort logs by modification time
                        logs.sort(key=lambda x: os.path.getmtime(os.path.join(root, x)), reverse=True)
                        
                        # Remove old logs
                        for log in logs[10:]:
                            log_path = os.path.join(root, log)
                            try:
                                os.remove(log_path)
                                logger.debug(f"Removed old log file: {log_path}")
                            except OSError as e:
                                logger.warning(f"Failed to remove old log file {log_path}: {e}")
            except Exception as e:
                logger.warning(f"Failed to clean up old logs: {e}")
    
    def _generate_report(self) -> None:
        """Generate standby report."""
        logger.info("Generating standby report")
        
        # Create report directory
        report_dir = os.path.join(self.state_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)
        
        # Create timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Collect report data
        report = {
            "timestamp": timestamp,
            "uptime": int(time.time()) - self.state.get("last_wake", int(time.time())),
            "status": self.state.get("status", "unknown"),
            "components": self.state.get("components", {}),
            "performance": self._collect_performance_data(),
            "system_info": self._collect_system_info()
        }
        
        # Save report
        report_path = os.path.join(report_dir, f"standby_report_{timestamp}.json")
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Standby report saved to {report_path}")
            
            # Create human-readable report
            text_report_path = os.path.join(report_dir, f"standby_report_{timestamp}.txt")
            
            with open(text_report_path, 'w') as f:
                f.write("=== CTM-AbsoluteZero Standby Report ===\n\n")
                f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Uptime: {report['uptime']} seconds\n")
                f.write(f"Status: {report['status']}\n\n")
                
                # Write performance data
                f.write("=== Performance ===\n\n")
                
                if "router" in report["performance"]:
                    router = report["performance"]["router"]
                    f.write(f"Tasks processed: {router.get('total_tasks', 0)}\n")
                    f.write(f"Success rate: {router.get('success_rate', 0) * 100:.1f}%\n")
                    f.write(f"Average task duration: {router.get('avg_duration', 0):.2f}s\n\n")
                
                if "benchmarks" in report["performance"]:
                    benchmarks = report["performance"]["benchmarks"]
                    f.write("Benchmarks:\n")
                    
                    for benchmark_name, benchmark in benchmarks.items():
                        f.write(f"  {benchmark_name}:\n")
                        
                        for metric_name, metric_value in benchmark.items():
                            f.write(f"    {metric_name}: {metric_value}\n")
                        
                        f.write("\n")
                
                # Write system info
                f.write("=== System Info ===\n\n")
                
                for key, value in report["system_info"].items():
                    f.write(f"{key}: {value}\n")
                
                f.write("\n=== End of Report ===\n")
            
            logger.info(f"Human-readable standby report saved to {text_report_path}")
            
        except Exception as e:
            logger.error(f"Failed to save standby report: {e}")
    
    def _collect_performance_data(self) -> Dict[str, Any]:
        """
        Collect performance data.
        
        Returns:
            Dictionary of performance data
        """
        performance = {}
        
        # Collect router stats
        try:
            router_stats_path = os.path.join(self.state_dir, "router_stats.json")
            
            if os.path.exists(router_stats_path):
                with open(router_stats_path, 'r') as f:
                    router_stats = json.load(f)
                    
                performance["router"] = router_stats
        except Exception as e:
            logger.warning(f"Failed to collect router stats: {e}")
        
        # Collect benchmark results
        try:
            benchmark_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark_results")
            
            if os.path.exists(benchmark_dir):
                benchmarks = {}
                
                # Find latest benchmark results
                for root, dirs, files in os.walk(benchmark_dir):
                    for file in files:
                        if file.startswith("benchmark_results_") and file.endswith(".json"):
                            benchmark_path = os.path.join(root, file)
                            
                            try:
                                with open(benchmark_path, 'r') as f:
                                    benchmark_data = json.load(f)
                                    
                                # Extract summary
                                if "results" in benchmark_data and "summary" in benchmark_data["results"]:
                                    summary = benchmark_data["results"]["summary"]
                                    
                                    benchmark_name = benchmark_path.split("_")[2]
                                    benchmarks[benchmark_name] = summary
                            except Exception as e:
                                logger.warning(f"Failed to load benchmark results {benchmark_path}: {e}")
                
                if benchmarks:
                    performance["benchmarks"] = benchmarks
        except Exception as e:
            logger.warning(f"Failed to collect benchmark results: {e}")
        
        return performance
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """
        Collect system information.
        
        Returns:
            Dictionary of system information
        """
        system_info = {}
        
        # Python version
        system_info["python_version"] = sys.version
        
        # Operating system
        system_info["os"] = sys.platform
        
        # CPU count
        try:
            import multiprocessing
            system_info["cpu_count"] = multiprocessing.cpu_count()
        except (ImportError, NotImplementedError):
            pass
        
        # Memory usage
        try:
            import psutil
            mem = psutil.virtual_memory()
            system_info["memory_total"] = mem.total
            system_info["memory_available"] = mem.available
            system_info["memory_used_percent"] = mem.percent
        except ImportError:
            pass
        
        # Disk usage
        try:
            import shutil
            disk = shutil.disk_usage(os.path.dirname(os.path.dirname(__file__)))
            system_info["disk_total"] = disk.total
            system_info["disk_used"] = disk.used
            system_info["disk_free"] = disk.free
        except (ImportError, OSError):
            pass
        
        # GPU information
        try:
            import torch
            system_info["cuda_available"] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                system_info["cuda_device_count"] = torch.cuda.device_count()
                system_info["cuda_device_name"] = torch.cuda.get_device_name(0)
                system_info["cuda_version"] = torch.version.cuda
        except ImportError:
            pass
        
        return system_info
    
    def _setup_wake_triggers(self) -> None:
        """Set up wake triggers."""
        logger.info("Setting up wake triggers")
        
        # Set up wake trigger daemon
        wake_trigger_script = os.path.join(os.path.dirname(__file__), "wake_triggers.py")
        
        if not os.path.exists(wake_trigger_script):
            logger.error(f"Wake trigger script not found: {wake_trigger_script}")
            return
        
        # Start wake trigger daemon
        try:
            # Create logs directory
            logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            # Create log file
            log_file = os.path.join(logs_dir, "wake_triggers.log")
            
            # Create daemon command
            cmd = [
                sys.executable,
                wake_trigger_script,
                "--daemon"
            ]
            
            if self.config_path:
                cmd.extend(["--config", self.config_path])
            
            if self.state_dir:
                cmd.extend(["--state-dir", self.state_dir])
            
            with open(log_file, 'a') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=f,
                    start_new_session=True
                )
            
            logger.info(f"Wake trigger daemon started with PID {process.pid}")
            
            # Update state
            if "components" not in self.state:
                self.state["components"] = {}
            
            if "services" not in self.state["components"]:
                self.state["components"]["services"] = {}
            
            self.state["components"]["services"]["wake_triggers"] = {
                "status": "running",
                "pid": process.pid,
                "start_time": int(time.time()),
                "log": log_file
            }
            
            # Save state
            self._save_state()
            
        except Exception as e:
            logger.error(f"Failed to start wake trigger daemon: {e}")

def main():
    """Run system standby from command line."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="CTM-AbsoluteZero Standby")
    
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
        "--no-report",
        help="Don't generate standby report",
        action="store_true"
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
    
    # Create system standby
    system_standby = SystemStandby(
        config_path=args.config,
        state_dir=args.state_dir,
        generate_report=not args.no_report
    )
    
    # Standby system
    success = system_standby.standby()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()