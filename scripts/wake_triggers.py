#!/usr/bin/env python3
"""
Wake Triggers for CTM-AbsoluteZero.

This script defines and checks for triggers that should wake up the system
from standby mode, such as new data, updates to dependencies, or scheduled tasks.
"""
import os
import sys
import argparse
import logging
import json
import time
import datetime
import asyncio
from typing import Dict, List, Any, Optional, Set
import subprocess
import hashlib

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.utils.logging import get_logger, configure_logging
from src.utils.config import ConfigManager

# Setup logger
logger = get_logger("ctm-az.wake_triggers")

class WakeManager:
    """Manager for system wake triggers."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        state_dir: str = "./state",
        trigger_interval: int = 3600,  # 1 hour
        high_priority_interval: int = 300  # 5 minutes
    ):
        """
        Initialize the wake manager.
        
        Args:
            config_path: Path to configuration file
            state_dir: Directory to store state information
            trigger_interval: Interval between trigger checks in seconds
            high_priority_interval: Interval between high priority trigger checks in seconds
        """
        self.config_path = config_path
        self.state_dir = state_dir
        self.trigger_interval = trigger_interval
        self.high_priority_interval = high_priority_interval
        
        # Create state directory
        os.makedirs(state_dir, exist_ok=True)
        
        # Load configuration
        self.config_manager = ConfigManager(config_path) if config_path else ConfigManager()
        self.config = self.config_manager.to_dict()
        
        # Load wake triggers
        self.triggers = self._load_triggers()
        
        # Load state
        self.state = self._load_state()
        
        # Create schedule
        self.schedule = self._create_schedule()
        
        logger.info(f"Wake manager initialized with {len(self.triggers)} triggers")
    
    def _load_triggers(self) -> Dict[str, Dict[str, Any]]:
        """
        Load wake triggers from configuration.
        
        Returns:
            Dictionary of trigger configurations
        """
        # Default triggers
        default_triggers = {
            "dependencies": {
                "type": "dependency",
                "priority": "medium",
                "paths": ["requirements.txt"],
                "enabled": True,
                "description": "Check for updates to dependencies"
            },
            "data": {
                "type": "data",
                "priority": "high",
                "paths": ["data"],
                "enabled": True,
                "description": "Check for new data"
            },
            "code": {
                "type": "code",
                "priority": "high",
                "paths": ["src"],
                "enabled": True,
                "description": "Check for code changes"
            },
            "daily": {
                "type": "schedule",
                "priority": "medium",
                "schedule": "daily",
                "time": "00:00",
                "enabled": True,
                "description": "Daily scheduled wake"
            },
            "weekly": {
                "type": "schedule",
                "priority": "low",
                "schedule": "weekly",
                "day": "monday",
                "time": "00:00",
                "enabled": True,
                "description": "Weekly scheduled wake"
            },
            "api_update": {
                "type": "api",
                "priority": "medium",
                "url": "https://api.example.com/status",
                "method": "GET",
                "enabled": False,
                "description": "Check for API status updates"
            }
        }
        
        # Load from configuration
        triggers = self.config.get("wake_triggers", default_triggers)
        
        return triggers
    
    def _load_state(self) -> Dict[str, Any]:
        """
        Load wake state from file.
        
        Returns:
            State dictionary
        """
        state_path = os.path.join(self.state_dir, "wake_state.json")
        
        # Default state
        default_state = {
            "last_check": 0,
            "last_wake": 0,
            "triggers": {},
            "hashes": {}
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
        """Save wake state to file."""
        state_path = os.path.join(self.state_dir, "wake_state.json")
        
        try:
            with open(state_path, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _create_schedule(self) -> Dict[str, Dict[str, Any]]:
        """
        Create schedule from triggers.
        
        Returns:
            Schedule dictionary
        """
        schedule = {}
        
        for trigger_id, trigger in self.triggers.items():
            if trigger.get("type") == "schedule" and trigger.get("enabled", True):
                schedule_type = trigger.get("schedule")
                
                if schedule_type == "daily":
                    time_str = trigger.get("time", "00:00")
                    hour, minute = map(int, time_str.split(":"))
                    
                    schedule[trigger_id] = {
                        "type": "daily",
                        "hour": hour,
                        "minute": minute,
                        "last_run": self.state.get("triggers", {}).get(trigger_id, {}).get("last_run", 0)
                    }
                    
                elif schedule_type == "weekly":
                    day = trigger.get("day", "monday").lower()
                    time_str = trigger.get("time", "00:00")
                    hour, minute = map(int, time_str.split(":"))
                    
                    day_map = {
                        "monday": 0,
                        "tuesday": 1,
                        "wednesday": 2,
                        "thursday": 3,
                        "friday": 4,
                        "saturday": 5,
                        "sunday": 6
                    }
                    
                    day_number = day_map.get(day, 0)
                    
                    schedule[trigger_id] = {
                        "type": "weekly",
                        "day": day_number,
                        "hour": hour,
                        "minute": minute,
                        "last_run": self.state.get("triggers", {}).get(trigger_id, {}).get("last_run", 0)
                    }
                    
                elif schedule_type == "monthly":
                    day = trigger.get("day", 1)
                    time_str = trigger.get("time", "00:00")
                    hour, minute = map(int, time_str.split(":"))
                    
                    schedule[trigger_id] = {
                        "type": "monthly",
                        "day": day,
                        "hour": hour,
                        "minute": minute,
                        "last_run": self.state.get("triggers", {}).get(trigger_id, {}).get("last_run", 0)
                    }
        
        return schedule
    
    async def run(self, daemon: bool = False) -> None:
        """
        Run the wake manager.
        
        Args:
            daemon: Whether to run as a daemon (continuous checks)
        """
        if daemon:
            logger.info("Running wake manager as daemon")
            
            while True:
                await self.check_triggers()
                
                # Wait for next check
                logger.info(f"Waiting {self.high_priority_interval}s until next high priority check")
                await asyncio.sleep(self.high_priority_interval)
        else:
            logger.info("Running wake manager once")
            await self.check_triggers()
    
    async def check_triggers(self) -> List[str]:
        """
        Check all triggers.
        
        Returns:
            List of triggered wake IDs
        """
        logger.info("Checking wake triggers")
        
        # Update state
        self.state["last_check"] = int(time.time())
        
        # Check triggers
        triggered = []
        
        for trigger_id, trigger in self.triggers.items():
            if not trigger.get("enabled", True):
                continue
                
            # Get priority
            priority = trigger.get("priority", "medium")
            
            # Skip low/medium priority triggers unless it's time for a full check
            if priority != "high" and (self.state["last_check"] - self.state.get("last_full_check", 0)) < self.trigger_interval:
                continue
            
            # Check trigger
            try:
                should_wake = await self._check_trigger(trigger_id, trigger)
                
                if should_wake:
                    triggered.append(trigger_id)
                    
                    logger.info(f"Trigger {trigger_id} activated")
                    
                    # Update trigger state
                    if "triggers" not in self.state:
                        self.state["triggers"] = {}
                    
                    if trigger_id not in self.state["triggers"]:
                        self.state["triggers"][trigger_id] = {}
                    
                    self.state["triggers"][trigger_id]["last_triggered"] = int(time.time())
            except Exception as e:
                logger.error(f"Error checking trigger {trigger_id}: {e}")
        
        # Update state if this was a full check
        if (self.state["last_check"] - self.state.get("last_full_check", 0)) >= self.trigger_interval:
            self.state["last_full_check"] = self.state["last_check"]
        
        # Save state
        self._save_state()
        
        if triggered:
            logger.info(f"Triggers activated: {', '.join(triggered)}")
            
            # Wake system
            await self._wake_system(triggered)
        else:
            logger.info("No triggers activated")
        
        return triggered
    
    async def _check_trigger(self, trigger_id: str, trigger: Dict[str, Any]) -> bool:
        """
        Check a single trigger.
        
        Args:
            trigger_id: Trigger ID
            trigger: Trigger configuration
            
        Returns:
            True if the trigger should wake the system, False otherwise
        """
        trigger_type = trigger.get("type")
        
        if trigger_type == "dependency":
            return await self._check_dependency_trigger(trigger_id, trigger)
        elif trigger_type == "data":
            return await self._check_data_trigger(trigger_id, trigger)
        elif trigger_type == "code":
            return await self._check_code_trigger(trigger_id, trigger)
        elif trigger_type == "schedule":
            return await self._check_schedule_trigger(trigger_id, trigger)
        elif trigger_type == "api":
            return await self._check_api_trigger(trigger_id, trigger)
        else:
            logger.warning(f"Unknown trigger type: {trigger_type}")
            return False
    
    async def _check_dependency_trigger(self, trigger_id: str, trigger: Dict[str, Any]) -> bool:
        """
        Check dependency trigger.
        
        Args:
            trigger_id: Trigger ID
            trigger: Trigger configuration
            
        Returns:
            True if dependencies have changed, False otherwise
        """
        paths = trigger.get("paths", [])
        
        for path in paths:
            if not os.path.exists(path):
                continue
                
            # Calculate hash
            file_hash = self._calculate_file_hash(path)
            
            # Check if hash has changed
            if "hashes" not in self.state:
                self.state["hashes"] = {}
            
            if path not in self.state["hashes"]:
                self.state["hashes"][path] = file_hash
                continue
            
            if self.state["hashes"][path] != file_hash:
                logger.info(f"Dependency change detected in {path}")
                
                # Update hash
                self.state["hashes"][path] = file_hash
                
                return True
        
        return False
    
    async def _check_data_trigger(self, trigger_id: str, trigger: Dict[str, Any]) -> bool:
        """
        Check data trigger.
        
        Args:
            trigger_id: Trigger ID
            trigger: Trigger configuration
            
        Returns:
            True if data has changed, False otherwise
        """
        paths = trigger.get("paths", [])
        
        for path in paths:
            if not os.path.exists(path):
                continue
                
            # Calculate directory hash
            dir_hash = await self._calculate_directory_hash(path)
            
            # Check if hash has changed
            if "hashes" not in self.state:
                self.state["hashes"] = {}
            
            if path not in self.state["hashes"]:
                self.state["hashes"][path] = dir_hash
                continue
            
            if self.state["hashes"][path] != dir_hash:
                logger.info(f"Data change detected in {path}")
                
                # Update hash
                self.state["hashes"][path] = dir_hash
                
                return True
        
        return False
    
    async def _check_code_trigger(self, trigger_id: str, trigger: Dict[str, Any]) -> bool:
        """
        Check code trigger.
        
        Args:
            trigger_id: Trigger ID
            trigger: Trigger configuration
            
        Returns:
            True if code has changed, False otherwise
        """
        paths = trigger.get("paths", [])
        
        for path in paths:
            if not os.path.exists(path):
                continue
                
            # Calculate directory hash
            dir_hash = await self._calculate_directory_hash(path, extensions=[".py", ".c", ".cpp", ".h", ".js", ".ts"])
            
            # Check if hash has changed
            if "hashes" not in self.state:
                self.state["hashes"] = {}
            
            if path not in self.state["hashes"]:
                self.state["hashes"][path] = dir_hash
                continue
            
            if self.state["hashes"][path] != dir_hash:
                logger.info(f"Code change detected in {path}")
                
                # Update hash
                self.state["hashes"][path] = dir_hash
                
                return True
        
        return False
    
    async def _check_schedule_trigger(self, trigger_id: str, trigger: Dict[str, Any]) -> bool:
        """
        Check schedule trigger.
        
        Args:
            trigger_id: Trigger ID
            trigger: Trigger configuration
            
        Returns:
            True if it's time to wake, False otherwise
        """
        if trigger_id not in self.schedule:
            return False
            
        schedule = self.schedule[trigger_id]
        schedule_type = schedule["type"]
        
        # Get current time
        now = datetime.datetime.now()
        current_timestamp = int(time.time())
        
        # Get last run
        last_run = schedule["last_run"]
        
        # Check if it's time to run
        if schedule_type == "daily":
            # Check if it's the right time
            if now.hour == schedule["hour"] and now.minute == schedule["minute"]:
                # Check if we've already run today
                last_run_date = datetime.datetime.fromtimestamp(last_run).date()
                
                if last_run_date < now.date():
                    logger.info(f"Daily schedule triggered for {trigger_id}")
                    
                    # Update last run
                    self.schedule[trigger_id]["last_run"] = current_timestamp
                    
                    if "triggers" not in self.state:
                        self.state["triggers"] = {}
                    
                    if trigger_id not in self.state["triggers"]:
                        self.state["triggers"][trigger_id] = {}
                    
                    self.state["triggers"][trigger_id]["last_run"] = current_timestamp
                    
                    return True
        
        elif schedule_type == "weekly":
            # Check if it's the right day and time
            if now.weekday() == schedule["day"] and now.hour == schedule["hour"] and now.minute == schedule["minute"]:
                # Check if we've already run this week
                last_run_date = datetime.datetime.fromtimestamp(last_run).date()
                days_diff = (now.date() - last_run_date).days
                
                if days_diff >= 7:
                    logger.info(f"Weekly schedule triggered for {trigger_id}")
                    
                    # Update last run
                    self.schedule[trigger_id]["last_run"] = current_timestamp
                    
                    if "triggers" not in self.state:
                        self.state["triggers"] = {}
                    
                    if trigger_id not in self.state["triggers"]:
                        self.state["triggers"][trigger_id] = {}
                    
                    self.state["triggers"][trigger_id]["last_run"] = current_timestamp
                    
                    return True
        
        elif schedule_type == "monthly":
            # Check if it's the right day and time
            if now.day == schedule["day"] and now.hour == schedule["hour"] and now.minute == schedule["minute"]:
                # Check if we've already run this month
                last_run_date = datetime.datetime.fromtimestamp(last_run).date()
                
                if last_run_date.month != now.month or last_run_date.year != now.year:
                    logger.info(f"Monthly schedule triggered for {trigger_id}")
                    
                    # Update last run
                    self.schedule[trigger_id]["last_run"] = current_timestamp
                    
                    if "triggers" not in self.state:
                        self.state["triggers"] = {}
                    
                    if trigger_id not in self.state["triggers"]:
                        self.state["triggers"][trigger_id] = {}
                    
                    self.state["triggers"][trigger_id]["last_run"] = current_timestamp
                    
                    return True
        
        return False
    
    async def _check_api_trigger(self, trigger_id: str, trigger: Dict[str, Any]) -> bool:
        """
        Check API trigger.
        
        Args:
            trigger_id: Trigger ID
            trigger: Trigger configuration
            
        Returns:
            True if API status has changed, False otherwise
        """
        url = trigger.get("url")
        method = trigger.get("method", "GET")
        
        if not url:
            return False
            
        # Skip API checks in this implementation
        return False
    
    def _calculate_file_hash(self, path: str) -> str:
        """
        Calculate hash for a file.
        
        Args:
            path: File path
            
        Returns:
            File hash
        """
        if not os.path.isfile(path):
            return ""
            
        hasher = hashlib.md5()
        
        with open(path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        
        return hasher.hexdigest()
    
    async def _calculate_directory_hash(
        self,
        path: str,
        extensions: Optional[List[str]] = None
    ) -> str:
        """
        Calculate hash for a directory.
        
        Args:
            path: Directory path
            extensions: Optional list of file extensions to include
            
        Returns:
            Directory hash
        """
        if not os.path.isdir(path):
            return ""
            
        hasher = hashlib.md5()
        
        # Get file list
        file_list = []
        
        for root, dirs, files in os.walk(path):
            for file in files:
                if extensions and not any(file.endswith(ext) for ext in extensions):
                    continue
                    
                file_path = os.path.join(root, file)
                file_list.append(file_path)
        
        # Sort file list for consistent hashing
        file_list.sort()
        
        # Calculate hash for each file
        for file_path in file_list:
            file_hash = self._calculate_file_hash(file_path)
            hasher.update(file_hash.encode())
        
        return hasher.hexdigest()
    
    async def _wake_system(self, triggers: List[str]) -> None:
        """
        Wake the system.
        
        Args:
            triggers: List of triggered wake IDs
        """
        logger.info(f"Waking system due to triggers: {', '.join(triggers)}")
        
        # Update state
        self.state["last_wake"] = int(time.time())
        self.state["last_wake_triggers"] = triggers
        
        # Save state
        self._save_state()
        
        # Check if wake script exists
        wake_script = os.path.join(os.path.dirname(__file__), "wake.py")
        
        if os.path.exists(wake_script):
            # Run wake script
            try:
                subprocess.run([sys.executable, wake_script, "--triggers", ",".join(triggers)])
            except Exception as e:
                logger.error(f"Failed to run wake script: {e}")
        else:
            logger.warning(f"Wake script not found: {wake_script}")

def main():
    """Run wake triggers from command line."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="CTM-AbsoluteZero Wake Triggers")
    
    # Add arguments
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file",
        type=str
    )
    parser.add_argument(
        "--state-dir", "-s",
        help="Directory to store state information",
        type=str,
        default="./state"
    )
    parser.add_argument(
        "--daemon", "-d",
        help="Run as daemon",
        action="store_true"
    )
    parser.add_argument(
        "--interval", "-i",
        help="Interval between trigger checks in seconds",
        type=int,
        default=3600  # 1 hour
    )
    parser.add_argument(
        "--high-priority-interval",
        help="Interval between high priority trigger checks in seconds",
        type=int,
        default=300  # 5 minutes
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
    
    # Create wake manager
    wake_manager = WakeManager(
        config_path=args.config,
        state_dir=args.state_dir,
        trigger_interval=args.interval,
        high_priority_interval=args.high_priority_interval
    )
    
    # Run wake manager
    asyncio.run(wake_manager.run(daemon=args.daemon))

if __name__ == "__main__":
    main()