"""
Configuration utilities for CTM-AbsoluteZero.
"""
import os
import yaml
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger("ctm-az.config")

class ConfigManager:
    """Configuration manager for CTM-AbsoluteZero."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = {}
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration
        """
        logger.info(f"Loading configuration from {config_path}")
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            return self.config
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """
        Save configuration to a file.
        
        Args:
            config_path: Path to save configuration to (default: self.config_path)
        """
        save_path = config_path or self.config_path
        if not save_path:
            logger.error("No configuration path specified")
            raise ValueError("No configuration path specified")
            
        logger.info(f"Saving configuration to {save_path}")
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Failed to save configuration to {save_path}: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (supports dot notation for nested values)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if "." not in key:
            return self.config.get(key, default)
            
        # Handle nested keys
        parts = key.split(".")
        current = self.config
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (supports dot notation for nested values)
            value: Value to set
        """
        if "." not in key:
            self.config[key] = value
            return
            
        # Handle nested keys
        parts = key.split(".")
        current = self.config
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    
    def update(self, config: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            config: New configuration values
        """
        self.config.update(config)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration as a dictionary
        """
        return self.config


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge two configurations.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if (
            key in result and
            isinstance(result[key], dict) and
            isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
            
    return result