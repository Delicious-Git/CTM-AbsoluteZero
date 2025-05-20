"""
Logging utilities for CTM-AbsoluteZero.
"""
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL = logging.INFO

class CTMLogger:
    """Logging manager for CTM-AbsoluteZero."""
    
    def __init__(
        self,
        name: str = "ctm-az",
        log_level: int = DEFAULT_LOG_LEVEL,
        log_format: str = DEFAULT_LOG_FORMAT,
        log_file: Optional[str] = None,
        console_output: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            log_level: Logging level
            log_format: Log message format
            log_file: Path to log file (if None, no file logging)
            console_output: Whether to output logs to console
            max_file_size: Maximum log file size before rotation
            backup_count: Number of backup files to keep
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.log_format = log_format
        self.formatter = logging.Formatter(log_format)
        
        # Clear existing handlers if any
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self.formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger."""
        return self.logger


class PerformanceTracker:
    """Track performance metrics of operations."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the performance tracker.
        
        Args:
            logger: Logger to use (if None, creates a default one)
        """
        self.logger = logger or logging.getLogger("ctm-az.performance")
        self.timers = {}
        self.counters = {}
        self.metrics = {}
    
    def start_timer(self, name: str) -> None:
        """Start a timer with the given name."""
        self.timers[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """
        Stop the timer with the given name and return elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        if name not in self.timers:
            self.logger.warning(f"Timer '{name}' does not exist")
            return 0.0
            
        elapsed = time.time() - self.timers[name]
        self.logger.debug(f"Timer '{name}' elapsed: {elapsed:.4f}s")
        
        # Store in metrics
        metric_name = f"time_{name}"
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(elapsed)
        
        return elapsed
    
    def increment_counter(self, name: str, value: int = 1) -> int:
        """
        Increment a counter by the given value.
        
        Args:
            name: Counter name
            value: Value to increment by
            
        Returns:
            New counter value
        """
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += value
        
        # Store in metrics
        metric_name = f"count_{name}"
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(self.counters[name])
        
        return self.counters[name]
    
    def get_counter(self, name: str) -> int:
        """Get the current value of a counter."""
        return self.counters.get(name, 0)
    
    def record_metric(self, name: str, value: Any) -> None:
        """Record a custom metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_metrics(self) -> Dict[str, list]:
        """Get all recorded metrics."""
        return self.metrics
    
    def log_metrics(self, level: int = logging.INFO) -> None:
        """Log all metrics at the specified level."""
        for name, values in self.metrics.items():
            if values:
                avg = sum(values) / len(values) if isinstance(values[0], (int, float)) else None
                self.logger.log(
                    level,
                    f"Metric {name}: latest={values[-1]}, "
                    f"count={len(values)}"
                    + (f", avg={avg:.4f}" if avg is not None else "")
                )
                
    def reset(self) -> None:
        """Reset all timers, counters and metrics."""
        self.timers = {}
        self.counters = {}
        self.metrics = {}


# Create default logger
default_logger = CTMLogger().get_logger()

# Create default performance tracker
default_tracker = PerformanceTracker(default_logger)

def get_logger(name: str = "ctm-az") -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def configure_logging(
    log_level: int = DEFAULT_LOG_LEVEL,
    log_format: str = DEFAULT_LOG_FORMAT,
    log_file: Optional[str] = None,
    console_output: bool = True,
) -> None:
    """
    Configure the root logger.
    
    Args:
        log_level: Logging level
        log_format: Log message format
        log_file: Path to log file
        console_output: Whether to output logs to console
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)