"""
Solver loader for Universal Router.

This module provides functionality to dynamically load solvers for the Universal Router.
"""
import os
import sys
import importlib
import importlib.util
import logging
import inspect
from typing import Dict, List, Any, Optional, Type, Callable

from .universal_router import UniversalRouter, SolverInterface
from ..utils.logging import get_logger

# Setup logger
logger = get_logger("ctm-az.router.loader")

def discover_solvers(
    paths: List[str] = None,
    base_class: Type = SolverInterface
) -> List[Type[SolverInterface]]:
    """
    Discover solver classes in the given paths.
    
    Args:
        paths: Paths to search for solver classes
        base_class: Base class that solvers must inherit from
        
    Returns:
        List of discovered solver classes
    """
    # Default paths
    if paths is None:
        paths = [
            os.path.join(os.path.dirname(__file__), "..", "solvers"),
            os.path.join(os.path.dirname(__file__), "..", "ctm")
        ]
    
    # Initialize results
    solver_classes = []
    
    # Search for solver modules
    for path in paths:
        if not os.path.exists(path):
            logger.warning(f"Path does not exist: {path}")
            continue
            
        logger.info(f"Searching for solvers in: {path}")
        
        # Get Python files in path
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(".py") and not file.startswith("__"):
                        file_path = os.path.join(root, file)
                        
                        # Load module
                        solver_classes.extend(_load_solver_classes_from_file(file_path, base_class))
        elif os.path.isfile(path) and path.endswith(".py"):
            # Load module
            solver_classes.extend(_load_solver_classes_from_file(path, base_class))
    
    logger.info(f"Discovered {len(solver_classes)} solver classes")
    
    return solver_classes

def _load_solver_classes_from_file(
    file_path: str,
    base_class: Type
) -> List[Type[SolverInterface]]:
    """
    Load solver classes from a Python file.
    
    Args:
        file_path: Path to Python file
        base_class: Base class that solvers must inherit from
        
    Returns:
        List of solver classes in the file
    """
    try:
        # Get module name
        module_name = os.path.basename(file_path)[:-3]  # Remove .py extension
        
        # Load module from file
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find solver classes
        solver_classes = []
        
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Check if class inherits from base_class and is not the base_class itself
            if (
                issubclass(obj, base_class) and
                obj != base_class and
                obj.__module__ == module.__name__
            ):
                solver_classes.append(obj)
                logger.debug(f"Found solver class: {obj.__name__}")
        
        return solver_classes
        
    except Exception as e:
        logger.error(f"Failed to load solver classes from {file_path}: {e}")
        return []

def load_solvers(
    router: UniversalRouter,
    paths: List[str] = None,
    config: Dict[str, Any] = None
) -> int:
    """
    Load solvers into the Universal Router.
    
    Args:
        router: Universal Router instance
        paths: Paths to search for solver classes
        config: Configuration for solvers
        
    Returns:
        Number of solvers loaded
    """
    # Discover solver classes
    solver_classes = discover_solvers(paths)
    
    # Config for solvers
    solver_config = config or {}
    
    # Load solvers
    loaded_count = 0
    
    for solver_class in solver_classes:
        try:
            # Get solver name
            solver_name = solver_class.__name__
            
            # Get config for this solver
            specific_config = solver_config.get(solver_name, {})
            
            # Check if solver is enabled
            if not specific_config.get("enabled", True):
                logger.info(f"Solver {solver_name} is disabled, skipping")
                continue
            
            # Get domains
            domains = specific_config.get("domains")
            
            # Create solver instance
            solver = solver_class(solver_name, specific_config)
            
            # Register solver
            router.register_solver(solver, domains)
            
            loaded_count += 1
            
        except Exception as e:
            logger.error(f"Failed to load solver {solver_class.__name__}: {e}")
    
    logger.info(f"Loaded {loaded_count} solvers")
    
    return loaded_count