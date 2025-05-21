"""
Implementation of the AbsoluteZero Agent for production use with CTM.
This is a more modern and modular implementation of the CTM_AbsoluteZero_Agent.
"""
import os
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import random
import uuid
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Internal imports
from .rewards.composite import CompositeRewardSystem
from .rewards.novelty import SemanticNoveltyTracker
from .rewards.progress import SkillPyramid
from .transfer.adapter import NeuralTransferAdapter
from .transfer.phase import PhaseController
from .ctm.interface import RealCTMInterface
from .utils.logging import get_logger

logger = get_logger("ctm-az.absolute_zero")

class AbsoluteZeroAgent:
    """
    AbsoluteZero Agent - A modular, production-ready implementation of the
    Absolute Zero Reasoner paradigm with CTM integration.
    
    This agent uses a Proposer-Solver architecture where:
    - Proposer: A Large Language Model that generates tasks
    - Solver: A CTM instance that executes the tasks
    
    The agent incorporates advanced reward systems, cross-domain knowledge transfer,
    and phase-based training for optimal performance.
    """
    
    def __init__(
        self,
        proposer_model_path: str,
        proposer_tokenizer_path: Optional[str] = None,
        solver_model_path: Optional[str] = None,
        solver_tokenizer_path: Optional[str] = None,
        reward_system: Optional[CompositeRewardSystem] = None,
        transfer_adapter: Optional[NeuralTransferAdapter] = None,
        ctm_interface: Optional[RealCTMInterface] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the AbsoluteZero Agent.
        
        Args:
            proposer_model_path: Path to the Proposer model
            proposer_tokenizer_path: Path to the Proposer tokenizer (defaults to model_path)
            solver_model_path: Path to the Solver model (optional, CTM may handle this)
            solver_tokenizer_path: Path to the Solver tokenizer (optional)
            reward_system: Reward system instance (created if not provided)
            transfer_adapter: Transfer adapter instance (created if not provided)
            ctm_interface: CTM interface instance (created if not provided)
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Device configuration
        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"Using device: {self.device}")
        
        # Initialize Proposer
        self._initialize_proposer(
            proposer_model_path, 
            proposer_tokenizer_path or proposer_model_path
        )
        
        # Initialize Solver (optional, may be handled by CTM)
        if solver_model_path:
            self._initialize_solver(
                solver_model_path,
                solver_tokenizer_path or solver_model_path
            )
        
        # Set up CTM interface
        self.ctm_interface = ctm_interface
        if not self.ctm_interface:
            ctm_config = self.config.get("ctm", {})
            self.ctm_interface = RealCTMInterface(ctm_config)
        
        # Set up reward system
        self.reward_system = reward_system
        if not self.reward_system:
            self._initialize_reward_system()
        
        # Set up transfer adapter
        self.transfer_adapter = transfer_adapter
        if not self.transfer_adapter:
            domains = self.config.get("domains", ["general"])
            self.transfer_adapter = NeuralTransferAdapter(domains)
        
        # Phase controller (shared with reward system)
        if hasattr(self.reward_system, "phase_controller"):
            self.phase_controller = self.reward_system.phase_controller
        else:
            self.phase_controller = PhaseController(
                phase_duration=self.config.get("phase_duration", 600),
                initial_phase=self.config.get("initial_phase", "exploration")
            )
        
        # Task history and tracking
        self.task_history = []
        self.performance_metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "avg_reward": 0.0,
            "domain_metrics": {}
        }
        
        # Global step counter
        self.global_step = 0
        
        logger.info("AbsoluteZero Agent initialized successfully")
    
    def _initialize_proposer(self, model_path: str, tokenizer_path: str) -> None:
        """
        Initialize the Proposer LLM.
        
        Args:
            model_path: Path to the model
            tokenizer_path: Path to the tokenizer
        """
        logger.info(f"Initializing Proposer from {model_path}")
        
        try:
            # Handle local and HuggingFace Hub paths
            self.proposer_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.proposer_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.config.get("fp16", True) else torch.float32
            ).to(self.device)
            
            # Ensure padding token is set
            if self.proposer_tokenizer.pad_token is None:
                self.proposer_tokenizer.pad_token = self.proposer_tokenizer.eos_token
                
            logger.info("Proposer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Proposer: {e}")
            # Use a fallback or raise error based on configuration
            if self.config.get("require_proposer", True):
                raise
            else:
                logger.warning("Continuing without Proposer model")
                self.proposer_model = None
                self.proposer_tokenizer = None
    
    def _initialize_solver(self, model_path: str, tokenizer_path: str) -> None:
        """
        Initialize the optional Solver LLM (if separate from CTM).
        
        Args:
            model_path: Path to the model
            tokenizer_path: Path to the tokenizer
        """
        logger.info(f"Initializing Solver from {model_path}")
        
        try:
            self.solver_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.solver_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.config.get("fp16", True) else torch.float32
            ).to(self.device)
            
            if self.solver_tokenizer.pad_token is None:
                self.solver_tokenizer.pad_token = self.solver_tokenizer.eos_token
                
            logger.info("Solver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Solver: {e}")
            self.solver_model = None
            self.solver_tokenizer = None
    
    def _initialize_reward_system(self) -> None:
        """Initialize the reward system components."""
        # Set up novelty tracker
        reward_config = self.config.get("rewards", {})
        novelty_tracker = SemanticNoveltyTracker(
            embedding_dim=reward_config.get("embedding_dim", 768),
            novelty_threshold=reward_config.get("novelty_threshold", 0.2)
        )
        
        # Set up skill pyramid
        domains = self.config.get("domains", ["general"])
        skill_pyramid = SkillPyramid(
            domains=domains,
            levels=reward_config.get("skill_levels", 5)
        )
        
        # Set up phase controller
        self.phase_controller = PhaseController(
            phase_duration=self.config.get("phase_duration", 600),
            initial_phase=self.config.get("initial_phase", "exploration")
        )
        
        # Create reward system
        self.reward_system = CompositeRewardSystem(
            novelty_tracker=novelty_tracker,
            skill_pyramid=skill_pyramid,
            phase_controller=self.phase_controller,
            hyperparams=reward_config.get("hyperparams")
        )
        
        logger.info("Reward system initialized successfully")
    
    def _generate_proposer_prompt(
        self, 
        domain: str, 
        feedback: str = "",
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate a prompt for the Proposer LLM.
        
        Args:
            domain: Task domain
            feedback: Optional feedback on past performance
            conversation_history: Optional conversation history
            
        Returns:
            Formatted prompt text
        """
        # Domain-specific examples
        examples = {
            "maze": {
                "id": "task_12345",
                "domain": "maze",
                "description": "Navigate through a complex maze with branching paths",
                "parameters": {
                    "size_x": 12, 
                    "size_y": 12, 
                    "complexity": 0.8, 
                    "seed": 42,
                    "visual_patterns": False
                }
            },
            "quantum": {
                "id": "task_67890",
                "domain": "quantum",
                "description": "Implement a quantum variational circuit for energy minimization",
                "parameters": {
                    "algorithm": "vqe",
                    "num_qubits": 5,
                    "noise_level": 0.02,
                    "circuit_depth": 6
                }
            },
            "general": {
                "id": "task_13579",
                "domain": "general",
                "description": "Create an adaptive learning algorithm for mixed inputs",
                "parameters": {
                    "difficulty": 0.7,
                    "inputs": ["numeric", "text", "categorical"],
                    "expected_accuracy": 0.85
                }
            }
        }
        
        # Fall back to generic example if domain not found
        if domain not in examples:
            examples[domain] = examples["general"].copy()
            examples[domain]["domain"] = domain
        
        # Build prompt
        prompt = "# CTM-AbsoluteZero Task Generator\n\n"
        prompt += f"You are a specialized task generator for the CTM-AbsoluteZero system.\n"
        prompt += f"Domain: {domain}\n\n"
        
        # Add conversation context if provided
        if conversation_history:
            prompt += "## Recent Conversation\n"
            for entry in conversation_history[-3:]:  # Last 3 exchanges
                role = entry.get("role", "unknown")
                content = entry.get("content", "")
                prompt += f"{role}: {content}\n"
            prompt += "\n"
        
        # Add feedback if provided
        if feedback:
            prompt += f"## Performance Feedback\n{feedback}\n\n"
        
        # Task format instructions
        prompt += "## Task Generation Instructions\n"
        prompt += "Generate a challenging but solvable task with the following properties:\n"
        prompt += "- Target difficulty level: medium to hard\n"
        prompt += "- Success rate should be around 50-70%\n"
        prompt += "- Task should be specific and measurable\n"
        prompt += "- Include clear constraints and objectives\n\n"
        
        # Example format
        prompt += "## Example Task Format\n"
        prompt += json.dumps(examples[domain], indent=2) + "\n\n"
        
        # Domain-specific constraints
        if domain == "maze":
            prompt += "## Domain Constraints\n"
            prompt += "- size_x and size_y must be between 5-20\n"
            prompt += "- complexity must be between 0.1-0.9\n\n"
        elif domain == "quantum":
            prompt += "## Domain Constraints\n"
            prompt += "- num_qubits must be between 2-10\n"
            prompt += "- algorithm must be one of: vqe, grover, qft\n\n"
        
        prompt += "## Generate New Task\n"
        prompt += "Create a new task for this domain. Respond with ONLY the JSON object:\n"
        
        return prompt
    
    def generate_tasks(
        self, 
        domain: str = "general", 
        count: int = 3,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple tasks for a specific domain.
        
        Args:
            domain: Task domain
            count: Number of tasks to generate
            conversation_history: Optional conversation history for context
            
        Returns:
            List of generated tasks
        """
        logger.info(f"Generating {count} tasks for domain: {domain}")
        
        tasks = []
        start_time = time.time()
        
        for i in range(count):
            # Generate task with slight variations in feedback
            if i == 0:
                feedback = "Generate a challenging task for this domain."
            elif i == 1:
                feedback = "The previous task was good. Create a different variation."
            else:
                feedback = "Create a task with a novel approach or twist."
                
            try:
                task = self._generate_single_task(domain, feedback, conversation_history)
                tasks.append(task)
            except Exception as e:
                logger.error(f"Failed to generate task {i+1}/{count}: {e}")
        
        duration = time.time() - start_time
        logger.info(f"Generated {len(tasks)} tasks in {duration:.2f}s")
        
        return tasks
    
    def _generate_single_task(
        self,
        domain: str,
        feedback: str = "",
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a single task using the Proposer.
        
        Args:
            domain: Task domain
            feedback: Optional feedback
            conversation_history: Optional conversation history
            
        Returns:
            Generated task dictionary
        """
        # Create prompt
        prompt = self._generate_proposer_prompt(domain, feedback, conversation_history)
        
        if not self.proposer_model or not self.proposer_tokenizer:
            # Fallback for when no model is available
            logger.warning("No Proposer model available, using template task")
            return self._generate_template_task(domain)
        
        # Generate task with the model
        try:
            inputs = self.proposer_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generation parameters
            max_tokens = self.config.get("max_tokens", 1024)
            temperature = self.config.get("temperature", 0.7)
            top_p = self.config.get("top_p", 0.9)
            
            # Generate response
            with torch.no_grad():
                output_ids = self.proposer_model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.proposer_tokenizer.pad_token_id
                )
            
            # Decode response
            response = self.proposer_tokenizer.decode(
                output_ids[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # Parse JSON from response
            return self._parse_task_json(response, domain)
        except Exception as e:
            logger.error(f"Error generating task: {e}")
            return self._generate_template_task(domain)
    
    def _parse_task_json(self, text: str, domain: str) -> Dict[str, Any]:
        """
        Parse JSON task from generated text.
        
        Args:
            text: Generated text
            domain: Task domain for fallback
            
        Returns:
            Task dictionary
        """
        # Extract JSON from text
        try:
            # Find JSON object in text
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx+1]
                task = json.loads(json_str)
                
                # Ensure required fields
                if "id" not in task:
                    task["id"] = f"task_{uuid.uuid4().hex[:8]}"
                
                if "domain" not in task:
                    task["domain"] = domain
                
                if "description" not in task:
                    task["description"] = f"Generated task for {domain} domain"
                
                if "parameters" not in task:
                    task["parameters"] = {}
                
                return task
            else:
                logger.warning(f"No JSON found in generated text: {text}")
                return self._generate_template_task(domain)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            return self._generate_template_task(domain)
    
    def _generate_template_task(self, domain: str) -> Dict[str, Any]:
        """
        Generate a template task when model generation fails.
        
        Args:
            domain: Task domain
            
        Returns:
            Template task dictionary
        """
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        if domain == "maze":
            return {
                "id": task_id,
                "domain": "maze",
                "description": "Navigate through a procedurally generated maze",
                "parameters": {
                    "size_x": random.randint(8, 15),
                    "size_y": random.randint(8, 15),
                    "complexity": random.uniform(0.4, 0.8),
                    "seed": random.randint(1, 1000)
                }
            }
        elif domain == "quantum":
            algorithms = ["vqe", "grover", "qft"]
            return {
                "id": task_id,
                "domain": "quantum",
                "description": f"Execute a {random.choice(algorithms)} quantum algorithm",
                "parameters": {
                    "algorithm": random.choice(algorithms),
                    "num_qubits": random.randint(3, 8),
                    "noise_level": random.uniform(0.01, 0.1),
                    "circuit_depth": random.randint(3, 8)
                }
            }
        else:
            # Generic template for other domains
            return {
                "id": task_id,
                "domain": domain,
                "description": f"Complete a task in the {domain} domain",
                "parameters": {
                    "difficulty": random.uniform(0.4, 0.8),
                    "seed": random.randint(1, 1000)
                }
            }
    
    def solve_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve a task using the CTM interface.
        
        Args:
            task: Task dictionary containing domain, description, and parameters
            
        Returns:
            Solution results
        """
        task_id = task.get("id", f"task_{uuid.uuid4().hex[:8]}")
        domain = task.get("domain", "general")
        description = task.get("description", "")
        parameters = task.get("parameters", {})
        
        logger.info(f"Solving task: {task_id} ({domain})")
        logger.debug(f"Task description: {description}")
        logger.debug(f"Task parameters: {parameters}")
        
        start_time = time.time()
        
        try:
            # Prepare task for CTM
            ctm_task = {
                "type": domain,
                "params": parameters
            }
            
            # Execute task
            result = self.ctm_interface.execute_task(ctm_task)
            
            # Calculate performance metrics
            duration = time.time() - start_time
            success = result.get("main_score", 0.0) >= self.config.get("success_threshold", 0.6)
            
            # Update metrics
            self.performance_metrics["total_tasks"] += 1
            if success:
                self.performance_metrics["successful_tasks"] += 1
            else:
                self.performance_metrics["failed_tasks"] += 1
            
            # Domain-specific metrics
            if domain not in self.performance_metrics["domain_metrics"]:
                self.performance_metrics["domain_metrics"][domain] = {
                    "total": 0,
                    "successful": 0,
                    "avg_score": 0.0,
                    "avg_time": 0.0
                }
            
            domain_metrics = self.performance_metrics["domain_metrics"][domain]
            domain_metrics["total"] += 1
            if success:
                domain_metrics["successful"] += 1
            
            # Update running averages
            main_score = result.get("main_score", 0.0)
            prev_avg = domain_metrics["avg_score"]
            domain_metrics["avg_score"] = (prev_avg * (domain_metrics["total"] - 1) + main_score) / domain_metrics["total"]
            
            prev_time = domain_metrics["avg_time"]
            domain_metrics["avg_time"] = (prev_time * (domain_metrics["total"] - 1) + duration) / domain_metrics["total"]
            
            # Calculate reward
            reward = 0.0
            if self.reward_system:
                reward = self.reward_system.calculate_reward(result, ctm_task)
            
            # Store in task history
            task_result = {
                "task_id": task_id,
                "domain": domain,
                "description": description,
                "parameters": parameters,
                "result": result,
                "success": success,
                "reward": reward,
                "duration": duration,
                "timestamp": time.time()
            }
            
            # Keep history to configured limit
            history_limit = self.config.get("task_history_size", 100)
            self.task_history.append(task_result)
            if len(self.task_history) > history_limit:
                self.task_history = self.task_history[-history_limit:]
            
            # Include summary in returned result
            solution = {
                "task_id": task_id,
                "success": success,
                "score": main_score,
                "reward": reward,
                "duration": duration,
                "details": result.get("details", ""),
                "metrics": {k: v for k, v in result.items() if k not in ["details"]}
            }
            
            logger.info(f"Task {task_id} solved in {duration:.2f}s with score {main_score:.3f}")
            return solution
        except Exception as e:
            logger.error(f"Error solving task {task_id}: {e}")
            
            # Update metrics
            self.performance_metrics["total_tasks"] += 1
            self.performance_metrics["failed_tasks"] += 1
            
            # Return error result
            return {
                "task_id": task_id,
                "success": False,
                "score": 0.0,
                "reward": 0.0,
                "duration": time.time() - start_time,
                "error": str(e)
            }
    
    def train(
        self,
        domain: str = "general",
        max_iterations: int = 100,
        eval_interval: int = 10
    ) -> Dict[str, Any]:
        """
        Run a training session to improve the agent's performance.
        
        Args:
            domain: Task domain to train on
            max_iterations: Maximum number of training iterations
            eval_interval: Interval for evaluation
            
        Returns:
            Training results summary
        """
        logger.info(f"Starting training on domain: {domain}")
        logger.info(f"Max iterations: {max_iterations}")
        logger.info(f"Evaluation interval: {eval_interval}")
        
        training_start = time.time()
        rewards = []
        eval_scores = []
        
        try:
            for iteration in range(1, max_iterations + 1):
                logger.info(f"Training iteration {iteration}/{max_iterations}")
                
                # Generate a task
                task = self._generate_single_task(domain)
                
                # Solve the task
                solution = self.solve_task(task)
                reward = solution.get("reward", 0.0)
                rewards.append(reward)
                
                # Update global step
                self.global_step += 1
                
                # Update phase controller
                if iteration % 10 == 0:
                    success_rate = self.performance_metrics["successful_tasks"] / max(1, self.performance_metrics["total_tasks"])
                    
                    # Get cross-domain correlation
                    cross_domain_corr = 0.5  # Default value
                    if hasattr(self.transfer_adapter, "get_correlation"):
                        cross_domain_corr = self.transfer_adapter.get_correlation()
                    
                    # Update phase based on performance
                    self.phase_controller.update_phase({
                        "success_rate": success_rate,
                        "cross_domain_corr": cross_domain_corr
                    })
                
                # Run evaluation
                if iteration % eval_interval == 0 or iteration == max_iterations:
                    logger.info(f"Running evaluation at iteration {iteration}")
                    eval_result = self.evaluate(domain, num_tasks=5)
                    eval_scores.append(eval_result["success_rate"])
                    
                    # Log evaluation results
                    logger.info(f"Evaluation results - Success rate: {eval_result['success_rate'] * 100:.1f}%, "
                               f"Avg reward: {eval_result['avg_reward']:.3f}")
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Error during training: {e}")
        
        # Calculate training summary
        training_time = time.time() - training_start
        
        return {
            "domain": domain,
            "iterations": min(iteration, max_iterations),
            "training_time": training_time,
            "rewards": rewards,
            "eval_scores": eval_scores,
            "final_success_rate": eval_scores[-1] if eval_scores else 0.0,
            "avg_reward": sum(rewards) / max(1, len(rewards))
        }
    
    def evaluate(
        self,
        domain: str = "general",
        num_tasks: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate the agent's performance on a set of tasks.
        
        Args:
            domain: Task domain to evaluate on
            num_tasks: Number of tasks to evaluate
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating agent on {num_tasks} {domain} tasks")
        
        eval_start = time.time()
        results = []
        
        # Generate and solve tasks
        for i in range(num_tasks):
            task = self._generate_single_task(domain)
            solution = self.solve_task(task)
            results.append(solution)
        
        # Calculate evaluation metrics
        successful = sum(1 for r in results if r.get("success", False))
        success_rate = successful / max(1, len(results))
        avg_score = sum(r.get("score", 0.0) for r in results) / max(1, len(results))
        avg_reward = sum(r.get("reward", 0.0) for r in results) / max(1, len(results))
        avg_time = sum(r.get("duration", 0.0) for r in results) / max(1, len(results))
        
        eval_time = time.time() - eval_start
        
        # Return evaluation summary
        return {
            "domain": domain,
            "total_tasks": len(results),
            "successful_tasks": successful,
            "success_rate": success_rate,
            "avg_score": avg_score,
            "avg_reward": avg_reward,
            "avg_time": avg_time,
            "eval_time": eval_time,
            "results": results
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get agent performance metrics.
        
        Returns:
            Performance metrics dictionary
        """
        metrics = self.performance_metrics.copy()
        
        # Calculate global metrics
        total_tasks = metrics["total_tasks"]
        if total_tasks > 0:
            metrics["success_rate"] = metrics["successful_tasks"] / total_tasks
            metrics["failure_rate"] = metrics["failed_tasks"] / total_tasks
        else:
            metrics["success_rate"] = 0.0
            metrics["failure_rate"] = 0.0
        
        # Add phase info
        metrics["current_phase"] = self.phase_controller.current_phase
        metrics["phase_weights"] = self.phase_controller.get_phase_weights()
        
        # Add step info
        metrics["global_step"] = self.global_step
        
        return metrics
    
    def save_state(self, path: str) -> bool:
        """
        Save agent state to disk.
        
        Args:
            path: Path to save state to
            
        Returns:
            True if successful
        """
        logger.info(f"Saving agent state to {path}")
        
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save metrics
            metrics_path = f"{path}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.get_performance_metrics(), f, indent=2)
            
            # Save history
            history_path = f"{path}_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.task_history, f, indent=2)
            
            # Save models if available
            if self.proposer_model and self.proposer_tokenizer:
                model_path = f"{path}_proposer"
                os.makedirs(model_path, exist_ok=True)
                self.proposer_model.save_pretrained(model_path)
                self.proposer_tokenizer.save_pretrained(model_path)
            
            logger.info(f"Agent state saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving agent state: {e}")
            return False
    
    def load_state(self, path: str) -> bool:
        """
        Load agent state from disk.
        
        Args:
            path: Path to load state from
            
        Returns:
            True if successful
        """
        logger.info(f"Loading agent state from {path}")
        
        try:
            # Load metrics
            metrics_path = f"{path}_metrics.json"
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    self.performance_metrics.update(metrics)
                    if "global_step" in metrics:
                        self.global_step = metrics["global_step"]
            
            # Load history
            history_path = f"{path}_history.json"
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.task_history = json.load(f)
            
            # Load models if available
            model_path = f"{path}_proposer"
            if os.path.exists(model_path):
                self.proposer_model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
                self.proposer_tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            logger.info(f"Agent state loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading agent state: {e}")
            return False
    
    def reset(self) -> None:
        """Reset agent state but keep models loaded."""
        logger.info("Resetting agent state")
        
        self.task_history = []
        self.performance_metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "avg_reward": 0.0,
            "domain_metrics": {}
        }
        self.global_step = 0
        
        # Reset reward system
        if hasattr(self.reward_system, "reset"):
            self.reward_system.reset()
        
        # Reset transfer adapter
        if hasattr(self.transfer_adapter, "reset"):
            self.transfer_adapter.reset()
        
        # Reset phase controller
        self.phase_controller.reset()