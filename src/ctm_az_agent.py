import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig
import numpy as np
import json
import time
from collections import defaultdict, deque

from .rewards.novelty import SemanticNoveltyTracker
from .rewards.progress import SkillPyramid
from .rewards.composite import CompositeRewardSystem
from .transfer.adapter import NeuralTransferAdapter
from .transfer.phase import PhaseController
from .ctm.interface import RealCTMInterface

class CTM_AbsoluteZero_Agent:
    """
    Agent principal intégrant le paradigme Absolute Zero Reasoner avec CTM.
    Utilise une architecture Proposer/Solver où un LLM (Proposer) génère des
    tâches que le CTM (Solver) exécute, avec un système de récompense avancé.
    """
    def __init__(self, ctm_solver_config, proposer_llm_name="mistralai/Mistral-7B-Instruct-v0.1", 
                 az_hyperparams=None, ppo_config_dict=None):
        """
        Initialise l'agent CTM-AbsoluteZero.
        
        Args:
            ctm_solver_config: Configuration du CTM Solver
            proposer_llm_name: Nom du modèle LLM à utiliser comme Proposer
            az_hyperparams: Hyperparamètres pour Absolute Zero
            ppo_config_dict: Configuration pour l'entraînement PPO
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation du dispositif: {self.device}")

        # 1. Initialiser CTM (Solver)
        self.ctm_solver = RealCTMInterface(ctm_solver_config)

        # 2. Initialiser LLM (Proposer)
        self._initialize_proposer(proposer_llm_name)

        # 3. Initialiser PPO Trainer
        self._setup_ppo_trainer(ppo_config_dict)

        # 4. Initialiser les composants de récompense
        self._initialize_reward_components()

        # 5. Hyperparamètres Absolute Zero
        self._setup_hyperparams(az_hyperparams)
        
        # 6. Suivi d'entraînement
        self.global_step = 0
        self.phase_controller = PhaseController()
        
        # Pour la collecte de batch PPO
        self.batch_proposer_query_texts = []
        self.batch_proposer_response_texts = []
        self.batch_solve_rewards = []

    def _initialize_proposer(self, model_name):
        """
        Initialise le modèle Proposer.
        
        Args:
            model_name: Nom du modèle Hugging Face à charger
        """
        self.proposer_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.proposer_model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        if self.proposer_tokenizer.pad_token is None:
            self.proposer_tokenizer.pad_token = self.proposer_tokenizer.eos_token

    def _setup_ppo_trainer(self, config_dict):
        """
        Configure le trainer PPO pour l'entraînement RL du Proposer.
        
        Args:
            config_dict: Dictionnaire de configuration PPO optionnel
        """
        default_ppo_config = {
            "learning_rate": 3e-6,
            "batch_size": 32,
            "mini_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "ppo_epochs": 4,
            "clip_range": 0.15,
            "gamma": 0.99,
            "lam": 0.95,
            "entropy_coef": 0.01,
            "target_kl": 0.015,
            "init_kl_coef": 0.15
        }
        if config_dict: 
            default_ppo_config.update(config_dict)
        ppo_config_obj = PPOConfig(**default_ppo_config)
        
        self.ppo_trainer = PPOTrainer(
            config=ppo_config_obj, 
            model=self.proposer_model,
            ref_model=None, 
            tokenizer=self.proposer_tokenizer
        )

    def _initialize_reward_components(self):
        """Initialise les composants de récompense avancés"""
        self.novelty_tracker = SemanticNoveltyTracker()
        self.progress_monitor = SkillPyramid()
        self.cross_domain_adapter = NeuralTransferAdapter([
            'maze', 'image_classification', 'sorting', 'parity', 
            'rl', 'qamnist', 'quantum'
        ])

    def _setup_hyperparams(self, az_hyperparams):
        """
        Configure les hyperparamètres du système.
        
        Args:
            az_hyperparams: Dictionnaire d'hyperparamètres optionnel
        """
        default_az_hparams = {
            "w_solve": 0.6,
            "w_propose": 0.2,
            "w_novelty": 0.1,
            "w_progress": 0.1,
            "solve_success_threshold": 0.6,
            "learnability_target_success_rate": 0.5
        }
        if az_hyperparams: 
            default_az_hparams.update(az_hyperparams)
        self.az_hparams = default_az_hparams
        
        # Initialiser le système de récompense composite
        self.reward_system = CompositeRewardSystem(
            self.novelty_tracker,
            self.progress_monitor,
            self.phase_controller,
            self.az_hparams
        )

    def _generate_proposer_prompt(self, ctm_domain, feedback=""):
        """
        Génère un prompt pour le Proposer LLM.
        
        Args:
            ctm_domain: Domaine CTM pour lequel générer une tâche
            feedback: Feedback optionnel sur les performances précédentes
            
        Returns:
            Texte du prompt formaté
        """
        # Exemples spécifiques par domaine avec contraintes
        examples = {
            "maze": {
                "type": "maze", 
                "params": {
                    "size_x": 12, "size_y": 12, 
                    "complexity": 0.8, 
                    "seed": 42,
                    "visual_patterns": False
                }
            },
            "image_classification": {
                "type": "image_classification", 
                "params": {
                    "target_classes": ["cat", "dog"], 
                    "difficulty": "hard", 
                    "data_source": "imagenet_subset_xyz",
                    "hierarchy_depth": 1
                }
            },
            "sorting": {
                "type": "sorting", 
                "params": {
                    "list_length": 50, 
                    "value_range": [0, 1000], 
                    "list_type": "almost_sorted_descending"
                }
            },
            "parity": {
                "type": "parity", 
                "params": {
                    "sequence_length": 32, 
                    "noise_level": 0.2,
                    "recursive": False
                }
            },
            "quantum": {
                "type": "quantum",
                "params": {
                    "algorithm": "vqe",  # ou "grover", "qft"
                    "num_qubits": 5,
                    "noise_level": 0.02,
                    "circuit_depth": 6
                }
            }
        }
        
        # Construire le prompt avec des exemples spécifiques au domaine
        prompt = f"<|system|>Vous êtes un générateur de tâches IA pour la Continuous Thought Machine (CTM).\n"
        prompt += f"Domaine CTM: {ctm_domain}\n"
        
        if feedback: 
            prompt += f"Retour de performance: {feedback}\n"
        
        prompt += (f"Générez une tâche spécifique pour CTM au format JSON. Les tâches doivent être stimulantes "
                  f"mais réalisables, visant environ 50% de taux de réussite.\n\n")
        
        # Ajouter un exemple spécifique au domaine
        if ctm_domain in examples:
            prompt += f"Format JSON d'exemple pour {ctm_domain}:\n"
            prompt += json.dumps(examples[ctm_domain], indent=2) + "\n\n"
        
        # Ajouter des contraintes spécifiques au domaine
        if ctm_domain == "maze":
            prompt += "Contraintes: size_x et size_y doivent être entre 5 et 20. Complexity entre 0.1 et 0.9.\n"
        elif ctm_domain == "quantum":
            prompt += "Contraintes: num_qubits doit être entre 2 et 10. Algorithm doit être l'un des suivants: vqe, grover, qft.\n"
        
        prompt += "<|user|>Générez une nouvelle tâche:\n<|assistant|>"
        return prompt

    def propose_task(self, ctm_domain, feedback=""):
        """
        Génère une tâche avec le Proposer LLM.
        
        Args:
            ctm_domain: Domaine CTM pour lequel générer une tâche
            feedback: Feedback optionnel sur les performances précédentes
            
        Returns:
            Dictionnaire des paramètres de tâche générés
        """
        prompt_text = self._generate_proposer_prompt(ctm_domain, feedback)
        
        # Tokeniser et générer
        inputs = self.proposer_tokenizer(prompt_text, return_tensors="pt").to(self.device)
        
        response_ids = self.proposer_model.generate(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask,
            max_new_tokens=200, 
            pad_token_id=self.proposer_tokenizer.pad_token_id,
            do_sample=True, 
            top_p=0.9, 
            temperature=0.7
        )
        
        # Extraire uniquement la partie générée
        task_description_str = self.proposer_tokenizer.decode(
            response_ids[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )

        # Stocker prompt et réponse pour PPO
        self.batch_proposer_query_texts.append(prompt_text)
        self.batch_proposer_response_texts.append(task_description_str)

        # Parser le JSON généré
        try:
            # Trouver le contenu JSON
            start_idx = task_description_str.find('{')
            end_idx = task_description_str.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = task_description_str[start_idx:end_idx+1]
                parsed_task = json.loads(json_str)
                
                # Validation de base
                if not isinstance(parsed_task.get("params"), dict):
                    parsed_task["params"] = {}
                
                # S'assurer que le champ type est présent
                if "type" not in parsed_task:
                    parsed_task["type"] = ctm_domain
            else:
                print(f"Avertissement: Aucun objet JSON valide trouvé dans la sortie LLM: '{task_description_str}'")
                parsed_task = {"type": "parse_error", "raw_output": task_description_str, "params": {}}
        except json.JSONDecodeError as e:
            print(f"Erreur de parsing JSON: {e}. Raw: '{task_description_str}'")
            parsed_task = {"type": "parse_error", "raw_output": task_description_str, "error_msg": str(e), "params": {}}
        
        return parsed_task

    def validate_task(self, task_params):
        """
        Vérifie que les paramètres de tâche générés sont valides.
        
        Args:
            task_params: Paramètres de tâche à valider
            
        Returns:
            Booléen indiquant si la tâche est valide
        """
        if task_params['type'] == 'parse_error':
            return False
            
        if task_params['type'] == 'maze':
            # Valider les paramètres de labyrinthe
            size_x = task_params['params'].get('size_x', 10)
            size_y = task_params['params'].get('size_y', 10)
            complexity = task_params['params'].get('complexity', 0.5)
            
            return 5 <= size_x <= 20 and 5 <= size_y <= 20 and 0.1 <= complexity <= 0.9
            
        elif task_params['type'] == 'quantum':
            # Valider les paramètres quantiques
            num_qubits = task_params['params'].get('num_qubits', 4)
            algorithm = task_params['params'].get('algorithm', '')
            
            return (2 <= num_qubits <= 10 and 
                    algorithm in ['vqe', 'grover', 'qft'])
                    
        # Cas par défaut - supposer valide
        return True

    def solve_task_with_ctm(self, task_params):
        """
        Exécute une tâche avec le CTM Solver.
        
        Args:
            task_params: Paramètres de la tâche à exécuter
            
        Returns:
            Métriques de performance du CTM
        """
        if task_params.get("type") == "parse_error":
            return {"main_score": 0.0, "details": "Tâche non analysable par CTM en raison d'erreur de formatage."}
        
        # Appeler la logique d'exécution/évaluation CTM réelle
        performance_metrics = self.ctm_solver.execute_task(task_params)
        return performance_metrics

    def calculate_reward(self, ctm_performance, task_params):
        """
        Calcule la récompense composite à partir de multiples composants.
        
        Args:
            ctm_performance: Métriques de performance du CTM
            task_params: Paramètres de la tâche
            
        Returns:
            Score de récompense composite
        """
        return self.reward_system.calculate_reward(ctm_performance, task_params)

    def run_self_play_step(self, ctm_domain, feedback_for_proposer=""):
        """
        Exécute une étape d'auto-apprentissage complète.
        
        Args:
            ctm_domain: Domaine CTM à utiliser pour cette étape
            feedback_for_proposer: Feedback optionnel pour guider le Proposer
            
        Returns:
            Tuple (récompense, paramètres de tâche)
        """
        # 1. Générer une tâche
        task_params = self.propose_task(ctm_domain, feedback_for_proposer)
        
        # 2. Valider la tâche
        valid_task = self.validate_task(task_params)
        attempts = 1
        
        # Retry si invalide (jusqu'à 3 tentatives)
        while not valid_task and attempts < 3:
            print(f"  Validation de tâche échouée, tentative {attempts}/3")
            task_params = self.propose_task(
                ctm_domain, 
                f"{feedback_for_proposer} La tâche précédente était invalide."
            )
            valid_task = self.validate_task(task_params)
            attempts += 1
        
        # 3. Exécuter la tâche avec CTM
        if valid_task:
            # Appliquer le transfert entre domaines
            if self.global_step > 5000:  # Seulement après la phase d'exploration
                all_domains_perf = self.progress_monitor.get_all_domain_metrics()
                task_params = self.cross_domain_adapter.transfer_parameters(
                    ctm_domain, 
                    task_params,
                    all_domains_perf
                )
                
            ctm_performance = self.solve_task_with_ctm(task_params)
        else:
            ctm_performance = {"main_score": 0.0, "details": "Tâche invalide après 3 tentatives"}
        
        # 4. Calculer la récompense composite
        reward = self.calculate_reward(ctm_performance, task_params)
        self.batch_solve_rewards.append(reward)
        
        # 5. Mettre à jour le contrôleur de phase
        if self.global_step % 100 == 0:
            performance_metrics = {
                'success_rate': self.progress_monitor.get_global_success_rate(),
                'cross_domain_corr': self.cross_domain_adapter.get_correlation()
            }
            self.phase_controller.update_phase(performance_metrics)
        
        self.global_step += 1
        
        print(f"Étape {self.global_step}: Domaine: {ctm_domain}, "
              f"Tâche: {task_params.get('type')}, Récompense: {reward:.3f}")
        
        return reward, task_params

    def update_proposer_via_ppo(self):
        """
        Met à jour le Proposer via PPO en utilisant les expériences collectées.
        
        Returns:
            Booléen indiquant si la mise à jour a réussi
        """
        if len(self.batch_proposer_query_texts) < self.ppo_trainer.config.batch_size:
            # Pas assez de données pour la mise à jour PPO
            return False

        print(f"Début de la mise à jour PPO avec {len(self.batch_proposer_query_texts)} expériences...")
        
        # Blanchiment des récompenses (normalisation)
        rewards = np.array(self.batch_solve_rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Tokeniser les entrées pour PPO
        query_tensors = [
            self.proposer_tokenizer(
                q_text, return_tensors="pt", truncation=True
            ).input_ids.squeeze(0).to(self.device)
            for q_text in self.batch_proposer_query_texts
        ]
        
        response_tensors = [
            self.proposer_tokenizer(
                r_text, return_tensors="pt", truncation=True
            ).input_ids.squeeze(0).to(self.device)
            for r_text in self.batch_proposer_response_texts
        ]
        
        # Convertir les récompenses en tenseurs
        ppo_rewards = torch.tensor(rewards, device=self.device)
        
        # Exécuter une étape PPO
        stats = self.ppo_trainer.step(
            query_tensors, response_tensors, list(ppo_rewards)
        )
        print(f"  Statistiques de mise à jour PPO: {stats}")

        # Vider les batchs
        self.batch_proposer_query_texts = []
        self.batch_proposer_response_texts = []
        self.batch_solve_rewards = []
        
        return True

    def train(self, steps=10000, domains=None):
        """
        Exécute la boucle complète d'entraînement.
        
        Args:
            steps: Nombre total d'étapes d'entraînement
            domains: Liste des domaines à utiliser, si None, utilise les domaines par défaut
        """
        if domains is None:
            domains = ["maze", "sorting", "quantum", "image_classification"]
        
        steps_per_ppo_update = self.ppo_trainer.config.batch_size * 2
        
        for step in range(1, steps + 1):
            # Sélection du domaine avec curriculum
            if step < steps // 10:
                # Focus sur les domaines de base au début
                selected_domain = np.random.choice(domains[:2])
            else:
                # Étendre à tous les domaines
                selected_domain = np.random.choice(domains)
            
            # Exécuter une étape d'auto-apprentissage
            reward, task = self.run_self_play_step(selected_domain)
            
            # Mise à jour du Proposer
            if len(self.batch_proposer_query_texts) >= steps_per_ppo_update:
                self.update_proposer_via_ppo()
                
                # Sauvegarde périodique
                if step % (steps_per_ppo_update * 5) == 0:
                    self.save_checkpoint(f"checkpoint_step_{step}")
            
            # Journalisation
            if step % 10 == 0:
                print(f"Étape {step}/{steps}: Domaine={selected_domain}, Récompense={reward:.3f}")
        
        # Sauvegarde finale
        self.save_checkpoint("final_model")
        print("Entraînement terminé!")
    
    def save_checkpoint(self, name):
        """
        Sauvegarde un checkpoint du modèle Proposer.
        
        Args:
            name: Nom du checkpoint
        """
        import os
        os.makedirs("./checkpoints", exist_ok=True)
        
        save_path = f"./checkpoints/{name}"
        self.proposer_model.save_pretrained(save_path)
        self.proposer_tokenizer.save_pretrained(save_path)
        print(f"Checkpoint sauvegardé: {save_path}")
        
    def load_checkpoint(self, path):
        """
        Charge un checkpoint du modèle Proposer.
        
        Args:
            path: Chemin du checkpoint
        """
        self.proposer_model = AutoModelForCausalLM.from_pretrained(path).to(self.device)
        self.proposer_tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Recréer le PPO Trainer avec le modèle chargé
        self._setup_ppo_trainer(None)
        print(f"Checkpoint chargé depuis: {path}")
        
if __name__ == "__main__":
    # Exemple d'utilisation simple
    ctm_config = {
        "model_path": "models/ctm_base",
        "device": "cuda",
        "precision": "float16"
    }
    
    agent = CTM_AbsoluteZero_Agent(ctm_solver_config=ctm_config)
    
    # Exécuter quelques étapes de test
    for domain in ["maze", "quantum"]:
        reward, task = agent.run_self_play_step(domain)
        print(f"Domaine: {domain}, Récompense: {reward:.3f}")
        print(f"Tâche: {task}")
        print("-" * 40)