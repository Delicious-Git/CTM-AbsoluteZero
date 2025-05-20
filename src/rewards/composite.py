import numpy as np

class CompositeRewardSystem:
    """
    Système de récompense composite qui combine plusieurs composants de récompense
    avec des pondérations dynamiques.
    """
    def __init__(self, novelty_tracker, skill_pyramid, phase_controller, hyperparams=None):
        """
        Initialise le système de récompense composite.
        
        Args:
            novelty_tracker: Instance de SemanticNoveltyTracker
            skill_pyramid: Instance de SkillPyramid
            phase_controller: Instance de PhaseController
            hyperparams: Hyperparamètres optionnels
        """
        self.novelty_tracker = novelty_tracker
        self.skill_pyramid = skill_pyramid
        self.phase_controller = phase_controller
        
        # Hyperparamètres par défaut
        self.hyperparams = {
            "solve_success_threshold": 0.6,
            "learnability_target_success_rate": 0.5
        }
        
        if hyperparams:
            self.hyperparams.update(hyperparams)
    
    def calculate_reward(self, ctm_performance, task_params):
        """
        Calcule la récompense composite à partir de multiples composants.
        
        Args:
            ctm_performance: Dictionnaire des métriques de performance du CTM
            task_params: Paramètres de la tâche
            
        Returns:
            Récompense composite (score entre -1 et 1)
        """
        # Récompense de base (performance normalisée par domaine)
        r_solve = ctm_performance.get("main_score", 0.0)
        
        # Composante de nouveauté
        r_novelty = self.novelty_tracker.calculate_novelty(task_params)
        
        # Composante de progression
        r_progress = self.skill_pyramid.record_and_assess(
            task_params, 
            r_solve
        )
        
        # Composante d'apprentissage - basée sur l'écart avec le taux de réussite cible
        recent_scores = self.skill_pyramid.get_recent_scores(task_params['type'])
        success_rate = np.mean([
            s > self.hyperparams["solve_success_threshold"] 
            for s in recent_scores
        ]) if recent_scores else 0.5
        
        r_propose = 1.0 - abs(
            success_rate - self.hyperparams["learnability_target_success_rate"]
        )
        
        # Obtenir les poids dépendant de la phase
        weights = self.phase_controller.get_weights(task_params['type'])
        
        # Combinaison de toutes les composantes
        total_reward = (
            weights['solve'] * r_solve +
            weights['propose'] * r_propose +
            weights['novelty'] * r_novelty +
            weights['progress'] * r_progress
        )
        
        # Journal des récompenses (pour debug)
        print(f"  Récompenses: solve={r_solve:.2f}, propose={r_propose:.2f}, "
              f"novelty={r_novelty:.2f}, progress={r_progress:.2f}")
        
        # Écrêtage pour s'assurer que la récompense est dans [-1, 1]
        return float(np.clip(total_reward, -1.0, 1.0))
    
    def update_hyperparams(self, new_hyperparams):
        """
        Met à jour les hyperparamètres du système de récompense.
        
        Args:
            new_hyperparams: Dictionnaire de nouveaux hyperparamètres
        """
        self.hyperparams.update(new_hyperparams)