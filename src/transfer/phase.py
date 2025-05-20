import time
import numpy as np

class PhaseController:
    """
    Contrôleur de phases d'entraînement qui ajuste dynamiquement les récompenses.
    """
    PHASES = {
        'exploration': {
            'solve': 0.4,
            'propose': 0.2,
            'novelty': 0.3,
            'progress': 0.1
        },
        'specialization': {
            'solve': 0.5,
            'propose': 0.3,
            'novelty': 0.1,
            'progress': 0.1
        },
        'transfer': {
            'solve': 0.4,
            'propose': 0.2,
            'novelty': 0.1,
            'progress': 0.3
        },
        'refinement': {
            'solve': 0.6,
            'propose': 0.1,
            'novelty': 0.0,
            'progress': 0.3
        }
    }
    
    def __init__(self):
        """Initialise le contrôleur de phase"""
        self.current_phase = 'exploration'
        self.phase_history = []
    
    def update_phase(self, performance_metrics):
        """
        Met à jour la phase d'entraînement en fonction des métriques de performance.
        
        Args:
            performance_metrics: Dictionnaire de métriques de performance
        """
        # Logique de transition de phase
        if self.current_phase == 'exploration':
            success_rate = np.mean(performance_metrics['success_rate'])
            if success_rate > 0.65:
                self.current_phase = 'specialization'
                print("Transition vers la phase de SPÉCIALISATION")
        
        elif self.current_phase == 'specialization':
            cross_domain = performance_metrics['cross_domain_corr']
            if cross_domain > 0.6:
                self.current_phase = 'transfer'
                print("Transition vers la phase de TRANSFERT")
        
        elif self.current_phase == 'transfer':
            success_rate = np.mean(performance_metrics['success_rate'])
            if success_rate > 0.8:
                self.current_phase = 'refinement'
                print("Transition vers la phase de RAFFINEMENT")
        
        self.phase_history.append({
            'phase': self.current_phase,
            'timestamp': time.time()
        })
    
    def get_weights(self, domain=None):
        """
        Récupère les poids de récompense pour la phase actuelle.
        
        Args:
            domain: Domaine pour lequel récupérer les poids (optionnel)
            
        Returns:
            Dictionnaire des poids de récompense
        """
        weights = self.PHASES[self.current_phase].copy()
        
        # Ajustements spécifiques par domaine (si nécessaire)
        if domain == 'quantum' and self.current_phase == 'exploration':
            weights['novelty'] = 0.4  # Plus d'exploration pour les tâches quantiques
        
        return weights
    
    def get_phase_duration(self, phase_name=None):
        """
        Calcule la durée d'une phase spécifique.
        
        Args:
            phase_name: Nom de la phase (utilise la phase actuelle si None)
            
        Returns:
            Durée de la phase en secondes
        """
        if phase_name is None:
            phase_name = self.current_phase
        
        # Filtrer l'historique pour la phase demandée
        phase_records = [record for record in self.phase_history 
                          if record['phase'] == phase_name]
        
        if not phase_records:
            return 0
        
        if phase_name == self.current_phase:
            # Phase en cours, calculer jusqu'à maintenant
            return time.time() - phase_records[0]['timestamp']
        else:
            # Phase terminée, calculer du début à la fin
            return phase_records[-1]['timestamp'] - phase_records[0]['timestamp']
    
    def get_phase_summary(self):
        """
        Génère un résumé des phases d'entraînement.
        
        Returns:
            Dictionnaire avec résumé des phases
        """
        unique_phases = sorted(set(record['phase'] for record in self.phase_history))
        
        summary = {}
        for phase in unique_phases:
            summary[phase] = {
                'duration': self.get_phase_duration(phase),
                'is_current': phase == self.current_phase,
                'weights': self.PHASES[phase]
            }
        
        return summary