import numpy as np
from collections import defaultdict
import time
from scipy.stats import linregress

class SkillPyramid:
    """
    Suit la progression hiérarchique des compétences dans différents domaines et 
    niveaux de difficulté.
    """
    def __init__(self):
        self.performance_history = defaultdict(lambda: defaultdict(list))
        self.domain_clusters = {
            'maze': {
                'size': [5, 10, 15, 20],
                'complexity': [0.3, 0.6, 0.9]
            },
            'sorting': {
                'length': [10, 50, 100, 500],
                'type': ['random', 'almost_sorted', 'reverse_sorted']
            },
            'quantum': {
                'qubits': [2, 5, 8, 10],
                'algorithm': ['vqe', 'grover', 'qft']
            }
        }
    
    def _get_task_cluster(self, task_params):
        """
        Identifie le cluster auquel appartient une tâche en fonction de ses paramètres.
        
        Args:
            task_params: Dictionnaire contenant les paramètres de la tâche
        
        Returns:
            Identifiant de cluster sous forme de chaîne
        """
        domain = task_params['type']
        
        if domain == 'maze':
            size_x = task_params['params'].get('size_x', 10)
            size_y = task_params['params'].get('size_y', 10)
            avg_size = (size_x + size_y) / 2
            size_level = next((s for s in self.domain_clusters['maze']['size'] 
                              if avg_size <= s), 20)
            
            complexity = task_params['params'].get('complexity', 0.5)
            complexity_level = next((c for c in self.domain_clusters['maze']['complexity'] 
                                   if complexity <= c), 0.9)
            
            return f"maze_s{size_level}_c{complexity_level}"
            
        elif domain == 'quantum':
            qubits = task_params['params'].get('num_qubits', 4)
            qubit_level = next((q for q in self.domain_clusters['quantum']['qubits'] 
                              if qubits <= q), 10)
            
            algorithm = task_params['params'].get('algorithm', 'vqe')
            
            return f"quantum_q{qubit_level}_{algorithm}"
        
        # Clustering par défaut pour les autres domaines
        return f"{domain}_default"
    
    def record_and_assess(self, task_params, score):
        """
        Enregistre une performance et évalue la progression.
        
        Args:
            task_params: Paramètres de la tâche
            score: Score de performance (0-1)
            
        Returns:
            Métrique de progression (0-1)
        """
        cluster = self._get_task_cluster(task_params)
        domain = task_params['type']
        
        # Enregistrement de la performance
        self.performance_history[domain]['scores'].append(score)
        self.performance_history[cluster]['scores'].append(score)
        self.performance_history[cluster]['timestamps'].append(time.time())
        
        # Calcul des métriques de progression
        window = min(10, len(self.performance_history[cluster]['scores']))
        recent_scores = self.performance_history[cluster]['scores'][-window:]
        
        if len(recent_scores) < 3:
            return 0.0
        
        # Calcul de la pente des performances récentes
        x = np.arange(len(recent_scores))
        slope, _, _, _, _ = linregress(x, recent_scores)
        
        # Fonction sigmoïde pour normaliser la récompense de progression
        progress = 1 / (1 + np.exp(-10 * slope))
        
        return progress
    
    def get_recent_scores(self, domain, window=20):
        """
        Récupère les scores récents pour un domaine.
        
        Args:
            domain: Type de domaine
            window: Taille de la fenêtre de récupération
            
        Returns:
            Liste des scores récents
        """
        scores = self.performance_history[domain]['scores']
        return scores[-window:] if len(scores) > 0 else []
    
    def get_global_success_rate(self, window=100):
        """
        Calcule le taux de réussite global sur tous les domaines.
        
        Args:
            window: Taille de la fenêtre pour le calcul
            
        Returns:
            Liste des scores récents tous domaines confondus
        """
        all_scores = []
        for domain, data in self.performance_history.items():
            if domain.find('_') == -1:  # Seulement les domaines principaux
                all_scores.extend(data['scores'][-window:])
        
        return all_scores[-window:] if all_scores else []
    
    def get_all_domain_metrics(self):
        """
        Récupère les métriques de performance pour tous les domaines.
        
        Returns:
            Dictionnaire des métriques par domaine
        """
        metrics = {}
        for domain, data in self.performance_history.items():
            if domain.find('_') == -1:  # Seulement les domaines principaux
                scores = data['scores'][-20:] if len(data['scores']) > 0 else []
                if scores:
                    metrics[domain] = {
                        'avg': np.mean(scores),
                        'trend': np.mean(scores[-5:]) - np.mean(scores[-10:-5]) if len(scores) >= 10 else 0,
                        'var': np.var(scores) if len(scores) > 1 else 0
                    }
        return metrics