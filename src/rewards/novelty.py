import numpy as np
from collections import defaultdict

class SemanticNoveltyTracker:
    """
    Détecte les tâches véritablement nouvelles en utilisant des empreintes de paramètres.
    """
    def __init__(self):
        self.task_embeddings = defaultdict(list)
    
    def _params_to_features(self, params):
        """Convertit les paramètres de tâche en vecteur de caractéristiques"""
        features = []
        task_type = params['type']
        
        if task_type == 'maze':
            features.extend([
                params['params'].get('size_x', 10) / 20,
                params['params'].get('size_y', 10) / 20,
                params['params'].get('complexity', 0.5),
                int(params['params'].get('visual_patterns', False))
            ])
        elif task_type == 'parity':
            features.extend([
                params['params'].get('sequence_length', 10) / 100,
                params['params'].get('noise_level', 0),
                int(params['params'].get('recursive', False))
            ])
        elif task_type == 'quantum':
            features.extend([
                params['params'].get('num_qubits', 4) / 10,
                {'vqe': 0.1, 'grover': 0.5, 'qft': 0.9}.get(
                    params['params'].get('algorithm', 'vqe'), 0
                ),
                params['params'].get('noise_level', 0),
                params['params'].get('circuit_depth', 5) / 10
            ])
        else:
            # Extraction de caractéristiques par défaut
            features = [hash(str(v)) % 100 / 100 for k, v in 
                       params['params'].items()][:5]
        
        return np.array(features, dtype=np.float32)
    
    def calculate_novelty(self, task_params):
        """Calcule le score de nouveauté d'une tâche"""
        domain = task_params['type']
        features = self._params_to_features(task_params)
        
        if len(self.task_embeddings[domain]) < 2:
            self.task_embeddings[domain].append(features)
            return 1.0  # Première tâche = maximalement nouvelle
        
        # Calcul de similarité cosinus avec les tâches précédentes
        similarities = []
        for old_features in self.task_embeddings[domain]:
            if len(old_features) == len(features):
                # Calcul de similarité cosinus
                dot_product = np.dot(old_features, features)
                norm_a = np.linalg.norm(old_features)
                norm_b = np.linalg.norm(features)
                similarity = dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
                similarities.append(similarity)
        
        max_similarity = max(similarities) if similarities else 0
        novelty = 1.0 - max_similarity
        
        # Stockage de l'embedding
        self.task_embeddings[domain].append(features)
        
        # Bonus binaire pour les tâches vraiment nouvelles
        threshold = 0.3 if domain in ['quantum', 'rl'] else 0.2
        return float(novelty > threshold)