import numpy as np
import time

class Sorter:
    """
    Module de tri pour le CTM.
    """
    def __init__(self, config):
        """
        Initialise le module de tri.
        
        Args:
            config: Dictionnaire de configuration
        """
        self.config = config
    
    def sort(self, params):
        """
        Exécute une tâche de tri.
        
        Args:
            params: Dictionnaire contenant:
                - list_length: Longueur de la liste à trier
                - value_range: Plage de valeurs [min, max]
                - list_type: Type de liste ('random', 'almost_sorted', etc.)
                
        Returns:
            Dictionnaire des métriques de performance
        """
        list_length = params.get('list_length', 50)
        value_range = params.get('value_range', [0, 1000])
        list_type = params.get('list_type', 'random')
        
        # Générer une liste conforme aux paramètres
        data = self._generate_list(list_length, value_range, list_type)
        
        # Simuler la performance du tri
        start_time = time.time()
        # Dans une implémentation réelle, on trierait réellement la liste
        # Ici on simule juste le temps et la performance
        
        # Simuler performance basée sur paramètres
        base_score = 0.9  # Précision de base
        length_penalty = min(0.4, list_length / 1000)
        
        if list_type == 'almost_sorted':
            type_bonus = 0.1
            comparisons = int(0.3 * list_length * np.log2(list_length))
        elif list_type == 'reverse_sorted':
            type_bonus = 0.05
            comparisons = int(1.5 * list_length * np.log2(list_length))
        else:  # random
            type_bonus = 0
            comparisons = int(list_length * np.log2(list_length))
            
        score = max(0.1, min(0.95, base_score - length_penalty + type_bonus))
        
        # Simuler le temps (proportionnel aux comparaisons)
        elapsed_time = comparisons * 0.0001  # Temps simulé
        
        # Collecter les résultats
        return {
            'main_score': score,
            'time_complexity': f"O({list_length}log{list_length})",
            'comparisons': comparisons,
            'execution_time': elapsed_time,
            'memory_usage': list_length * 8,  # Octets approximatifs
            'details': f"Trié {list_length} éléments de type {list_type}"
        }
        
    def _generate_list(self, length, value_range, list_type):
        """
        Génère une liste selon les paramètres spécifiés.
        
        Args:
            length: Longueur de la liste
            value_range: [min, max] des valeurs
            list_type: Type de liste à générer
            
        Returns:
            Liste générée selon les paramètres
        """
        min_val, max_val = value_range
        
        if list_type == 'random':
            # Liste complètement aléatoire
            return np.random.randint(min_val, max_val + 1, size=length).tolist()
            
        elif list_type == 'almost_sorted':
            # Liste presque triée (avec environ 10% d'éléments déplacés)
            data = np.sort(np.random.randint(min_val, max_val + 1, size=length))
            swaps = max(1, int(length * 0.1))
            
            for _ in range(swaps):
                i, j = np.random.choice(length, 2, replace=False)
                data[i], data[j] = data[j], data[i]
                
            return data.tolist()
            
        elif list_type == 'reverse_sorted':
            # Liste triée en ordre décroissant
            return np.sort(np.random.randint(min_val, max_val + 1, size=length))[::-1].tolist()
            
        elif list_type == 'few_unique':
            # Liste avec peu de valeurs uniques
            unique_vals = max(2, int(length * 0.1))
            values = np.random.randint(min_val, max_val + 1, size=unique_vals)
            return np.random.choice(values, size=length).tolist()
            
        else:
            # Par défaut: liste aléatoire
            return np.random.randint(min_val, max_val + 1, size=length).tolist()
    
    def get_available_algorithms(self):
        """
        Renvoie la liste des algorithmes de tri disponibles.
        
        Returns:
            Dictionnaire des algorithmes et leurs complexités
        """
        return {
            'quick_sort': {
                'avg_time': 'O(n log n)',
                'worst_time': 'O(n²)',
                'space': 'O(log n)'
            },
            'merge_sort': {
                'avg_time': 'O(n log n)',
                'worst_time': 'O(n log n)',
                'space': 'O(n)'
            },
            'heap_sort': {
                'avg_time': 'O(n log n)',
                'worst_time': 'O(n log n)',
                'space': 'O(1)'
            },
            'insertion_sort': {
                'avg_time': 'O(n²)',
                'worst_time': 'O(n²)',
                'space': 'O(1)'
            },
            'bubble_sort': {
                'avg_time': 'O(n²)',
                'worst_time': 'O(n²)',
                'space': 'O(1)'
            },
            'quantum_sort': {
                'avg_time': 'O(n log n)',  # Simulé
                'worst_time': 'O(n log n)',
                'space': 'O(n)'
            }
        }