import numpy as np

class MazeSolver:
    """
    Résolveur de labyrinthe pour le CTM.
    """
    def __init__(self, config):
        """
        Initialise le résolveur de labyrinthe.
        
        Args:
            config: Dictionnaire de configuration
        """
        self.config = config
    
    def solve(self, params):
        """
        Résout un labyrinthe avec les paramètres donnés.
        
        Args:
            params: Paramètres du labyrinthe
                - size_x: Largeur du labyrinthe
                - size_y: Hauteur du labyrinthe
                - complexity: Complexité du labyrinthe (0-1)
                - visual_patterns: Utilisation de motifs visuels
                
        Returns:
            Dictionnaire des métriques de performance
        """
        size_x = params.get('size_x', 10)
        size_y = params.get('size_y', 10)
        complexity = params.get('complexity', 0.5)
        visual_patterns = params.get('visual_patterns', False)
        
        # Vérification de la validité des paramètres
        size_x = max(5, min(size_x, 20))
        size_y = max(5, min(size_y, 20))
        complexity = max(0.1, min(complexity, 0.9))
        
        # Dans une implémentation réelle, nous générerions et résoudrions 
        # réellement un labyrinthe ici
        
        # Simulation de performance
        size_factor = 1.0 - (size_x * size_y) / (20 * 20)
        visual_bonus = 0.1 if visual_patterns else 0.0
        success_prob = 0.95 * size_factor - complexity * 0.3 + visual_bonus
        success = max(0.1, min(0.95, success_prob))
        
        return {
            'main_score': success,
            'path_length': int(size_x * size_y * (0.3 + 0.7 * complexity)),
            'completion_time': float(size_x * size_y * complexity * 0.01),
            'explored_cells_ratio': float(0.7 + 0.3 * (1 - complexity)),
            'details': f"Résolu labyrinthe {size_x}x{size_y} avec complexité {complexity}"
        }
    
    def generate_maze(self, size_x, size_y, complexity):
        """
        Génère une représentation de labyrinthe.
        
        Args:
            size_x: Largeur du labyrinthe
            size_y: Hauteur du labyrinthe
            complexity: Complexité du labyrinthe (0-1)
            
        Returns:
            Matrice numpy représentant le labyrinthe (0=passage, 1=mur)
        """
        # Assurer des dimensions impaires pour un labyrinthe standard
        size_x = size_x if size_x % 2 == 1 else size_x + 1
        size_y = size_y if size_y % 2 == 1 else size_y + 1
        
        # Créer une grille remplie de murs
        maze = np.ones((size_y, size_x), dtype=np.int8)
        
        # Définir points de départ et d'arrivée
        maze[1, 1] = 0  # Départ (coin supérieur gauche)
        maze[size_y-2, size_x-2] = 0  # Arrivée (coin inférieur droit)
        
        # Simulation simplifiée de génération de labyrinthe
        # Dans une implémentation réelle, on utiliserait un algorithme comme 
        # Recursive Backtracking, Prim ou Kruskal
        
        # Simuler un labyrinthe avec une complexité donnée
        wall_removal_probability = 1.0 - complexity
        
        for y in range(1, size_y-1, 2):
            for x in range(1, size_x-1, 2):
                maze[y, x] = 0  # Cellule
                
                # Créer des passages basés sur la complexité
                if x < size_x-2 and np.random.random() < wall_removal_probability:
                    maze[y, x+1] = 0  # Passage horizontal
                if y < size_y-2 and np.random.random() < wall_removal_probability:
                    maze[y+1, x] = 0  # Passage vertical
        
        return maze