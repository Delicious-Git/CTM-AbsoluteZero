import numpy as np

class ImageClassifier:
    """
    Classificateur d'images pour le CTM.
    """
    def __init__(self, config):
        """
        Initialise le classificateur d'images.
        
        Args:
            config: Dictionnaire de configuration
        """
        self.config = config
    
    def classify(self, params):
        """
        Exécute une tâche de classification d'images.
        
        Args:
            params: Dictionnaire contenant:
                - difficulty: 'easy', 'medium', ou 'hard'
                - target_classes: Liste de classes à classifier
                - hierarchy_depth: Profondeur de la hiérarchie de classification
                - data_source: Source des données d'images
                
        Returns:
            Dictionnaire des métriques de performance
        """
        difficulty = params.get('difficulty', 'medium')
        target_classes = params.get('target_classes', [])
        hierarchy_depth = params.get('hierarchy_depth', 1)
        data_source = params.get('data_source', 'imagenet_subset')
        
        # Simulation de performance
        if difficulty == 'easy':
            base_accuracy = 0.9
        elif difficulty == 'medium':
            base_accuracy = 0.8
        else:  # hard
            base_accuracy = 0.6
            
        # Plus de classes = problème plus difficile
        class_penalty = min(0.3, len(target_classes) * 0.02)
        
        # Hiérarchies plus profondes sont plus difficiles
        hierarchy_penalty = min(0.4, hierarchy_depth * 0.1)
        
        accuracy = max(0.1, min(0.95, base_accuracy - class_penalty - hierarchy_penalty))
        
        # Simulation de métriques par classe
        class_metrics = {}
        for cls in target_classes[:5]:  # Limiter à 5 classes pour simplifier
            # Simuler des performances légèrement variables par classe
            class_accuracy = accuracy * np.random.uniform(0.8, 1.2)
            class_accuracy = max(0.1, min(0.98, class_accuracy))
            
            class_metrics[cls] = {
                'accuracy': round(class_accuracy, 2),
                'precision': round(max(0.1, class_accuracy * np.random.uniform(0.9, 1.1)), 2),
                'recall': round(max(0.1, class_accuracy * np.random.uniform(0.9, 1.1)), 2),
                'f1_score': round(max(0.1, class_accuracy * np.random.uniform(0.9, 1.1)), 2)
            }
        
        # Construire la matrice de confusion simulée (simplifiée)
        confusion_matrix = {}
        if len(target_classes) >= 2:
            for i, cls1 in enumerate(target_classes[:3]):
                confusion_matrix[cls1] = {}
                for j, cls2 in enumerate(target_classes[:3]):
                    if i == j:
                        # Diagonale (prédictions correctes)
                        confusion_matrix[cls1][cls2] = accuracy
                    else:
                        # Erreurs (prédictions incorrectes)
                        confusion_matrix[cls1][cls2] = (1 - accuracy) / (len(target_classes[:3]) - 1)
        
        return {
            'main_score': accuracy,
            'class_specific_accuracy': {cls: metrics['accuracy'] for cls, metrics in class_metrics.items()},
            'class_metrics': class_metrics,
            'confusion_matrix': confusion_matrix,
            'model_details': {
                'architecture': 'CTM-Vision',
                'parameters': 'auto',
                'training_epochs': int(10 + 20 * (1 - accuracy))
            },
            'details': f"Classifié images avec précision {accuracy:.2f} sur {len(target_classes)} classes"
        }
    
    def get_available_datasets(self):
        """
        Renvoie la liste des jeux de données disponibles.
        
        Returns:
            Dictionnaire des jeux de données et leurs descriptions
        """
        return {
            'imagenet_subset': 'Sous-ensemble d\'ImageNet avec 100 classes',
            'cifar10': 'CIFAR-10 avec 10 classes d\'objets communs',
            'mnist': 'Chiffres manuscrits (0-9)',
            'fashion_mnist': 'Articles vestimentaires (10 classes)',
            'qamnist': 'Version quantique de MNIST avec algorithmes d\'IA quantique'
        }