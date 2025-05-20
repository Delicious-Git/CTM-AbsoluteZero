import numpy as np

class QuantumSimulator:
    """
    Simulateur quantique léger pour les algorithmes de base.
    """
    def __init__(self, config=None):
        """
        Initialise le simulateur quantique.
        
        Args:
            config: Configuration optionnelle du simulateur
        """
        self.config = config or {}
    
    def run(self, params):
        """
        Exécute une simulation quantique basée sur les paramètres.
        
        Args:
            params: Dictionnaire contenant:
                - algorithm: 'vqe', 'grover', ou 'qft'
                - num_qubits: Nombre de qubits
                - noise_level: Niveau de bruit (0-1)
                - circuit_depth: Profondeur du circuit
                
        Returns:
            Dictionnaire avec les métriques de performance
        """
        algorithm = params.get('algorithm', 'vqe')
        num_qubits = params.get('num_qubits', 4)
        noise_level = params.get('noise_level', 0.01)
        circuit_depth = params.get('circuit_depth', 5)
        
        # Vérification des paramètres
        num_qubits = max(2, min(num_qubits, 10))
        noise_level = max(0, min(noise_level, 0.5))
        circuit_depth = max(1, min(circuit_depth, 20))
        
        # Simulation de performance basée sur l'algorithme et les paramètres
        if algorithm == 'vqe':
            # VQE diminue avec le nombre de qubits et le bruit
            vqe_result = self.simulate_vqe(num_qubits, noise_level, circuit_depth)
            base_score = 0.9
            qubit_penalty = num_qubits * 0.05
            noise_impact = noise_level * 10
            depth_impact = min(0.3, circuit_depth * 0.03)
            success = max(0.1, min(0.95, base_score - qubit_penalty - noise_impact - depth_impact))
            
            # Métriques spécifiques à VQE
            extras = {
                'energy_accuracy': vqe_result['success_metric'],
                'iterations': len(vqe_result['convergence']),
                'energy_convergence': vqe_result['convergence'][-5:] if len(vqe_result['convergence']) >= 5 else vqe_result['convergence']
            }
            
        elif algorithm == 'grover':
            # Grover s'améliore avec les qubits mais est sensible au bruit
            base_score = 0.85
            qubit_bonus = min(0.1, num_qubits * 0.02)
            noise_impact = noise_level * 15
            depth_impact = min(0.3, circuit_depth * 0.04)
            success = max(0.1, min(0.95, base_score + qubit_bonus - noise_impact - depth_impact))
            
            # Métriques spécifiques à Grover
            optimal_iterations = int(np.pi/4 * np.sqrt(2**num_qubits))
            extras = {
                'marked_states': 1,
                'total_states': 2**num_qubits,
                'optimal_iterations': optimal_iterations,
                'actual_iterations': int(optimal_iterations * (1 + noise_level * 2))
            }
            
        elif algorithm == 'qft':
            # QFT est très sensible au bruit et à la profondeur
            base_score = 0.8
            qubit_penalty = num_qubits * 0.03
            noise_impact = noise_level * 20
            depth_impact = min(0.4, circuit_depth * 0.05)
            success = max(0.1, min(0.95, base_score - qubit_penalty - noise_impact - depth_impact))
            
            # Métriques spécifiques à QFT
            phase_error = np.random.normal(0, noise_level)
            extras = {
                'phase_precision': max(0, 1.0 - 5 * abs(phase_error)),
                'frequency_resolution': 2**num_qubits,
                'circuit_gates': num_qubits**2
            }
            
        else:
            success = 0.5  # Par défaut pour les algorithmes inconnus
            extras = {}
        
        # Métriques quantiques génériques
        fidelity = max(0.1, success * 0.9)
        coherence_time = int(100 * (1.0 - noise_level))
        entanglement = max(0.1, min(0.99, 0.7 + 0.1 * num_qubits - 0.5 * noise_level))
        
        # Combinaison des résultats génériques et spécifiques
        result = {
            'main_score': success,
            'circuit_fidelity': fidelity,
            'coherence_time': coherence_time,
            'entanglement_measure': entanglement,
            'details': f"Exécuté {algorithm} avec {num_qubits} qubits, bruit={noise_level}, profondeur={circuit_depth}"
        }
        
        # Ajouter les métriques spécifiques à l'algorithme
        result.update(extras)
        
        return result
        
    def simulate_vqe(self, num_qubits, noise_level, circuit_depth):
        """
        Simulation spécifique pour VQE.
        
        Args:
            num_qubits: Nombre de qubits
            noise_level: Niveau de bruit
            circuit_depth: Profondeur du circuit
            
        Returns:
            Résultats de la simulation VQE
        """
        # Simuler une optimisation de fonction objectif quantique
        iterations = 20
        energies = []
        
        # Dans un VQE réel, on minimiserait l'énergie d'un hamiltonien
        # Ici on simule simplement une convergence vers une valeur cible
        current_energy = -0.5 * num_qubits
        target_energy = -1.0 * num_qubits
        
        for i in range(iterations):
            # Simuler convergence avec bruit
            improvement = (target_energy - current_energy) * 0.1 * (1 - noise_level)
            noise = np.random.normal(0, noise_level * 0.2)
            current_energy += improvement + noise
            energies.append(float(current_energy))
        
        return {
            'final_energy': float(current_energy),
            'target_energy': float(target_energy),
            'convergence': energies,
            'success_metric': float(abs(current_energy / target_energy))
        }
    
    def get_supported_algorithms(self):
        """
        Renvoie la liste des algorithmes supportés.
        
        Returns:
            Liste des algorithmes supportés et leurs descriptions
        """
        return {
            'vqe': 'Variational Quantum Eigensolver - optimise les paramètres pour trouver l\'état fondamental',
            'grover': 'Algorithme de recherche de Grover - recherche dans une base de données non structurée',
            'qft': 'Transformée de Fourier Quantique - effectue une DFT sur un état quantique'
        }