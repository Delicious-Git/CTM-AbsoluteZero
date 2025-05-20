"""
Démonstration de tâches quantiques avec CTM-AbsoluteZero
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ctm.quantum_sim import QuantumSimulator
from src.ctm_az_agent import CTM_AbsoluteZero_Agent

def load_quantum_config():
    """Charge la configuration quantique depuis un fichier YAML"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               "configs", "quantum.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def run_quantum_algorithms_demo():
    """Démontre les différents algorithmes quantiques"""
    print("\n==== Démonstration des algorithmes quantiques ====")
    simulator = QuantumSimulator()
    
    algorithms = ['vqe', 'grover', 'qft']
    qubit_ranges = range(2, 11)
    
    results = {}
    
    for algo in algorithms:
        scores = []
        for qubits in qubit_ranges:
            params = {
                'algorithm': algo,
                'num_qubits': qubits,
                'noise_level': 0.01,
                'circuit_depth': 5
            }
            result = simulator.run(params)
            scores.append(result['main_score'])
            print(f"  {algo.upper()} avec {qubits} qubits: Score = {result['main_score']:.3f}, "
                  f"Fidélité = {result['circuit_fidelity']:.3f}")
        results[algo] = scores
    
    # Visualisation
    plt.figure(figsize=(10, 6))
    for algo, scores in results.items():
        plt.plot(list(qubit_ranges), scores, marker='o', label=algo)
    
    plt.xlabel('Nombre de qubits')
    plt.ylabel('Score de performance')
    plt.title('Performance des algorithmes quantiques par nombre de qubits')
    plt.legend()
    plt.grid(True)
    
    # Créer le répertoire results s'il n'existe pas
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/quantum_algorithms_performance.png")
    plt.show()

def run_noise_impact_demo():
    """Démontre l'impact du bruit sur la performance quantique"""
    print("\n==== Impact du bruit sur les performances quantiques ====")
    simulator = QuantumSimulator()
    
    noise_levels = np.linspace(0, 0.2, 10)
    
    vqe_scores = []
    grover_scores = []
    qft_scores = []
    
    for noise in noise_levels:
        vqe_params = {
            'algorithm': 'vqe',
            'num_qubits': 5,
            'noise_level': noise,
            'circuit_depth': 5
        }
        grover_params = {
            'algorithm': 'grover',
            'num_qubits': 5,
            'noise_level': noise,
            'circuit_depth': 5
        }
        qft_params = {
            'algorithm': 'qft',
            'num_qubits': 5,
            'noise_level': noise,
            'circuit_depth': 5
        }
        
        vqe_result = simulator.run(vqe_params)
        grover_result = simulator.run(grover_params)
        qft_result = simulator.run(qft_params)
        
        vqe_scores.append(vqe_result['main_score'])
        grover_scores.append(grover_result['main_score'])
        qft_scores.append(qft_result['main_score'])
        
        print(f"  Niveau de bruit {noise:.2f}: VQE = {vqe_result['main_score']:.3f}, "
              f"Grover = {grover_result['main_score']:.3f}, QFT = {qft_result['main_score']:.3f}")
    
    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, vqe_scores, marker='o', label='VQE')
    plt.plot(noise_levels, grover_scores, marker='s', label='Grover')
    plt.plot(noise_levels, qft_scores, marker='^', label='QFT')
    
    plt.xlabel('Niveau de bruit')
    plt.ylabel('Score de performance')
    plt.title('Impact du bruit sur les algorithmes quantiques')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/quantum_noise_impact.png")
    plt.show()

def train_quantum_agent():
    """Entraîne un agent spécifiquement sur les tâches quantiques"""
    print("\n==== Entraînement d'un agent sur les tâches quantiques ====")
    
    # Charger la configuration quantique
    config = load_quantum_config()
    
    # Créer l'agent
    agent = CTM_AbsoluteZero_Agent(
        ctm_solver_config=config["ctm"],
        az_hyperparams=config["absolute_zero"],
        ppo_config_dict=config["ppo"]
    )
    
    # Exécuter quelques étapes d'entraînement
    rewards = []
    print("\nExécution de 10 étapes d'entraînement sur le domaine quantique:")
    
    for i in range(10):
        reward, task = agent.run_self_play_step("quantum")
        rewards.append(reward)
        
        # Afficher les paramètres de tâche pour les itérations 1, 5 et 10
        if i in [0, 4, 9]:
            print(f"\nÉtape {i+1}, Tâche:")
            print(f"  Type: {task['type']}")
            print(f"  Paramètres: {task['params']}")
            print(f"  Récompense: {reward:.3f}")
    
    # Visualisation des récompenses
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(rewards) + 1), rewards, marker='o')
    plt.xlabel('Étape')
    plt.ylabel('Récompense')
    plt.title('Évolution des récompenses pour les tâches quantiques')
    plt.grid(True)
    plt.savefig("results/quantum_agent_rewards.png")
    plt.show()
    
    print("\nEntraînement terminé! Pour un entraînement complet, exécutez:")
    print("agent.train(steps=10000, domains=['quantum'])")

def main():
    """Fonction principale exécutant toutes les démonstrations"""
    print("Démonstration CTM-AbsoluteZero: Tâches Quantiques")
    print("=" * 50)
    
    # Créer le répertoire results s'il n'existe pas
    os.makedirs("results", exist_ok=True)
    
    # Exécuter les démos
    run_quantum_algorithms_demo()
    run_noise_impact_demo()
    
    # Vérifier si CUDA est disponible pour l'entraînement de l'agent
    if torch.cuda.is_available():
        train_quantum_agent()
    else:
        print("\nL'entraînement de l'agent a été ignoré car CUDA n'est pas disponible.")
        print("Pour exécuter cette partie, assurez-vous d'avoir un GPU compatible CUDA.")

if __name__ == "__main__":
    main()