"""
Exemple d'entraînement de base pour CTM-AbsoluteZero
"""
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ctm_az_agent import CTM_AbsoluteZero_Agent

def setup_agent():
    """Initialise et configure l'agent CTM-AbsoluteZero"""
    ctm_config = {
        "model_path": "models/ctm_base",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "precision": "float16"
    }
    
    az_hyperparams = {
        "w_solve": 0.5,
        "w_propose": 0.3,
        "w_novelty": 0.1,
        "w_progress": 0.1,
        "solve_success_threshold": 0.6,
        "learnability_target_success_rate": 0.5
    }
    
    ppo_params = {
        "learning_rate": 3e-6,
        "batch_size": 16,
        "mini_batch_size": 4,
        "ppo_epochs": 4,
        "clip_range": 0.15,
        "target_kl": 0.015
    }
    
    agent = CTM_AbsoluteZero_Agent(
        ctm_solver_config=ctm_config,
        proposer_llm_name="mistralai/Mistral-7B-Instruct-v0.1",
        az_hyperparams=az_hyperparams,
        ppo_config_dict=ppo_params
    )
    
    return agent

def run_single_domain_demo(agent, domain, steps=5):
    """
    Exécute quelques étapes d'auto-apprentissage sur un domaine spécifique.
    
    Args:
        agent: Instance de CTM_AbsoluteZero_Agent
        domain: Domaine à tester
        steps: Nombre d'étapes à exécuter
    """
    print(f"\n===== Test du domaine: {domain} =====")
    
    rewards = []
    tasks = []
    
    # Feedback initial
    feedback = "Commencez avec une tâche de difficulté moyenne."
    
    for i in range(steps):
        print(f"\nÉtape {i+1}/{steps}")
        reward, task = agent.run_self_play_step(domain, feedback)
        rewards.append(reward)
        tasks.append(task)
        
        # Générer un feedback basé sur la récompense
        if reward < 0.3:
            feedback = "La tâche était trop difficile. Générez une tâche plus simple."
        elif reward > 0.8:
            feedback = "La tâche était trop facile. Augmentez la complexité."
        else:
            feedback = "Le niveau de difficulté est bon. Essayez quelque chose de similaire."
    
    print(f"\nRésultats finaux pour {domain}:")
    print(f"Récompenses: {[round(r, 2) for r in rewards]}")
    print(f"Récompense moyenne: {np.mean(rewards):.3f}")
    
    # Montrer un exemple de tâche
    if tasks:
        print(f"Exemple de tâche générée: {tasks[-1]}")
    
    return rewards, tasks

def plot_rewards(domain_rewards):
    """
    Trace un graphique des récompenses pour différents domaines.
    
    Args:
        domain_rewards: Dictionnaire {domain: [rewards]}
    """
    plt.figure(figsize=(10, 6))
    
    for domain, rewards in domain_rewards.items():
        plt.plot(range(1, len(rewards) + 1), rewards, marker='o', label=domain)
    
    plt.xlabel('Étape')
    plt.ylabel('Récompense')
    plt.title('Évolution des récompenses par domaine')
    plt.legend()
    plt.grid(True)
    
    # Créer le répertoire results s'il n'existe pas
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/domain_rewards.png")
    plt.show()

def run_demo():
    """Exécute une démonstration complète"""
    agent = setup_agent()
    
    # Liste des domaines à tester
    domains = ["maze", "sorting", "image_classification", "quantum"]
    
    domain_rewards = {}
    
    for domain in domains:
        rewards, _ = run_single_domain_demo(agent, domain, steps=5)
        domain_rewards[domain] = rewards
    
    # Visualiser les résultats
    plot_rewards(domain_rewards)
    
    print("\nDémonstration terminée! Vous pouvez maintenant exécuter un entraînement complet avec:")
    print("agent.train(steps=10000, domains=domains)\n")

if __name__ == "__main__":
    print("CTM-AbsoluteZero - Démonstration d'entraînement de base")
    print("=" * 50)
    
    # Créer le répertoire models s'il n'existe pas
    Path("models").mkdir(exist_ok=True)
    
    run_demo()