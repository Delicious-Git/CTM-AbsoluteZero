# Guide Utilisateur CTM-AbsoluteZero

## Introduction

CTM-AbsoluteZero est un framework d'auto-apprentissage qui combine la Continuous Thought Machine (CTM) et le paradigme Absolute Zero Reasoner avec une simulation quantique légère. Ce guide détaille toutes les fonctionnalités disponibles et leur utilisation.

## Table des matières

1. [Installation](#installation)
2. [Architecture du système](#architecture-du-système)
3. [Interface en ligne de commande](#interface-en-ligne-de-commande)
4. [Composants principaux](#composants-principaux)
5. [Configuration](#configuration)
6. [Tâches et domaines](#tâches-et-domaines)
7. [Intégration DFZ](#intégration-dfz)
8. [Exécution concurrente de tâches](#exécution-concurrente-de-tâches)
9. [Simulateur quantique](#simulateur-quantique)
10. [Exemples pratiques](#exemples-pratiques)
11. [Dépannage](#dépannage)
12. [Bonnes pratiques](#bonnes-pratiques)

## Installation

### Prérequis

- Python 3.7+
- CUDA (facultatif, pour l'accélération GPU)
- 4 Go de RAM minimum (8 Go recommandés)

### Avec pip

```bash
git clone https://github.com/your-username/CTM-AbsoluteZero.git
cd CTM-AbsoluteZero
pip install -r requirements.txt
```

### Avec Docker

```bash
git clone https://github.com/your-username/CTM-AbsoluteZero.git
cd CTM-AbsoluteZero
docker-compose up -d
```

## Architecture du système

CTM-AbsoluteZero utilise une architecture Proposer/Solver :

1. **Proposer (LLM)** : Génère des tâches adaptatives basées sur l'apprentissage précédent.
2. **Solver (CTM)** : Résout les tâches proposées à l'aide de divers composants spécialisés.
3. **Système de récompense** : Évalue la performance, la faisabilité, la nouveauté et la progression.
4. **Transfert inter-domaines** : Facilite le partage de connaissances entre différents types de tâches.
5. **Contrôleur de phase** : Gère les transitions entre les phases d'évolution du système.

![Architecture du système](https://via.placeholder.com/800x400?text=Architecture+CTM-AbsoluteZero)

## Interface en ligne de commande

L'interface en ligne de commande est le moyen le plus direct d'utiliser CTM-AbsoluteZero :

### Générer des tâches

```bash
python -m src.cli generate --domain maze --count 3
```

Options :
- `--domain` : Le domaine des tâches (maze, quantum, sorting, image_classification)
- `--count` : Nombre de tâches à générer
- `--difficulty` : Niveau de difficulté (easy, medium, hard)

### Résoudre une tâche

```bash
python -m src.cli solve --task "Solve a 10x10 maze with multiple paths" --domain maze
```

Options :
- `--task` : Description de la tâche à résoudre
- `--domain` : Domaine de la tâche
- `--params` : Paramètres supplémentaires au format JSON

### Entraîner l'agent

```bash
python -m src.cli train --domain quantum --iterations 1000
```

Options :
- `--domain` : Domaine d'entraînement (peut être une liste séparée par des virgules)
- `--iterations` : Nombre d'itérations d'entraînement
- `--eval-interval` : Intervalle d'évaluation

### Évaluer l'agent

```bash
python -m src.cli evaluate --domain sorting --num-tasks 20
```

Options :
- `--domain` : Domaine d'évaluation
- `--num-tasks` : Nombre de tâches pour l'évaluation
- `--save-results` : Chemin pour enregistrer les résultats

### Intégration DFZ

```bash
python -m src.cli dfz --interactive
```

Options :
- `--interactive` : Mode interactif pour l'intégration DFZ
- `--config` : Chemin vers un fichier de configuration spécifique

## Composants principaux

### AbsoluteZeroAgent

Classe principale qui intègre le Proposer (LLM) et le Solver (CTM), gérant la boucle d'auto-apprentissage.

```python
from src.ctm_az_agent import AbsoluteZeroAgent
from src.ctm.interface import RealCTMInterface
from src.rewards.composite import CompositeRewardSystem
from src.transfer.adapter import NeuralTransferAdapter
from src.utils.config import ConfigManager

# Charger la configuration
config = ConfigManager("configs/default.yaml").to_dict()

# Initialiser les composants
ctm_interface = RealCTMInterface(config["ctm"])
reward_system = CompositeRewardSystem(...)
transfer_adapter = NeuralTransferAdapter(config["domains"])

# Initialiser l'agent
agent = AbsoluteZeroAgent(
    proposer_model_path="models/proposer",
    solver_model_path="models/solver",
    reward_system=reward_system,
    transfer_adapter=transfer_adapter,
    ctm_interface=ctm_interface,
    config=config["agent"]
)

# Génération de tâches
tasks = agent.generate_tasks(domain="quantum", count=1)
print(tasks[0])
# {'id': 'task_123', 'domain': 'quantum', 'description': 'Implement a QFT algorithm for 6 qubits', 'parameters': {'algorithm': 'qft', 'num_qubits': 6, 'noise_level': 0.01, 'circuit_depth': 5}}

# Résolution de tâche
result = agent.solve_task(tasks[0])
print(result)
# {'success': True, 'solution': {...}, 'metrics': {'execution_time': 0.34, 'solution_quality': 0.92}}
```

### SemanticNoveltyTracker

Détecte les tâches réellement nouvelles en utilisant des empreintes de paramètres et des similarités cosinus.

```python
from src.rewards.novelty import SemanticNoveltyTracker

tracker = SemanticNoveltyTracker()
task1 = {"description": "Solve 10x10 maze", "embedding": [...]}
task2 = {"description": "Solve 11x10 maze", "embedding": [...]}
task3 = {"description": "Solve 20x20 maze", "embedding": [...]}

print(tracker.compute_novelty(task1))  # 1.0 (première tâche)
print(tracker.compute_novelty(task2))  # 0.1 (similaire à task1)
print(tracker.compute_novelty(task3))  # 0.8 (significativement différente)
```

### SkillPyramid

Suit la progression hiérarchique des compétences à travers différents domaines et niveaux de difficulté.

```python
from src.rewards.progress import SkillPyramid

pyramid = SkillPyramid(domains=["maze", "quantum", "sorting"])
task = {"domain": "quantum", "challenge_level": 3, "success": True}

# Enregistrer l'achèvement réussi de la tâche
pyramid.update_skill(task)

# Obtenir le niveau de compétence actuel
skill_level = pyramid.get_skill_level("quantum")
print(f"Niveau de compétence quantique : {skill_level}")  # 2

# Mettre à l'échelle la récompense en fonction du niveau de compétence
reward = pyramid.scale_reward("quantum", 1.0)
print(f"Récompense mise à l'échelle : {reward}")  # 0.75 (récompense inférieure à mesure que la compétence s'améliore)
```

### NeuralTransferAdapter

Permet le transfert de connaissances entre domaines en modifiant les paramètres des tâches.

```python
from src.transfer.adapter import NeuralTransferAdapter

adapter = NeuralTransferAdapter(["maze", "quantum", "sorting"])

# Adapter une tâche d'un domaine à un autre
source_task = {
    "domain": "sorting",
    "description": "Implement quicksort",
    "parameters": {"array_size": 1000, "complexity": "O(n log n)"}
}

adapted_task = adapter.adapt_task(
    task=source_task,
    source_domain="sorting",
    target_domain="maze"
)

print(f"Tâche adaptée : {adapted_task['description']}")
# "Create a maze that requires a divide and conquer approach"
```

### PhaseController

Gère les transitions entre les phases d'entraînement et ajuste les poids de récompense en conséquence.

```python
from src.transfer.phase import PhaseController

controller = PhaseController()
print(controller.current_phase)  # 'exploration'
print(controller.get_phase_weights())  # {'solve': 0.4, 'discover': 0.6}

# Simuler une transition de phase
metrics = {'success_rate': 0.75, 'cross_domain_transfer': 0.45}
controller.update_phase(metrics)
print(controller.current_phase)  # 'exploitation'
```

## Configuration

CTM-AbsoluteZero utilise des fichiers YAML pour la configuration. Les principaux fichiers sont :

- `configs/default.yaml` : Configuration par défaut
- `configs/quantum.yaml` : Configuration spécifique au domaine quantique
- `configs/production.yaml` : Configuration de production

### Structure de la configuration par défaut

```yaml
ctm:
  model_path: "models/ctm_base"
  device: "cuda"
  precision: "float16"
  max_batch_size: 32

absolute_zero:
  w_solve: 0.5
  w_propose: 0.3
  w_novelty: 0.1
  w_progress: 0.1
  solve_success_threshold: 0.6
  learnability_target_success_rate: 0.5

ppo:
  learning_rate: 3.0e-6
  batch_size: 16
  mini_batch_size: 4
  ppo_epochs: 4
  clip_range: 0.15
  target_kl: 0.015
```

### Configuration personnalisée

```python
from src.utils.config import ConfigManager
from src.ctm_az_agent import AbsoluteZeroAgent

# Charger une configuration personnalisée
config_manager = ConfigManager("configs/quantum.yaml")
config = config_manager.to_dict()

# Initialiser l'agent
agent = AbsoluteZeroAgent(...)

# Générer des tâches quantiques
tasks = agent.generate_tasks(domain="quantum", count=3)
for task in tasks:
    print(f"Tâche : {task['description']}")
    
    # Résoudre la tâche
    result = agent.solve_task(task)
    print(f"Résultat : {result}")
```

## Tâches et domaines

CTM-AbsoluteZero prend en charge plusieurs domaines de tâches, chacun avec ses propres paramètres et défis.

### Domaine du labyrinthe (Maze)

```python
# Exemple de paramètres de tâche de labyrinthe
maze_task = {
    "type": "maze", 
    "params": {
        "size_x": 12, 
        "size_y": 12, 
        "complexity": 0.8, 
        "seed": 42,
        "visual_patterns": False
    }
}
```

### Domaine de classification d'images

```python
# Exemple de paramètres de tâche de classification d'images
image_task = {
    "type": "image_classification", 
    "params": {
        "target_classes": ["cat", "dog"], 
        "difficulty": "hard", 
        "data_source": "imagenet_subset_xyz",
        "hierarchy_depth": 1
    }
}
```

### Domaine de tri

```python
# Exemple de paramètres de tâche de tri
sorting_task = {
    "type": "sorting", 
    "params": {
        "list_length": 50, 
        "value_range": [0, 1000], 
        "list_type": "almost_sorted_descending"
    }
}
```

### Domaine quantique

```python
# Exemple de paramètres de tâche quantique
quantum_task = {
    "type": "quantum",
    "params": {
        "algorithm": "vqe",  # ou "grover", "qft"
        "num_qubits": 5,
        "noise_level": 0.02,
        "circuit_depth": 6
    }
}
```

## Intégration DFZ

CTM-AbsoluteZero peut être intégré au système d'intelligence conversationnelle DFZ.

### Utilisation de l'adaptateur DFZ

```python
from src.integration.dfz import DFZAdapter
import asyncio

async def main():
    # Initialiser l'adaptateur DFZ
    adapter = DFZAdapter(dfz_path="/path/to/dfz")
    await adapter.initialize()
    
    # Générer des tâches en utilisant le contexte de conversation
    tasks = await adapter.generate_task(
        domain="maze",
        context={"difficulty": "medium", "user_skill": "beginner"}
    )
    
    # Exécuter une tâche
    result = await adapter.execute_task(tasks[0])
    
    # Envoyer un message à DFZ
    response = await adapter.send_message(
        "La tâche a été accomplie avec succès avec un score de 85%",
        context={"task_id": tasks[0]["id"]}
    )
    
    print(f"Réponse DFZ : {response}")

# Exécuter la fonction asynchrone
asyncio.run(main())
```

## Exécution concurrente de tâches

CTM-AbsoluteZero prend en charge l'exécution concurrente de tâches pour améliorer les performances et le débit.

### Utilisation de l'adaptateur DFZ concurrent

```python
from src.integration.dfz_concurrent import ConcurrentDFZAdapter
import asyncio

async def main():
    # Créer un adaptateur DFZ concurrent avec 4 workers
    adapter = ConcurrentDFZAdapter(
        config={"config_path": "configs/default.yaml"},
        max_workers=4
    )
    
    # Initialiser l'adaptateur
    await adapter.initialize()
    
    # Définir un batch de tâches
    tasks = [
        {
            "id": "task_1",
            "domain": "quantum",
            "description": "Quantum task 1",
            "parameters": {
                "algorithm": "vqe",
                "num_qubits": 5,
                "noise_level": 0.01,
                "circuit_depth": 4
            }
        },
        {
            "id": "task_2",
            "domain": "quantum",
            "description": "Quantum task 2",
            "parameters": {
                "algorithm": "grover",
                "num_qubits": 6,
                "noise_level": 0.02,
                "circuit_depth": 3
            }
        }
    ]
    
    # Exécuter les tâches de manière concurrente
    result = await adapter.execute_tasks_batch(tasks, wait=True)
    
    # Traiter les résultats
    for task_id, task_result in result["results"].items():
        print(f"Tâche {task_id}: {task_result['status']}")

asyncio.run(main())
```

### Exécution de tests de performance

```bash
# Exécuter un benchmark de base sur des tâches quantiques avec 4 workers
python examples/test_dfz_concurrency.py --domain quantum --count 20 --workers 4

# Exécuter un test de mise à l'échelle pour mesurer les performances avec différents nombres de workers
python examples/test_dfz_concurrency.py --domain quantum --count 20 --scaling-test --max-workers 1,2,4,8,16

# Utiliser le parallélisme basé sur les processus au lieu des threads
python examples/test_dfz_concurrency.py --domain quantum --count 20 --workers 4 --use-processes
```

### Threads vs. Processus

- **Threads** : Adaptés aux tâches liées aux E/S, modèle de mémoire partagée, faible surcharge
- **Processus** : Adaptés aux tâches liées au CPU, modèle de mémoire isolée, surcharge plus élevée

### Nombre optimal de workers

Le nombre optimal de workers dépend de :

1. La nature des tâches (liées au CPU vs. liées aux E/S)
2. Les ressources système disponibles (cœurs CPU, mémoire)
3. Les caractéristiques des tâches (temps d'exécution, utilisation des ressources)

Pour les tâches liées au CPU, un bon point de départ est d'utiliser un nombre de workers égal au nombre de cœurs CPU disponibles. Pour les tâches liées aux E/S, vous pourriez bénéficier de l'utilisation de plus de workers que de cœurs CPU.

## Simulateur quantique

Le module de simulation quantique légère permet de tester des algorithmes quantiques sans matériel spécialisé.

```python
from src.ctm.quantum_sim import QuantumSimulator

simulator = QuantumSimulator()
result = simulator.run({
    'algorithm': 'grover', 
    'num_qubits': 5, 
    'noise_level': 0.01,
    'circuit_depth': 4
})

print(f"Succès : {result['main_score']}")
print(f"Fidélité du circuit : {result['circuit_fidelity']}")
print(f"Solution : {result['details']}")
```

### Algorithmes supportés

- **VQE (Variational Quantum Eigensolver)** : Optimise les paramètres pour trouver l'état fondamental
- **Algorithme de recherche de Grover** : Recherche dans une base de données non structurée
- **QFT (Transformée de Fourier Quantique)** : Effectue une transformée de Fourier discrète sur un état quantique

## Exemples pratiques

### Entraînement de base

```python
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

# Créer l'agent
agent = setup_agent()

# Entraîner sur un domaine spécifique
rewards = []
feedback = "Commencez avec une tâche de difficulté moyenne."

for i in range(5):
    print(f"\nÉtape {i+1}/5")
    reward, task = agent.run_self_play_step("quantum", feedback)
    rewards.append(reward)
    
    # Générer un feedback basé sur la récompense
    if reward < 0.3:
        feedback = "La tâche était trop difficile. Générez une tâche plus simple."
    elif reward > 0.8:
        feedback = "La tâche était trop facile. Augmentez la complexité."
    else:
        feedback = "Le niveau de difficulté est bon. Essayez quelque chose de similaire."

print(f"\nRésultats finaux pour quantum:")
print(f"Récompenses: {[round(r, 2) for r in rewards]}")
print(f"Récompense moyenne: {np.mean(rewards):.3f}")
```

### Tâches quantiques

```python
from src.ctm.quantum_sim import QuantumSimulator

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

run_quantum_algorithms_demo()
```

## Dépannage

### Problèmes d'installation

**Problème** : Erreur lors de l'installation des dépendances
**Solution** : Vérifiez que vous avez la bonne version de Python et pip, puis essayez :
```bash
pip install -r requirements.txt --no-cache-dir
```

**Problème** : Erreur CUDA lors de l'utilisation du GPU
**Solution** : Vérifiez la compatibilité de votre GPU avec la version CUDA installée :
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Problèmes d'exécution

**Problème** : Échecs fréquents des tâches
**Solution** : Vérifiez les paramètres de tâche et assurez-vous qu'ils sont dans des plages valides :
```python
# Vérifier les paramètres de tâche
valid_task = agent.validate_task(task_params)
print(f"Tâche valide : {valid_task}")
```

**Problème** : Performance médiocre
**Solution** : Ajustez les hyperparamètres dans la configuration :
```python
# Ajuster les hyperparamètres de l'agent
agent.az_hparams["w_solve"] = 0.7  # Augmenter l'importance de la résolution
agent.az_hparams["learning_rate"] = 5e-6  # Augmenter le taux d'apprentissage
```

## Bonnes pratiques

1. **Commencez avec des tâches simples** : Commencez l'entraînement avec des tâches simples et augmentez progressivement la difficulté.

2. **Utilisez le curriculum learning** : Alternez entre différents domaines pour favoriser le transfert de connaissances.

3. **Surveillez les métriques de performance** : Suivez les taux de réussite, les niveaux de compétence et les métriques de transfert.

4. **Utilisez l'exécution concurrente pour les grandes charges de travail** : Pour de nombreuses tâches indépendantes, utilisez l'exécution concurrente.

5. **Enregistrez régulièrement les points de contrôle** : Sauvegardez l'état de l'agent à intervalles réguliers pendant l'entraînement.

6. **Optimisez les paramètres de simulation quantique** : Ajustez le niveau de bruit et la profondeur du circuit en fonction de vos besoins de précision.

7. **Personnalisez les poids de récompense** : Adaptez les poids des différentes composantes de récompense en fonction de vos objectifs d'entraînement.

## Conclusion

CTM-AbsoluteZero est un framework puissant pour l'auto-apprentissage et le transfert de connaissances entre domaines. Ce guide utilisateur a présenté toutes les fonctionnalités disponibles et leur utilisation. Pour plus d'informations, consultez les exemples fournis et la documentation API détaillée.

Si vous rencontrez des problèmes ou avez des questions, n'hésitez pas à consulter les ressources supplémentaires ou à contacter l'équipe de support.