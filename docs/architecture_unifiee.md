# Architecture Unifiée CTM-AbsoluteZero

## Vue d'ensemble

L'architecture unifiée CTM-AbsoluteZero intègre tous les composants développés pour offrir une solution complète de génération et d'exécution de tâches adaptatives avec optimisation des performances et des coûts. Cette architecture tire parti des optimisations de tokens, de l'exécution concurrente, et de la gestion dynamique des ressources pour maximiser l'efficacité et la scalabilité du système.

![Architecture unifiée](https://via.placeholder.com/800x600?text=Architecture+Unifi%C3%A9e+CTM-AbsoluteZero)

## Composants Principaux

L'architecture unifiée est organisée en plusieurs couches fonctionnelles:

### 1. Couche Modèle de Langage

Cette couche abstrait les interactions avec les modèles de langage:

- **Adaptateurs LLM**: Interfaces standardisées pour différents modèles (Claude, DeepSeek)
- **Optimisation de tokens**: Réduction de la redondance et optimisation des prompts
- **Routage intelligent**: Sélection dynamique du modèle optimal selon le type de tâche

Le module `src/llm` contient toutes les classes et fonctions nécessaires à cette couche:
- `ClaudeAdapter`: Interface pour les modèles Claude
- `DeepSeekAdapter`: Interface pour les modèles DeepSeek
- `PromptOptimizer`: Optimisation des prompts pour réduire l'utilisation de tokens
- `TokenReducer`: Utilitaires pour réduire l'utilisation de tokens

### 2. Couche Exécution de Tâches

Cette couche gère l'exécution des tâches générées:

- **CTM Interface**: Interface principale pour l'exécution des tâches
- **Composants spécialisés**: Modules d'exécution pour différents domaines (maze, sorting, quantum, etc.)
- **DFZ Integration**: Connecteurs avec le système DFZ

Le répertoire `src/ctm` contient l'implémentation des différents composants d'exécution:
- `interface.py`: Interface principale du CTM
- `maze_solver.py`, `sorter.py`, `quantum_sim.py`, etc.: Composants spécialisés

### 3. Couche Concurrence et Ressources

Cette couche gère l'exécution concurrente et l'allocation de ressources:

- **ConcurrentDFZAdapter**: Adaptateur DFZ avec support pour l'exécution concurrente
- **ConcurrentTaskManager**: Gestionnaire de tâches concurrentes
- **ResourceManager**: Gestionnaire de ressources dynamique
- **PriorityQueue**: File d'attente prioritaire pour les tâches

Les modules `src/integration/dfz_concurrent.py` et `src/integration/resource_manager.py` forment le cœur de cette couche.

### 4. Couche Système et Monitoring

Cette couche fournit des fonctionnalités système et de monitoring:

- **Journalisation**: Système de journalisation unifié
- **Configuration**: Gestion de la configuration
- **Métriques**: Collection de métriques de performance
- **Persistence**: Stockage et récupération d'état

Les modules dans `src/utils` fournissent ces fonctionnalités de base.

### 5. Couche API et Interface

Cette couche expose les fonctionnalités du système:

- **CLI**: Interface en ligne de commande
- **API HTTP**: API REST pour l'intégration avec d'autres systèmes
- **WebApp**: Interface web pour le monitoring (optionnelle)

## Flux de Données

Le flux de données dans l'architecture unifiée suit les étapes suivantes:

1. **Génération de tâche**:
   - Le LLM Proposer génère une tâche
   - Le prompt est optimisé pour réduire l'utilisation de tokens
   - La tâche est validée

2. **Préparation de l'exécution**:
   - La tâche est soumise au ResourceManager
   - Une priorité est assignée selon le domaine et les paramètres
   - Les ressources nécessaires sont estimées

3. **Planification**:
   - Le ResourceManager détermine le nombre optimal de workers
   - La tâche est placée dans la file d'attente prioritaire
   - Le ConcurrentTaskManager planifie l'exécution

4. **Exécution**:
   - La tâche est exécutée par le composant CTM approprié
   - Les métriques d'exécution sont collectées
   - Le profil de ressources est mis à jour

5. **Analyse et feedback**:
   - Les résultats sont analysés
   - Le système de récompense calcule une récompense
   - Le LLM Proposer est mis à jour via PPO

## Intégration des composants

Les sections suivantes décrivent comment les différents composants développés sont intégrés dans l'architecture unifiée.

### Optimisation des Tokens

L'optimisation des tokens est intégrée à deux niveaux:

1. **Niveau adaptateur LLM**:
   - Les adaptateurs Claude et DeepSeek intègrent l'optimisation de prompts
   - L'optimisation peut être configurée en mode agressif, équilibré ou minimal

2. **Niveau message**:
   - La classe `optimize_message_history` permet d'optimiser les historiques de conversation
   - Réduction de la redondance entre les messages

Un système de métriques permet de mesurer les économies de tokens et de calculer le ROI des optimisations.

### Exécution Concurrente

L'exécution concurrente est intégrée via:

1. **ConcurrentDFZAdapter**:
   - Extension de l'adaptateur DFZ standard avec support concurrent
   - Support pour les threads et les processus

2. **ConcurrentCTMAbsoluteZeroPlugin**:
   - Plugin étendu pour l'exécution concurrente
   - Intégration avec le DFZ

Le système de concurrence supporte:
- L'exécution par lots de tâches
- La collecte de résultats asynchrone
- Le suivi de métriques de performance détaillées

### Gestion Dynamique des Ressources

La gestion dynamique des ressources est intégrée via:

1. **ResourceManager**:
   - Profiles de ressources par domaine et type de tâche
   - Ajustement dynamique du nombre de workers

2. **PriorityQueue**:
   - File d'attente prioritaire pour les tâches
   - Priorités basées sur le domaine et l'importance

Le système monitore en continu l'utilisation des ressources système et ajuste le nombre de workers en conséquence.

## Mécanismes de Wake/Sleep

Le système implémente des mécanismes de mise en veille et de réveil pour optimiser l'utilisation des ressources:

1. **Détection d'inactivité**:
   - Surveillance des patterns d'utilisation
   - Mise en veille progressive des composants

2. **Wake Triggers**:
   - Hiérarchie d'activation pour les composants
   - Restauration de contexte lors du réveil

3. **Cohérence d'état**:
   - Validation de la cohérence lors du réveil
   - Journalisation des transitions sleep/wake

## Implémentation pour les différents environnements

L'architecture unifiée supporte plusieurs environnements de déploiement:

### Environnement de développement

```python
from src.llm import ClaudeAdapter, DeepSeekAdapter, optimize_prompt
from src.ctm.interface import RealCTMInterface
from src.integration.dfz_concurrent import ConcurrentDFZAdapter
from src.integration.resource_manager import ResourceManager

# Configuration de l'optimisation des tokens
prompt = "..."
optimized_prompt, metrics = optimize_prompt(prompt, mode="balanced")

# Configuration de l'exécution concurrente
adapter = ConcurrentDFZAdapter(max_workers=4)
await adapter.initialize()

# Configuration du gestionnaire de ressources
resource_manager = ResourceManager(max_workers=8)
resource_manager.start_monitoring()

# Exécution de tâches
tasks = [...]
for task in tasks:
    resource_manager.add_task(task_id=task["id"], task=task, priority=5)

while resource_manager.get_queue_length() > 0:
    task = resource_manager.get_next_task()
    if task:
        result = await adapter.execute_task(task)
        # Process result...
```

### Environnement de production

Pour l'environnement de production, l'architecture unifiée est déployée via Docker Compose:

```yaml
version: '3'

services:
  ctm-az-core:
    image: ctm-az/core:latest
    environment:
      - CLAUDE_API_KEY=${CLAUDE_API_KEY}
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
      - MAX_WORKERS=8
      - TOKEN_OPTIMIZATION_MODE=balanced
    volumes:
      - ./config:/app/config
      - ./data:/app/data
    ports:
      - "8000:8000"
    restart: unless-stopped

  ctm-az-monitor:
    image: ctm-az/monitor:latest
    depends_on:
      - ctm-az-core
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

## Interface de Programmation (API)

L'architecture unifiée expose une API REST pour permettre l'intégration avec d'autres systèmes:

```
POST /tasks
GET /tasks/{task_id}
GET /metrics
POST /optimize_prompt
GET /resource_profiles
POST /wake
POST /sleep
```

Exemple d'utilisation de l'API:

```bash
# Soumission d'une tâche
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "quantum",
    "description": "Implement a QFT algorithm",
    "parameters": {
      "algorithm": "qft",
      "num_qubits": 5,
      "noise_level": 0.01
    },
    "priority": 3
  }'
```

## Conclusion

L'architecture unifiée CTM-AbsoluteZero intègre tous les composants développés (optimisation de tokens, exécution concurrente, gestion des ressources) pour offrir une solution complète et efficace. Cette architecture permet:

1. Une réduction des coûts grâce à l'optimisation des tokens
2. Une amélioration des performances via l'exécution concurrente
3. Une utilisation optimale des ressources avec la gestion dynamique
4. Une meilleure évolutivité grâce aux mécanismes de wake/sleep

Les prochaines étapes incluent:
- L'amélioration des modèles de prédiction pour l'allocation de ressources
- L'intégration de l'apprentissage automatique pour les préférences utilisateur
- Le développement d'outils avancés de monitoring et de visualisation