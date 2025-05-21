#!/bin/bash
# Script de démarrage pour CTM-AbsoluteZero en production

echo "Démarrage de CTM-AbsoluteZero en production..."

# Vérifier si Docker est installé
if ! command -v docker &> /dev/null
then
    echo "Docker n'est pas installé. Veuillez installer Docker et Docker Compose."
    exit 1
fi

# Vérifier si Docker Compose est installé
if ! command -v docker-compose &> /dev/null
then
    echo "Docker Compose n'est pas installé. Veuillez installer Docker Compose."
    exit 1
fi

# Créer les répertoires nécessaires s'ils n'existent pas
mkdir -p data logs models state

# Vérifier si le fichier .env existe
if [ ! -f .env ]; then
    echo "Le fichier .env n'existe pas. Création à partir de .env.example..."
    cp .env.example .env
    echo "ATTENTION: Veuillez éditer le fichier .env pour configurer vos clés API et autres paramètres."
    exit 1
fi

# Vérifier les clés API
if grep -q "your_claude_api_key_here" .env; then
    echo "ATTENTION: Clé API Claude non configurée dans le fichier .env"
    echo "Veuillez éditer le fichier .env et configurer votre clé API Claude."
    exit 1
fi

# Démarrer les services avec Docker Compose
echo "Démarrage des services Docker..."
docker-compose up -d

# Vérifier l'état des services
echo "Vérification de l'état des services..."
sleep 10
docker-compose ps

echo "Vérification des logs..."
docker-compose logs --tail=20

echo "CTM-AbsoluteZero démarré en production."
echo "Service de monitoring disponible à l'adresse: http://localhost:8080"
echo "API disponible à l'adresse: http://localhost:8000"
echo ""
echo "Pour arrêter les services, exécutez: docker-compose down"
echo "Pour voir les logs en temps réel, exécutez: docker-compose logs -f"