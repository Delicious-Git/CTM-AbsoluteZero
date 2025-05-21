# Guide d'exécution en production de CTM-AbsoluteZero

Ce guide vous explique étape par étape comment exécuter CTM-AbsoluteZero en production de manière sécurisée.

## Prérequis

- Docker installé et configuré
- Docker Compose installé
- Au moins 16 Go de RAM
- Un processeur avec au moins 4 cœurs
- 20 Go d'espace disque disponible
- Clés API pour Claude et DeepSeek (si nécessaire)
- Serveur proxy inverse (Nginx ou Traefik) pour la sécurité en production

## Étapes d'exécution

### 1. Préparation de l'environnement

Clonez le dépôt CTM-AbsoluteZero si ce n'est pas déjà fait :

```bash
git clone https://github.com/your-organization/CTM-AbsoluteZero.git
cd CTM-AbsoluteZero
```

### 2. Configuration des variables d'environnement

Créez un fichier `.env` à partir du fichier d'exemple :

```bash
cp .env.example .env
```

Ouvrez le fichier `.env` avec votre éditeur de texte préféré et configurez les variables d'environnement :

```bash
nano .env
```

Assurez-vous de configurer au minimum :
- `CLAUDE_API_KEY` - Votre clé API Claude
- `DEEPSEEK_API_KEY` - Votre clé API DeepSeek
- `AUTH_USERNAME` - Nom d'utilisateur pour l'interface d'administration (ne pas utiliser la valeur par défaut)
- `AUTH_PASSWORD` - Mot de passe robuste (min. 12 caractères, complexe)
- Ajustez les autres paramètres selon vos besoins (optimisation des tokens, nombre de workers, etc.)

**Important pour la sécurité** : Ne stockez JAMAIS les clés API dans des dépôts git ou des emplacements publics. En environnement d'entreprise, utilisez plutôt un gestionnaire de secrets comme HashiCorp Vault ou AWS Secrets Manager.

### 3. Création des répertoires nécessaires

Assurez-vous que les répertoires nécessaires existent :

```bash
mkdir -p data logs models state
mkdir -p models/proposer models/solver
```

### 4. Préparation pour les modèles (si nécessaire)

Si vous utilisez des modèles locaux, placez-les dans les répertoires appropriés :
- Modèle proposer dans `models/proposer/`
- Modèle solver dans `models/solver/`

Sinon, créez des fichiers de configuration factices pour les tests :

```bash
echo '{"model": "dummy"}' > models/proposer/config.json
echo '{"model": "dummy"}' > models/solver/config.json
```

### 5. Exécution avec Docker Compose

Exécutez les services avec Docker Compose :

```bash
docker-compose up -d
```

Cette commande démarre tous les services en arrière-plan.

### 6. Vérification des services

Vérifiez que les services fonctionnent correctement :

```bash
docker-compose ps
```

Vous devriez voir les services `ctm-az-core` et `ctm-az-monitor` en état "Up".

Pour voir les logs des services :

```bash
docker-compose logs -f
```

### 7. Accès aux services

Une fois les services démarrés, ils sont uniquement accessibles localement pour des raisons de sécurité :

- Interface du moniteur : http://127.0.0.1:8080 (authentification requise)
- API Core : http://127.0.0.1:8000
- Point de santé du Core : http://127.0.0.1:8000/health
- Point de santé du Monitor : http://127.0.0.1:8080/health

#### Configuration d'un serveur proxy inverse (Recommandé pour la production)

Pour accéder aux services de manière sécurisée depuis l'extérieur, configurez un proxy inverse avec TLS :

1. Installez Nginx ou un autre serveur proxy
2. Générez des certificats TLS (Let's Encrypt ou certificat d'entreprise)
3. Configurez le proxy avec des règles de sécurité renforcées

Exemple de configuration Nginx :

```nginx
server {
    listen 443 ssl;
    server_name monitor.votredomaine.com;

    ssl_certificate     /chemin/vers/certificat.crt;
    ssl_certificate_key /chemin/vers/cle.key;
    
    # Configuration TLS sécurisée
    ssl_protocols TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers 'TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256';
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    
    # En-têtes de sécurité
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    
    # Proxy vers l'application Monitor
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # Limitation de débit (protection contre les attaques)
        limit_req zone=monitor_limit burst=20 nodelay;
        
        # Restriction IP (si nécessaire)
        # allow 192.168.1.0/24;  # Réseau d'entreprise
        # deny all;              # Bloquer tout le reste
    }
}

### 8. Arrêt des services

Pour arrêter les services :

```bash
docker-compose down
```

## Utilisation du script de démarrage automatique

Pour simplifier le démarrage, vous pouvez utiliser le script `start_production.sh` :

```bash
chmod +x scripts/start_production.sh
./scripts/start_production.sh
```

Ce script :
1. Vérifie que Docker et Docker Compose sont installés
2. Crée les répertoires nécessaires s'ils n'existent pas
3. Vérifie la présence du fichier `.env` et des clés API
4. Démarre les services avec Docker Compose
5. Vérifie l'état des services
6. Affiche les logs récents

## Dépannage

### Les services ne démarrent pas

1. Vérifiez les logs :
   ```bash
   docker-compose logs
   ```

2. Assurez-vous que les clés API sont correctement configurées dans le fichier `.env`

3. Vérifiez que les ports 8000 et 8080 ne sont pas déjà utilisés par d'autres services

### Problèmes de mémoire

Si vous rencontrez des problèmes de mémoire, ajustez les limites dans le fichier `docker-compose.yml` :

```yaml
deploy:
  resources:
    limits:
      memory: 4G  # Réduire si nécessaire
    reservations:
      memory: 2G  # Réduire si nécessaire
```

### Problèmes de modèles

Si vous rencontrez des erreurs liées aux modèles :
1. Vérifiez que les chemins dans `configs/production.yaml` sont correctement configurés
2. Assurez-vous que les modèles sont présents dans les répertoires appropriés

## Support

Si vous rencontrez des problèmes non couverts dans ce guide, consultez la documentation complète dans le répertoire `docs/` ou contactez l'équipe de support.