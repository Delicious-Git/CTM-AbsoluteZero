# Recommandations de sécurité pour CTM-AbsoluteZero en entreprise

## Risques identifiés

1. **Exposition des services** - Les services sont actuellement configurés pour être accessibles sur toutes les interfaces réseau
2. **Gestion des secrets** - Les clés API sont stockées dans des fichiers .env
3. **Absence d'authentification** - Pas de contrôle d'accès sur les interfaces web et API
4. **Configuration Docker par défaut** - Utilisation de configurations Docker non sécurisées
5. **Absence de chiffrement** - Les communications ne sont pas chiffrées

## Recommandations

### 1. Restriction d'accès réseau

Modifiez le fichier `docker-compose.yml` pour limiter l'accès aux services uniquement depuis le réseau local :

```yaml
services:
  ctm-az-core:
    # ...
    ports:
      - "127.0.0.1:8000:8000"  # Limité à localhost uniquement
    # ...

  ctm-az-monitor:
    # ...
    ports:
      - "127.0.0.1:8080:8080"  # Limité à localhost uniquement
    # ...
```

Si un accès externe est nécessaire, utilisez un proxy inverse sécurisé (Nginx, Traefik) avec TLS.

### 2. Sécurisation des secrets

Utilisez un gestionnaire de secrets d'entreprise comme HashiCorp Vault, AWS Secrets Manager ou Azure Key Vault au lieu de fichiers .env :

1. Installez un client pour votre gestionnaire de secrets
2. Modifiez le script de démarrage pour récupérer les secrets au lancement

```bash
# Exemple avec HashiCorp Vault
export CLAUDE_API_KEY=$(vault kv get -field=api_key secrets/claude)
export DEEPSEEK_API_KEY=$(vault kv get -field=api_key secrets/deepseek)
```

Alternativement, utilisez Docker Swarm ou Kubernetes avec leurs propres mécanismes de gestion des secrets.

### 3. Mise en place d'une authentification

Ajoutez une couche d'authentification devant les services :

1. Modifiez `src/monitor/app.py` pour inclure une authentification de base :

```python
# Ajoutez ces imports
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

# Initialisation de l'authentification
auth = HTTPBasicAuth()

# Configuration des utilisateurs (stockez-les de façon sécurisée en production)
users = {
    "admin": generate_password_hash("changez_ce_mot_de_passe")
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username
    return None

# Protégez les routes
@app.route('/dashboard')
@auth.login_required
def dashboard():
    # ...
```

2. Ajoutez un proxy d'authentification comme OAuth2-Proxy pour une authentification d'entreprise.

### 4. Durcissement des conteneurs Docker

Mettez à jour vos Dockerfiles pour renforcer la sécurité :

```dockerfile
# Ajouter un utilisateur non-root
RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -s /sbin/nologin -c "App User" appuser
WORKDIR /home/appuser/app

# Copier les fichiers nécessaires
COPY --chown=appuser:appuser . .

# Définir des limites de ressources strictes
USER appuser

# Activer les options de sécurité Docker
```

Dans `docker-compose.yml`, ajoutez des contraintes de sécurité :

```yaml
services:
  ctm-az-core:
    # ...
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    read_only: true
    tmpfs:
      - /tmp
    volumes:
      - ./logs:/home/appuser/app/logs:rw
      - ./data:/home/appuser/app/data:rw
    # ...
```

### 5. Mise en place du chiffrement TLS

1. Générez des certificats TLS (auto-signés ou via une autorité de certification interne)
2. Configurez un proxy inverse (Nginx, Traefik) pour terminer TLS :

```nginx
server {
    listen 443 ssl;
    server_name ctm-az.example.com;

    ssl_certificate     /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    # Configuration TLS renforcée
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
    
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 6. Audit et journalisation

1. Activez une journalisation renforcée dans `configs/production.yaml` :

```yaml
logging:
  level: "info"
  file: "logs/ctm-az.log"
  rotation: true
  max_size: 10485760  # 10 MB
  backup_count: 10
  metrics_interval: 300  # 5 minutes
  audit: true
  audit_file: "logs/audit.log"
```

2. Configurez un système de centralisation des logs (ELK, Graylog, Splunk)

### 7. Mise en place d'un pare-feu applicatif

Utilisez un pare-feu applicatif (WAF) comme ModSecurity pour protéger vos API et interfaces web.

### 8. Limitation d'accès à l'API

Ajoutez une limitation de débit et un contrôle d'accès IP dans votre configuration Docker Compose ou nginx :

```nginx
# Limitation de débit
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

server {
    # ...
    
    location /api/ {
        # Limitation à 10 requêtes par seconde
        limit_req zone=api_limit burst=20 nodelay;
        
        # Restriction par IP
        allow 10.0.0.0/8;  # Réseau interne d'entreprise
        deny all;          # Bloquer tout le reste
        
        proxy_pass http://127.0.0.1:8000;
    }
}
```

## Plan d'implémentation

1. **Immédiat (haute priorité)**
   - Limiter les ports à localhost uniquement
   - Supprimer les clés API du code et des fichiers .env après utilisation
   - Mettre en place des utilisateurs non-root dans Docker

2. **Court terme (1-2 semaines)**
   - Configurer un proxy inverse avec TLS
   - Mettre en place l'authentification de base
   - Implémenter la gestion sécurisée des secrets

3. **Moyen terme (1 mois)**
   - Migration vers Kubernetes pour une gestion plus sécurisée
   - Configuration d'un WAF
   - Intégration avec l'authentification d'entreprise
   - Centralisation des logs

4. **Long terme**
   - Audit de sécurité complet
   - Tests de pénétration
   - Certification de sécurité