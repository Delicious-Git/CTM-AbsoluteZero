"""
Authentication module for CTM-AbsoluteZero monitor.
"""
import os
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize authentication
auth = HTTPBasicAuth()

# Get credentials from environment variables or use defaults
default_username = os.environ.get("AUTH_USERNAME", "admin")
default_password = os.environ.get("AUTH_PASSWORD", "changez_ce_mot_de_passe")

# Configure users (ideally, in a production environment, this would be stored in a secure database)
users = {
    default_username: generate_password_hash(default_password)
}

@auth.verify_password
def verify_password(username, password):
    """Verify username and password."""
    if username in users and check_password_hash(users.get(username), password):
        return username
    return None