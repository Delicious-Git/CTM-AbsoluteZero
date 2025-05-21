"""
API endpoints for CTM-AbsoluteZero monitor to interact with the core system.
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional
from flask import Flask, jsonify, request, Blueprint

# Import utilities
from src.utils.logging import get_logger

logger = get_logger("ctm-az.monitor.api")

# Create API blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Routes
@api_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """Get system metrics."""
    # In a real implementation, this would fetch data from the core system
    # For now, we'll return a placeholder
    return jsonify({
        "total_tasks": 0,
        "successful_tasks": 0,
        "failed_tasks": 0,
        "success_rate": 0,
        "avg_reward": 0,
        "current_phase": "exploration",
        "domain_metrics": {}
    })

@api_bp.route('/tasks', methods=['GET'])
def get_tasks():
    """Get recent tasks."""
    limit = request.args.get('limit', 10, type=int)
    # In a real implementation, this would fetch data from the core system
    # For now, we'll return a placeholder
    return jsonify([])

@api_bp.route('/domains', methods=['GET'])
def get_domains():
    """Get available domains."""
    # In a real implementation, this would fetch data from the core system
    # For now, we'll return a placeholder
    return jsonify(["general", "maze", "quantum", "sorting"])

def register_api(app: Flask) -> None:
    """Register API blueprints with the Flask app."""
    app.register_blueprint(api_bp)
    logger.info("API endpoints registered")