"""
CTM-AbsoluteZero Monitoring Dashboard Application

This module implements a Dash-based web dashboard for monitoring the 
CTM-AbsoluteZero system in real-time.
"""
import os
import sys
import time
import json
import logging
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from flask import request

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from flask_httpauth import HTTPBasicAuth

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import utilities
from src.utils.logging import configure_logging, get_logger
from src.utils.config import ConfigManager, load_config
from src.monitor.auth import auth

# Configure logging
logger = get_logger("ctm-az.monitor")

# Constants
REFRESH_INTERVAL = 5000  # Dashboard refresh interval in milliseconds
DEFAULT_CORE_API_URL = "http://localhost:8000"
CORE_API_URL = os.environ.get("CORE_API_URL", DEFAULT_CORE_API_URL)
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
DASH_DEBUG = os.environ.get("DASH_DEBUG", "false").lower() == "true"

# Configure logging
configure_logging(
    log_level=getattr(logging, LOG_LEVEL, logging.INFO),
    log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file="logs/monitor.log"
)

# Initialize app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="CTM-AbsoluteZero Monitor",
    suppress_callback_exceptions=True
)

# For gunicorn
server = app.server

# Create health check endpoint (no auth required)
@server.route("/health")
def health_check():
    from flask import jsonify
    return jsonify({"status": "healthy"})

# Protect all Dash routes with authentication
@server.before_request
def before_request():
    if request.path.startswith("/health"):
        # Skip authentication for health check endpoint
        return
    
    from flask import request
    if not request.path.startswith("/assets/") and not request.path.startswith("/_favicon"):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth.authenticate(auth_header, None):
            return auth.auth_error_callback()

# API interaction functions
def fetch_metrics() -> Dict[str, Any]:
    """Fetch metrics from the core API."""
    try:
        url = f"{CORE_API_URL}/api/metrics"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to fetch metrics: {response.status_code}")
            return {}
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        return {}

def fetch_tasks(limit: int = 20) -> List[Dict[str, Any]]:
    """Fetch recent tasks from the core API."""
    try:
        url = f"{CORE_API_URL}/api/tasks?limit={limit}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to fetch tasks: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error fetching tasks: {e}")
        return []

def fetch_domains() -> List[str]:
    """Fetch available domains from the core API."""
    try:
        url = f"{CORE_API_URL}/api/domains"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to fetch domains: {response.status_code}")
            return ["general", "maze", "quantum", "sorting"]
    except Exception as e:
        logger.error(f"Error fetching domains: {e}")
        return ["general", "maze", "quantum", "sorting"]

# Create dashboard layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("CTM-AbsoluteZero Monitor", className="text-center my-4"),
            html.P("Real-time monitoring of the CTM-AbsoluteZero system", className="text-center mb-4"),
            dcc.Interval(id='interval-component', interval=REFRESH_INTERVAL, n_intervals=0),
        ], width=12)
    ]),
    
    # System status
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("System Status"),
                dbc.CardBody([
                    html.Div(id="system-status")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Performance metrics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Performance Metrics"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Success Rate", className="card-title"),
                                    html.H2(id="success-rate", className="card-text text-center")
                                ])
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Tasks Completed", className="card-title"),
                                    html.H2(id="tasks-completed", className="card-text text-center")
                                ])
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Avg Reward", className="card-title"),
                                    html.H2(id="avg-reward", className="card-text text-center")
                                ])
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Current Phase", className="card-title"),
                                    html.H2(id="current-phase", className="card-text text-center")
                                ])
                            ])
                        ], width=3)
                    ])
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Success Rate by Domain"),
                dbc.CardBody([
                    dcc.Graph(id="domain-success-chart")
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Task Execution Time"),
                dbc.CardBody([
                    dcc.Graph(id="task-time-chart")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Recent tasks
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Recent Tasks"),
                dbc.CardBody([
                    html.Div(id="recent-tasks-table")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P([
                "CTM-AbsoluteZero Monitor Dashboard • ",
                html.Span(id="current-time"),
                " • ",
                html.A("Documentation", href="#", className="text-decoration-none")
            ], className="text-center text-muted")
        ], width=12)
    ])
], fluid=True)

# Callbacks
@app.callback(
    [
        Output("system-status", "children"),
        Output("success-rate", "children"),
        Output("tasks-completed", "children"),
        Output("avg-reward", "children"),
        Output("current-phase", "children"),
        Output("domain-success-chart", "figure"),
        Output("task-time-chart", "figure"),
        Output("recent-tasks-table", "children"),
        Output("current-time", "children")
    ],
    [Input("interval-component", "n_intervals")]
)
def update_dashboard(n_intervals):
    """Update all dashboard components."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Fetch data
    metrics = fetch_metrics()
    tasks = fetch_tasks(limit=10)
    
    # System status
    if metrics:
        status_color = "success"
        status_text = "Operational"
    else:
        status_color = "danger"
        status_text = "Unavailable"
    
    system_status = dbc.Alert(
        f"System is {status_text}",
        color=status_color,
        className="text-center mb-0"
    )
    
    # Performance metrics
    success_rate = f"{metrics.get('success_rate', 0) * 100:.1f}%" if metrics else "N/A"
    tasks_completed = str(metrics.get("total_tasks", 0)) if metrics else "N/A"
    avg_reward = f"{metrics.get('avg_reward', 0):.2f}" if metrics else "N/A"
    current_phase = str(metrics.get("current_phase", "Unknown")).capitalize() if metrics else "N/A"
    
    # Domain success chart
    domain_metrics = metrics.get("domain_metrics", {}) if metrics else {}
    domains = []
    success_rates = []
    
    for domain, domain_data in domain_metrics.items():
        if domain_data.get("total", 0) > 0:
            domains.append(domain)
            rate = domain_data.get("successful", 0) / domain_data.get("total", 1)
            success_rates.append(rate * 100)
    
    if domains:
        domain_fig = px.bar(
            x=domains,
            y=success_rates,
            labels={"x": "Domain", "y": "Success Rate (%)"},
            color=success_rates,
            color_continuous_scale="viridis"
        )
        domain_fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    else:
        domain_fig = go.Figure()
        domain_fig.update_layout(
            title="No domain data available",
            xaxis=dict(title="Domain"),
            yaxis=dict(title="Success Rate (%)"),
            margin=dict(l=40, r=40, t=40, b=40)
        )
    
    # Task execution time chart
    if tasks:
        task_times = []
        task_domains = []
        task_success = []
        
        for task in tasks:
            task_times.append(task.get("duration", 0))
            task_domains.append(task.get("domain", "unknown"))
            task_success.append("Success" if task.get("success", False) else "Failure")
        
        task_df = pd.DataFrame({
            "Domain": task_domains,
            "Duration": task_times,
            "Status": task_success
        })
        
        time_fig = px.scatter(
            task_df,
            x="Domain",
            y="Duration",
            color="Status",
            size=[10] * len(task_df),
            hover_data=["Duration"],
            color_discrete_map={"Success": "green", "Failure": "red"}
        )
        time_fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    else:
        time_fig = go.Figure()
        time_fig.update_layout(
            title="No task execution data available",
            xaxis=dict(title="Domain"),
            yaxis=dict(title="Execution Time (s)"),
            margin=dict(l=40, r=40, t=40, b=40)
        )
    
    # Recent tasks table
    if tasks:
        table_header = [
            html.Thead(html.Tr([
                html.Th("ID"),
                html.Th("Domain"),
                html.Th("Description"),
                html.Th("Status"),
                html.Th("Score"),
                html.Th("Time (s)")
            ]))
        ]
        
        rows = []
        for task in tasks:
            task_id = task.get("task_id", "")
            if len(task_id) > 8:
                task_id = task_id[:8] + "..."
            
            description = task.get("description", "")
            if len(description) > 50:
                description = description[:50] + "..."
            
            status = "Success" if task.get("success", False) else "Failure"
            status_badge = dbc.Badge(
                status,
                color="success" if status == "Success" else "danger",
                className="ms-1"
            )
            
            rows.append(html.Tr([
                html.Td(task_id),
                html.Td(task.get("domain", "")),
                html.Td(description),
                html.Td(status_badge),
                html.Td(f"{task.get('score', 0):.2f}"),
                html.Td(f"{task.get('duration', 0):.2f}")
            ]))
        
        table_body = [html.Tbody(rows)]
        recent_tasks_table = dbc.Table(
            table_header + table_body,
            bordered=True,
            hover=True,
            responsive=True,
            striped=True,
        )
    else:
        recent_tasks_table = html.P("No recent tasks available")
    
    return (
        system_status,
        success_rate,
        tasks_completed,
        avg_reward,
        current_phase,
        domain_fig,
        time_fig,
        recent_tasks_table,
        current_time
    )

# Run the app
if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 8080))
        logger.info(f"Starting monitor dashboard on port {port}")
        app.run_server(debug=DASH_DEBUG, host="0.0.0.0", port=port)
    except Exception as e:
        logger.error(f"Error starting monitor dashboard: {e}")