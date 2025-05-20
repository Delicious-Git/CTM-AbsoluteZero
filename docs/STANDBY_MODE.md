# CTM-AbsoluteZero Standby Mode

This document describes the standby mode functionality of CTM-AbsoluteZero, including how it works, how to use it, and how to configure wake triggers.

## Overview

Standby mode allows the CTM-AbsoluteZero system to conserve resources while still being able to wake up when needed. This is particularly useful for long-running deployments where the system needs to be responsive to events without consuming resources continuously.

The standby mode system consists of several components:

1. **Standby Script**: Puts the system into standby mode
2. **Wake Triggers**: Monitors for events that should wake up the system
3. **Wake Script**: Wakes up the system when triggered

## Standby Mode

To put the system into standby mode, use the `standby.py` script:

```bash
python scripts/standby.py --config configs/production.yaml
```

This will:

1. Stop all running services
2. Clean up temporary files and old logs
3. Generate a comprehensive standby report
4. Set up wake triggers to monitor for events
5. Update the system state to "standby"

### Standby Options

```
usage: standby.py [-h] [--config CONFIG] [--state-dir STATE_DIR] [--no-report] [--log-level {debug,info,warning,error}]

CTM-AbsoluteZero Standby

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Path to configuration file
  --state-dir STATE_DIR, -s STATE_DIR
                        Directory for state information
  --no-report           Don't generate standby report
  --log-level {debug,info,warning,error}
                        Logging level
```

### Standby Report

By default, the standby script generates a detailed report of the system's state and performance before entering standby mode. These reports are stored in the `state/reports` directory and include:

- System uptime
- Task performance statistics
- Benchmark results
- System information
- Service status

## Wake Triggers

Wake triggers are conditions that, when met, will cause the system to wake up from standby mode. These triggers are configured in the configuration file and are monitored by the `wake_triggers.py` script.

### Built-in Triggers

The following triggers are built into the system:

1. **Dependencies**: Detects changes to dependencies files (`requirements.txt`)
2. **Data**: Detects new or changed data files
3. **Code**: Detects changes to the source code
4. **Schedule**: Wakes up the system at scheduled times (daily, weekly, monthly)
5. **API**: Checks an API endpoint for updates

### Configuring Triggers

Triggers are configured in the `wake_triggers` section of the configuration file:

```yaml
wake_triggers:
  dependencies:
    type: dependency
    priority: medium
    paths: 
      - requirements.txt
    enabled: true
    description: Check for updates to dependencies
  
  data:
    type: data
    priority: high
    paths: 
      - data
    enabled: true
    description: Check for new data
  
  daily:
    type: schedule
    priority: medium
    schedule: daily
    time: "00:00"
    enabled: true
    description: Daily scheduled wake
```

### Trigger Priority

Triggers can have one of three priority levels:

- **High**: Checked frequently (default: every 5 minutes)
- **Medium**: Checked less frequently (part of the regular check interval)
- **Low**: Checked least frequently (part of the regular check interval)

The check intervals can be configured in the `wake_triggers.py` script.

### Running Wake Triggers

In most cases, the wake triggers daemon is started automatically by the standby script. However, you can also run it manually:

```bash
python scripts/wake_triggers.py --daemon --config configs/production.yaml
```

This will continuously monitor for wake triggers and wake up the system when they are activated.

### Wake Trigger Options

```
usage: wake_triggers.py [-h] [--config CONFIG] [--state-dir STATE_DIR] [--daemon] [--interval INTERVAL]
                       [--high-priority-interval HIGH_PRIORITY_INTERVAL]
                       [--log-level {debug,info,warning,error}]

CTM-AbsoluteZero Wake Triggers

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Path to configuration file
  --state-dir STATE_DIR, -s STATE_DIR
                        Directory to store state information
  --daemon, -d          Run as daemon
  --interval INTERVAL, -i INTERVAL
                        Interval between trigger checks in seconds
  --high-priority-interval HIGH_PRIORITY_INTERVAL
                        Interval between high priority trigger checks in seconds
  --log-level {debug,info,warning,error}
                        Logging level
```

## Wake Process

When a wake trigger is activated, the system wakes up using the `wake.py` script:

```bash
python scripts/wake.py --config configs/production.yaml --triggers data,code
```

This will:

1. Run system tests to verify integrity
2. Start required services based on the triggers
3. Update the system state to "awake"

### Wake Options

```
usage: wake.py [-h] [--config CONFIG] [--state-dir STATE_DIR] [--triggers TRIGGERS]
              [--log-level {debug,info,warning,error}]

CTM-AbsoluteZero Wake

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Path to configuration file
  --state-dir STATE_DIR, -s STATE_DIR
                        Directory for state information
  --triggers TRIGGERS, -t TRIGGERS
                        Comma-separated list of trigger IDs
  --log-level {debug,info,warning,error}
                        Logging level
```

## Services

Services are components of the system that are started and stopped during the wake and standby processes. They are configured in the `services` section of the configuration file:

```yaml
services:
  api_server:
    cmd: ["python", "-m", "src.api.server", "--port", "8080"]
    detach: true
    priority: high
    description: API server
    triggers: ["daily", "data"]
```

Each service can specify:

- **cmd**: Command to run to start the service
- **detach**: Whether to run the service in the background
- **priority**: Service priority (affects startup order)
- **description**: Service description
- **triggers**: List of triggers that should start this service

## State Management

The system state is stored in the `state` directory:

- `system_state.json`: Overall system state
- `wake_state.json`: Wake triggers state
- `reports/`: Standby reports
- `tests/`: Test results

### System State

The system state includes:

- **status**: Current system status (awake, standby, waking, entering_standby, error)
- **last_wake**: Timestamp of last wake
- **last_standby**: Timestamp of last standby
- **components**: State of system components, including services

## Examples

### Putting the System into Standby Mode

```bash
python scripts/standby.py --config configs/production.yaml
```

### Running Wake Triggers Daemon

```bash
python scripts/wake_triggers.py --daemon --config configs/production.yaml
```

### Waking Up the System Manually

```bash
python scripts/wake.py --config configs/production.yaml --triggers manual
```

## Integration with CI/CD

You can integrate standby mode with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: Deploy and Standby

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: pytest
    - name: Deploy
      run: ./deploy.sh
    - name: Put into standby mode
      run: python scripts/standby.py --config configs/production.yaml
```

## Best Practices

1. **Configure Appropriate Triggers**: Enable only the triggers you need
2. **Set Priority Levels**: Use high priority only for critical triggers
3. **Monitor Wake Logs**: Check wake logs for false positives
4. **Regular Testing**: Periodically test the wake system to ensure it works
5. **Resource Considerations**: Wake triggers consume minimal resources, but monitor usage