# CTM-AbsoluteZero and DFZ Integration Guide

This guide explains how to integrate CTM-AbsoluteZero with the DFZ conversational intelligence system across Linux (WSL) and Windows environments.

## Overview

The setup allows CTM-AbsoluteZero to function as a plugin for DFZ, enabling:
- Task generation based on conversational context
- Execution of tasks generated from conversations
- Knowledge sharing between systems
- Seamless operation across both Linux and Windows environments

## System Locations

- **CTM-AbsoluteZero Primary**: `/home/delicious-linux/CTM-AbsoluteZero` (WSL)
- **CTM-AbsoluteZero Secondary**: `/mnt/c/Dev/CTM-AbsoluteZero` (Windows)
- **DFZ System**: `/mnt/c/Dev/DFZ-Monorepo/evolution` (Windows)

## Initial Setup

1. Run the setup script to configure all necessary components:

```bash
/home/delicious-linux/CTM-AbsoluteZero/scripts/setup_windows_integration.sh
```

This script will:
- Create all required directories
- Install the CTM-AbsoluteZero plugin in the DFZ system
- Set up automatic synchronization between environments
- Create convenience scripts for both Windows and Linux

## Manual Synchronization

To manually synchronize the systems:

**From Linux (WSL):**
```bash
/home/delicious-linux/CTM-AbsoluteZero/scripts/sync_dfz.sh
```

**From Windows:**
```
C:\Dev\CTM-AbsoluteZero\sync_with_dfz.bat
```

## Running the Integrated System

To run CTM-AbsoluteZero with DFZ integration:

```bash
/home/delicious-linux/CTM-AbsoluteZero/scripts/run_dfz_integration.sh
```

This will:
1. Synchronize the latest changes
2. Start CTM-AbsoluteZero in DFZ integration mode
3. Connect to the DFZ system for conversational intelligence

## Configuration

The integration configuration is stored in:
```
/home/delicious-linux/CTM-AbsoluteZero/configs/dfz_integration.yaml
```

Key configuration options:
- `dfz.dfz_path`: Path to the DFZ installation
- `sync.sync_interval`: How often files are synchronized (in seconds)
- `sync.shared_directories`: Which directories are kept in sync
- `agent.*`: Configuration for the CTM-AbsoluteZero agent

## Troubleshooting

### Connection Issues

If CTM-AbsoluteZero cannot connect to DFZ:

1. Verify both systems are running
2. Check the paths in the configuration file
3. Review the logs at `/home/delicious-linux/CTM-AbsoluteZero/logs/dfz_integration.log`

### Synchronization Issues

If files are not synchronizing properly:

1. Check if the rsync process is running:
   ```bash
   ps aux | grep sync_dfz.sh
   ```
2. Verify the cron job is active:
   ```bash
   crontab -l | grep sync_dfz
   ```
3. Run the sync script manually to see any errors

### Plugin Loading Issues

If the DFZ system cannot load the CTM-AbsoluteZero plugin:

1. Check if the plugin directory exists:
   ```bash
   ls -la /mnt/c/Dev/DFZ-Monorepo/evolution/plugins/ctm_az_adapter
   ```
2. Verify the plugin's `__init__.py` file contains the correct paths
3. Check the DFZ logs for plugin loading errors

## Advanced Usage

### Custom Integration

You can customize the integration by editing:
- `/home/delicious-linux/CTM-AbsoluteZero/src/integration/dfz.py` - Core integration code
- `/home/delicious-linux/CTM-AbsoluteZero/configs/dfz_integration.yaml` - Integration configuration

### Adding New Features

When adding new features to either system:

1. Implement the feature in the primary location
2. Run the sync script to propagate changes
3. Update the integration code if necessary to support the new feature

## Security Notes

- The integration shares files between environments, but does not expose network services by default
- The WebSocket interface (if enabled) should be secured in production environments
- API keys and secrets should be stored in environment variables, not in the configuration files