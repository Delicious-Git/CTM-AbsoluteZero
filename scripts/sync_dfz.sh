#!/bin/bash
# Script to synchronize CTM-AbsoluteZero with the DFZ system

set -e

# Configuration
PRIMARY_LOCATION="/home/delicious-linux/CTM-AbsoluteZero"
SECONDARY_LOCATION="/mnt/c/Dev/CTM-AbsoluteZero"
DFZ_LOCATION="/mnt/c/Dev/DFZ-Monorepo/evolution"
SYNC_DIRS=("data" "models" "state")
EXCLUDE_PATTERNS=("*.log" "*.tmp" "__pycache__")

# Create directories if they don't exist
echo "Creating directories if needed..."
mkdir -p "$SECONDARY_LOCATION"
for dir in "${SYNC_DIRS[@]}"; do
    mkdir -p "$PRIMARY_LOCATION/$dir"
    mkdir -p "$SECONDARY_LOCATION/$dir"
done

# Build rsync exclude options
EXCLUDE_OPTS=""
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    EXCLUDE_OPTS="$EXCLUDE_OPTS --exclude=$pattern"
done

# Sync from primary to secondary location
echo "Syncing from primary to secondary location..."
for dir in "${SYNC_DIRS[@]}"; do
    rsync -avz --delete $EXCLUDE_OPTS "$PRIMARY_LOCATION/$dir/" "$SECONDARY_LOCATION/$dir/"
done

# Copy config files
echo "Syncing configuration files..."
rsync -avz --delete $EXCLUDE_OPTS "$PRIMARY_LOCATION/configs/" "$SECONDARY_LOCATION/configs/"

# Create DFZ plugin directory if it doesn't exist
DFZ_PLUGIN_DIR="$DFZ_LOCATION/plugins/ctm_az_adapter"
mkdir -p "$DFZ_PLUGIN_DIR"

# Install CTM-AbsoluteZero as a plugin for DFZ
echo "Installing CTM-AbsoluteZero as a DFZ plugin..."
cat > "$DFZ_PLUGIN_DIR/__init__.py" << 'EOF'
"""
CTM-AbsoluteZero adapter plugin for DFZ.
This plugin allows DFZ to use CTM-AbsoluteZero for task generation and execution.
"""

import os
import sys
import importlib.util
import logging

# Add CTM-AbsoluteZero paths
CTM_AZ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'CTM-AbsoluteZero'))
sys.path.append(CTM_AZ_PATH)

try:
    # Import the DFZ plugin module from CTM-AbsoluteZero
    spec = importlib.util.spec_from_file_location(
        "dfz_plugin",
        os.path.join(CTM_AZ_PATH, "src", "integration", "dfz.py")
    )
    dfz_plugin_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dfz_plugin_module)
    
    # Export the plugin class
    CTMAbsoluteZeroPlugin = dfz_plugin_module.CTMAbsoluteZeroPlugin
    create_dfz_plugin = dfz_plugin_module.create_dfz_plugin
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=os.path.join(CTM_AZ_PATH, "logs", "dfz_plugin.log")
    )
    logger = logging.getLogger("dfz_plugin")
    logger.info("CTM-AbsoluteZero DFZ plugin loaded successfully")
    
except Exception as e:
    logging.error(f"Failed to load CTM-AbsoluteZero DFZ plugin: {e}")
    raise

# Create a default plugin instance
default_plugin = None
try:
    config_path = os.path.join(CTM_AZ_PATH, "configs", "dfz_integration.yaml")
    default_plugin = create_dfz_plugin(config_path=config_path)
    logging.info("Created default CTM-AbsoluteZero plugin instance")
except Exception as e:
    logging.error(f"Failed to create default plugin instance: {e}")
EOF

# Create a simple Windows batch file to run the sync script from Windows
echo "Creating Windows sync script..."
cat > "$SECONDARY_LOCATION/sync_with_dfz.bat" << 'EOF'
@echo off
echo Syncing CTM-AbsoluteZero with DFZ...
wsl -e bash /home/delicious-linux/CTM-AbsoluteZero/scripts/sync_dfz.sh
echo Done!
pause
EOF

echo "Creating integration run script..."
cat > "$PRIMARY_LOCATION/scripts/run_dfz_integration.sh" << 'EOF'
#!/bin/bash
# Script to run CTM-AbsoluteZero with DFZ integration

set -e

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CTM_AZ_DIR="$( dirname "$SCRIPT_DIR" )"

# Run sync first
echo "Syncing with DFZ..."
"$SCRIPT_DIR/sync_dfz.sh"

# Run the CLI with DFZ integration
echo "Starting CTM-AbsoluteZero with DFZ integration..."
cd "$CTM_AZ_DIR"
python -m src.cli --config configs/dfz_integration.yaml dfz --dfz-path "/mnt/c/Dev/DFZ-Monorepo/evolution" --interactive
EOF

# Make the run script executable
chmod +x "$PRIMARY_LOCATION/scripts/run_dfz_integration.sh"

echo "Sync complete! DFZ integration is now set up."
echo "To run CTM-AbsoluteZero with DFZ integration, use:"
echo "  $PRIMARY_LOCATION/scripts/run_dfz_integration.sh"
echo ""
echo "From Windows, you can sync by running:"
echo "  $SECONDARY_LOCATION/sync_with_dfz.bat"