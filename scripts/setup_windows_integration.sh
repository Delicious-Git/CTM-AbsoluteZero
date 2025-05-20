#!/bin/bash
# Script to set up Windows integration prerequisites

set -e

echo "Setting up CTM-AbsoluteZero to DFZ integration for Windows..."

# Create secondary location directory
SECONDARY_LOCATION="/mnt/c/Dev/CTM-AbsoluteZero"
mkdir -p "$SECONDARY_LOCATION"

# Create DFZ plugins directory if it doesn't exist
DFZ_LOCATION="/mnt/c/Dev/DFZ-Monorepo/evolution"
DFZ_PLUGIN_DIR="$DFZ_LOCATION/plugins"
mkdir -p "$DFZ_PLUGIN_DIR"

# Ensure rsync is installed
if ! command -v rsync &> /dev/null; then
    echo "Installing rsync..."
    sudo apt-get update
    sudo apt-get install -y rsync
fi

# Create directories
mkdir -p /home/delicious-linux/CTM-AbsoluteZero/logs
mkdir -p /home/delicious-linux/CTM-AbsoluteZero/data
mkdir -p /home/delicious-linux/CTM-AbsoluteZero/models
mkdir -p /home/delicious-linux/CTM-AbsoluteZero/state

# Run the sync script for the first time
echo "Running initial sync..."
/home/delicious-linux/CTM-AbsoluteZero/scripts/sync_dfz.sh

# Create an autorun script for Windows startup folder
echo "Creating Windows autorun script..."
STARTUP_DIR="/mnt/c/Users/$(whoami)/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup"
mkdir -p "$STARTUP_DIR"

cat > "$STARTUP_DIR/CTM-DFZ-AutoSync.bat" << 'EOF'
@echo off
start /min wsl -e bash -c "cd /home/delicious-linux/CTM-AbsoluteZero && ./scripts/sync_dfz.sh"
EOF

echo "Setting up cron job for regular sync..."
# Create a temporary file for the new crontab
TEMP_CRONTAB=$(mktemp)

# Get existing crontab content
crontab -l > "$TEMP_CRONTAB" 2>/dev/null || true

# Add the sync job if it's not already there
if ! grep -q "sync_dfz.sh" "$TEMP_CRONTAB"; then
    echo "# CTM-DFZ sync - runs every 5 minutes" >> "$TEMP_CRONTAB"
    echo "*/5 * * * * /home/delicious-linux/CTM-AbsoluteZero/scripts/sync_dfz.sh > /home/delicious-linux/CTM-AbsoluteZero/logs/sync.log 2>&1" >> "$TEMP_CRONTAB"
    # Install the new crontab
    crontab "$TEMP_CRONTAB"
    echo "Cron job added successfully"
else
    echo "Cron job already exists, skipping"
fi

# Clean up
rm "$TEMP_CRONTAB"

echo ""
echo "Setup complete! The systems will now stay in sync automatically."
echo ""
echo "You can manually sync at any time by running:"
echo "  /home/delicious-linux/CTM-AbsoluteZero/scripts/sync_dfz.sh"
echo ""
echo "To run with DFZ integration:"
echo "  /home/delicious-linux/CTM-AbsoluteZero/scripts/run_dfz_integration.sh"
echo ""
echo "From Windows, you can sync by running:"
echo "  C:\\Dev\\CTM-AbsoluteZero\\sync_with_dfz.bat"