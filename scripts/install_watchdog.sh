#!/bin/bash
set -e

SERVICE_FILE="scripts/sim00-watchdog.service"
DEST="/etc/systemd/system/sim00-watchdog.service"

if [ ! -f "$SERVICE_FILE" ]; then
    echo "Error: $SERVICE_FILE not found. Run from project root."
    exit 1
fi

echo "Installing SIM00 watchdog service..."
sudo cp "$SERVICE_FILE" "$DEST"
sudo systemctl daemon-reload
sudo systemctl enable sim00-watchdog
sudo systemctl start sim00-watchdog

echo ""
echo "Watchdog installed and running."
echo "  Status:  sudo systemctl status sim00-watchdog"
echo "  Logs:    sudo journalctl -u sim00-watchdog -f"
echo "  Stop:    sudo systemctl stop sim00-watchdog"
echo "  Restart: sudo systemctl restart sim00-watchdog"
