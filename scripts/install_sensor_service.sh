#!/bin/bash
# Installs the SafeNet sensor as a macOS LaunchDaemon so it captures traffic
# continuously (starts at boot, restarts on failure). Run once:
#     sudo bash scripts/install_sensor_service.sh [interface]
# Remove with:
#     sudo launchctl bootout system/com.safenet.sensor && sudo rm /Library/LaunchDaemons/com.safenet.sensor.plist
set -euo pipefail
if [[ $EUID -ne 0 ]]; then echo "run with sudo"; exit 1; fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IFACE="${1:-$(route -n get default | awk '/interface:/{print $2}')}"
PLIST=/Library/LaunchDaemons/com.safenet.sensor.plist
mkdir -p "$ROOT/logs"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.safenet.sensor</string>
  <key>ProgramArguments</key><array>
    <string>$ROOT/.venv/bin/python</string><string>-m</string><string>scripts.sensor</string>
    <string>--iface</string><string>$IFACE</string>
  </array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>$ROOT/logs/sensor.log</string>
  <key>StandardErrorPath</key><string>$ROOT/logs/sensor.log</string>
</dict></plist>
EOF
chmod 644 "$PLIST"
launchctl bootout system/com.safenet.sensor 2>/dev/null || true
launchctl bootstrap system "$PLIST"
echo "SafeNet sensor installed on $IFACE. Logs: $ROOT/logs/sensor.log"
