#!/bin/bash
# Installs the SafeNet sensor as a systemd service on a Linux machine (laptop, server, Raspberry Pi, gateway or a
# host on a switch mirror/SPAN port). Run from a clone of the repository:
#     sudo bash scripts/install_sensor_linux.sh [interface]
# It creates .venv, installs requirements-sensor.txt, downloads the released models and starts the service.
# DATABASE_URL must be in .env next to this repository's README. Remove with:
#     sudo systemctl disable --now safenet-sensor && sudo rm /etc/systemd/system/safenet-sensor.service
set -euo pipefail
if [[ $EUID -ne 0 ]]; then echo "run with sudo"; exit 1; fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IFACE="${1:-$(ip route show default | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')}"
if ! grep -q '^DATABASE_URL=' "$ROOT/.env" 2>/dev/null; then
  echo "put DATABASE_URL=postgresql://... in $ROOT/.env first"; exit 1
fi
command -v python3 >/dev/null || { echo "python3 (3.12 recommended) is required"; exit 1; }
python3 -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/pip" install --quiet --upgrade pip
TORCH="$(grep '^torch==' "$ROOT/requirements-sensor.txt" | cut -d= -f3)"
"$ROOT/.venv/bin/pip" install --quiet --index-url https://download.pytorch.org/whl/cpu "torch==$TORCH"   # CPU-only build
"$ROOT/.venv/bin/pip" install --quiet -r "$ROOT/requirements-sensor.txt"
(cd "$ROOT" && .venv/bin/python -m scripts.fetch_models)
mkdir -p "$ROOT/logs"
cat > /etc/systemd/system/safenet-sensor.service <<UNIT
[Unit]
Description=SafeNet network intrusion detection sensor
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=$ROOT
EnvironmentFile=$ROOT/.env
ExecStart=$ROOT/.venv/bin/python -m scripts.sensor --iface $IFACE
Restart=always
RestartSec=30
StandardOutput=append:$ROOT/logs/sensor.log
StandardError=append:$ROOT/logs/sensor.log

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now safenet-sensor
echo "SafeNet sensor running on $IFACE. Logs: $ROOT/logs/sensor.log"
