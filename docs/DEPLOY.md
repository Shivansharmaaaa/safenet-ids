# Deploying SafeNet sensors

Every sensor captures packets on one network interface, scores the flows with the released models and writes them
to the same PostgreSQL database. The dashboard (https://safenet-ids.streamlit.app) shows all sensors together, or
one at a time via the **Sensor** picker in the sidebar. Its **Sensors** page lists each sensor's status.

## What one sensor can see

A sensor sees the traffic that reaches its own interface.

| Where the sensor runs | What it sees |
|---|---|
| A laptop or server on Wi-Fi or a switch port | Only that machine's own traffic. Other stations' Wi-Fi frames are encrypted with their own keys, and a switch only forwards frames meant for this port. |
| The **gateway** (a Linux router, or a Raspberry Pi set up to route the LAN) | All traffic between the LAN and the internet, for every device |
| A host plugged into a switch **mirror / SPAN port** | Everything the switch copies to that port, usually a whole VLAN, including LAN-to-LAN traffic |

To cover a network: put one sensor on the gateway or a mirror port, or run a sensor on each machine you care about.
Hosts are identified by their IP, so the per-host LSTM builds one history per device the sensor sees.

## Requirements

* A `DATABASE_URL` for the project's Postgres. Supabase → Connect → **Session pooler** string, IPv4.
* Root privileges or `CAP_NET_RAW` + `CAP_NET_ADMIN` for packet capture.
* No training data. The models are downloaded from the GitHub release
  [`models-20261117`](https://github.com/Shivansharmaaaa/safenet-ids/releases/tag/models-20261117) and checked against
  the published SHA-256 sums (`python -m scripts.fetch_models`).

## Option 1: Docker (any Linux host, gateway or mirror-port box)

```bash
git clone https://github.com/Shivansharmaaaa/safenet-ids && cd safenet-ids
docker build -t safenet-sensor .        # add --build-arg MODELS_REFRESH=$(date +%s) after a model update
docker run -d --name safenet-sensor --restart unless-stopped --net=host \
    --cap-add NET_RAW --cap-add NET_ADMIN -e DATABASE_URL='postgresql://…' safenet-sensor --iface eth0
docker logs -f safenet-sensor
```

`--net=host` is what lets the container see the host's interfaces. On Docker Desktop or Colima on a Mac, "host" is
the Linux VM, so the sensor sees the VM's own traffic, not the Mac's. Use the LaunchDaemon there.

## Option 2: Linux systemd service

```bash
git clone https://github.com/Shivansharmaaaa/safenet-ids && cd safenet-ids
echo "DATABASE_URL=postgresql://…" > .env && chmod 600 .env
sudo bash scripts/install_sensor_linux.sh eth0      # venv, pinned requirements, model download, service
journalctl -u safenet-sensor -f                     # or: tail -f logs/sensor.log
```

## Option 3: macOS LaunchDaemon

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-sensor.txt && .venv/bin/python -m scripts.fetch_models
echo "DATABASE_URL=postgresql://…" > .env && chmod 600 .env
sudo bash scripts/install_sensor_service.sh en0
```

If the project lives in a protected folder (Desktop, Documents), grant **Full Disk Access** to both
`.venv/bin/python3.12` and the `Python.app` it links to (System Settings → Privacy & Security).

## Operating notes

* Each sensor heartbeats every few seconds. The dashboard shows a sensor as offline after 90 s of silence.
* Sensors restart themselves when their model files, `.env` or code change, and when packet capture or the
  database connection fails. The service manager brings them back.
* Every sensor prunes flows that are not linked to an alert after 3 days, and caps them at 300,000 rows, so the
  free 500 MB Supabase tier is not exhausted. The cap applies to the shared table, so with many busy sensors,
  raise the plan or lower `keep_days`.
