# SafeNet: Explainable and Robust Network Intrusion Detection

SafeNet monitors real network traffic, groups packets into flows, and uses machine learning to detect attacks
such as DDoS, port scanning and brute force. Every alert comes with a SHAP explanation. The models are trained on
the public **CIC-IDS2017** and **UNSW-NB15** datasets and are tested **across** datasets and against
**adversarial evasion**.

```
 live packets ──► flow meter ──► 22 shared flow features ──► XGBoost (adv.-trained) ──► alert + SHAP
 (scapy, en0)     (5-tuple,        (same definition for      XGBoost attack type        │
                   FIN/RST,         CIC-IDS2017, UNSW-NB15     Autoencoder (anomaly)      ▼
                   timeouts)        and live traffic)                         PostgreSQL (Supabase)
                                                                                         │
                                                         Streamlit dashboard (Google sign-in) ◄┘
```

## What is in the repo

| Path | Purpose |
|---|---|
| `safenet/features.py` | The shared flow-feature schema and the mappings from CIC-IDS2017 and UNSW-NB15 columns |
| `safenet/data.py` | Loads and cleans the real dataset releases (checks the official row counts) |
| `safenet/models.py` | Random Forest, XGBoost and a PyTorch Autoencoder (trained on benign traffic only) |
| `safenet/adversarial.py` | Realistic evasion attack (pad / slow down / inject dummy packets) and adaptive black-box search |
| `safenet/explain.py` | SHAP global importance and per-alert explanations |
| `safenet/flowmeter.py` | Live bidirectional flow meter that produces the same features as training (TSO re-split, client→server orientation) |
| `safenet/behavior.py` | Multi-flow behavioural rules: port scans, host sweeps and brute force over a 60 s window |
| `safenet/intel.py` | Threat intel: abuse.ch Feodo Tracker, Spamhaus DROP, CISA KEV (matched locally), and ipwho.is / Shodan InternetDB / GreyNoise enrichment for alerted IPs, plus MITRE ATT&CK mapping |
| `safenet/policy.py` | Alert scope: ICMP and LAN discovery (mDNS/SSDP/NetBIOS…) are scored but cannot raise ML alerts |
| `tests/` + `.github/workflows/ci.yml` | Unit tests and GitHub Actions CI |
| `scripts/train.py` | Runs all experiments, saves the live model bundle, and writes results to the database |
| `scripts/sensor.py` | Live sensor: capture → score → explain → database |
| `streamlit_app.py` | Dashboard: live flows, alerts with SHAP, and the full evaluation report |
| `sql/schema.sql` | Database schema (own `safenet` schema with RLS on) |

## Experiments (`scripts/train.py`)
1. **In-dataset**: native features and shared features, RF / XGBoost / Autoencoder, with per-label detection rates
   for rare attacks (Heartbleed, Infiltration, SQL injection, Worms, …).
2. **Cross-dataset**: train on CIC-IDS2017 and test on UNSW-NB15 (and the reverse), plus training on both.
3. **Robustness**: an adaptive attacker searches pad/slow/inject perturbations of up to budget ε on
   the attacker's own packets to evade each model.
4. **Adversarial training**: XGBoost retrained with perturbed attacks. Clean accuracy is compared and robustness re-measured.

## Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-ml.txt
brew install libomp            # OpenMP runtime needed by XGBoost on macOS
```

Datasets (not committed) go in `data/raw/`:
* `data/raw/cicids2017/*.pcap_ISCX.csv`: the 8 files of the official CIC-IDS2017 *MachineLearningCSV* release
  (UNB CIC; mirrored unchanged at `huggingface.co/datasets/c01dsnap/CIC-IDS2017`)
* `data/raw/unswnb15/{train,test}.csv`: the official UNSW-NB15 training/testing partitions
  (UNSW Canberra; mirrored at `huggingface.co/datasets/Mireu-Lab/UNSW-NB15`)

`.env` needs the database connection string (Supabase → Connect → Session pooler):
```
DATABASE_URL=postgresql://postgres.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres
```

## Run

```bash
.venv/bin/python -m scripts.train                       # ~30-60 min: experiments + live model + DB push
sudo .venv/bin/python -m scripts.sensor --iface en0     # live capture (root needed for packet capture)
sudo bash scripts/install_sensor_service.sh             # or: run the sensor permanently via launchd
.venv/bin/streamlit run streamlit_app.py                # dashboard locally
```

## Dashboard deployment (Streamlit Community Cloud)
Live at **https://safenet-ids.streamlit.app** (Google sign-in, allow-listed accounts only). Main file `streamlit_app.py`. Secrets (App → Settings → Secrets):
```toml
DATABASE_URL = "postgresql://..."
ALLOWED_EMAILS = "you@gmail.com,other@gmail.com"

[auth]
redirect_uri = "https://<app>.streamlit.app/oauth2callback"
cookie_secret = "<random string>"
client_id = "<Google OAuth client id>"
client_secret = "<Google OAuth client secret>"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

## Live results so far (real traffic on the author's Mac)
* TCP traffic: no false alerts across the first ~1,000 live TCP flows.
* ML false positives seen on protocols absent from both datasets: ICMP, mDNS/SSDP discovery, QUIC and STUN (WebRTC).
  The first three are now out of scope by policy; STUN alerts are kept as an honest record of the generalisation gap.
* A real TCP connect scan of `scanme.nmap.org` (a host the Nmap project provides for scan testing) scored *benign*
  flow-by-flow (max 0.013): CIC-IDS2017's scans are LAN nmap SYN scans with a different signature. The behavioural
  rule caught it ("192 distinct ports in 60 s", T1046), and enrichment pulled Linode/AS63949, Shodan's 4 open ports
  and 2 CISA-KEV CVEs.

## Notes and limitations
* CICFlowMeter and Argus define features differently. The shared feature set is the intersection that
  can be computed identically from live packets, so the cross-dataset results measure real transfer, not leakage.
* UNSW-NB15's train/test CSVs carry no port numbers, so its "service" column gives the service one-hot.
* Live alerts are model output on real traffic. Treat them as triage hints, not ground truth.
