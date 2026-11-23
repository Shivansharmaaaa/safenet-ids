# SafeNet: Explainable and Robust Network Intrusion Detection

SafeNet monitors real network traffic, groups packets into flows, and uses machine learning to detect attacks
such as DDoS, port scanning and brute force. Every alert comes with a SHAP explanation. The models are trained on
the public **CIC-IDS2017** and **UNSW-NB15** datasets and are tested **across** datasets and against
**adversarial evasion**.

```
 live packets ──► flow meter ──► 22 shared flow features ──► XGBoost (adv.-trained) ──► alert + SHAP
                                      │                       per-host LSTM (last 16 flows of the same host)
 (scapy, en0)     (5-tuple,        (same definition for      XGBoost attack type        │
                   FIN/RST,         CIC-IDS2017, UNSW-NB15     Autoencoder (anomaly)      ▼
                   timeouts)        and live traffic)                         PostgreSQL (Supabase)
                                                                                         │
                                                         Streamlit dashboard (Google sign-in) ◄┘
```

**Performance report** (generalisation, evasion resistance, live findings): [docs/REPORT.md](docs/REPORT.md)

## What is in the repo

| Path | Purpose |
|---|---|
| `safenet/features.py` | The shared flow-feature schema and the mappings from CIC-IDS2017 and UNSW-NB15 columns |
| `safenet/data.py` | Loads and cleans the real dataset releases (checks the official row counts) |
| `safenet/models.py` | Random Forest, XGBoost and a PyTorch Autoencoder (trained on benign traffic only) |
| `safenet/adversarial.py` | Realistic evasion attack (pad / slow down / inject dummy packets) and adaptive black-box search |
| `safenet/explain.py` | SHAP global importance and per-alert explanations |
| `safenet/flowmeter.py` | Live bidirectional flow meter that produces the same features as training (TSO re-split, client→server orientation) |
| `safenet/sequence.py` | Per-host LSTM over each host's last 16 flows: step features, window gathering, live `HostHistory`, expected-gradients SHAP |
| `safenet/behavior.py` | Multi-flow behavioural rules: port scans, host sweeps and brute force over a 60 s window |
| `safenet/intel.py` | Threat intel: abuse.ch Feodo Tracker, Spamhaus DROP, CISA KEV (matched locally), and ipwho.is / Shodan InternetDB / GreyNoise enrichment for alerted IPs, plus MITRE ATT&CK mapping |
| `safenet/policy.py` | Alert scope: ICMP and LAN discovery (mDNS/SSDP/NetBIOS…) are scored but cannot raise ML alerts |
| `tests/` + `.github/workflows/ci.yml` | Unit tests and GitHub Actions CI |
| `scripts/train.py` | Runs all experiments, saves the live model bundle, and writes results to the database |
| `scripts/train_sequence.py` | Trains the per-host LSTM and a same-rows XGBoost baseline; in-/cross-dataset, per-category and evasion results |
| `scripts/fetch_models.py` | Downloads the released model bundles (SHA-256 checked), so new sensors need no training data |
| `Dockerfile`, `scripts/install_sensor_linux.sh`, `requirements-sensor.txt` | Sensor for Linux hosts, gateways and mirror ports (Docker or systemd); see [docs/DEPLOY.md](docs/DEPLOY.md) |
| `scripts/sensor.py` | Live sensor: capture → score → explain → database |
| `scripts/apply_policy.py` | Re-applies the alert-scope policy to open alerts after a rule change |
| `streamlit_app.py` | Dashboard: live flows, alerts with SHAP, and the full evaluation report |
| `sql/schema.sql` | Database schema (own `safenet` schema with RLS on) |

## Experiments (`scripts/train.py`)
1. **In-dataset**: native features and shared features, RF / XGBoost / Autoencoder, with per-label detection rates
   for rare attacks (Heartbleed, Infiltration, SQL injection, Worms, …).
2. **Cross-dataset**: train on CIC-IDS2017 and test on UNSW-NB15 (and the reverse), plus training on both.
3. **Robustness**: an adaptive attacker searches pad/slow/inject perturbations of up to budget ε on
   the attacker's own packets to evade each model.
4. **Adversarial training**: XGBoost retrained with perturbed attacks. Clean accuracy is compared and robustness re-measured.

## Per-host sequence model (`scripts/train_sequence.py`)
Each flow is scored together with the same host's previous 15 flows. Four LSTM variants are compared with an XGBoost
trained on exactly the same rows: **LSTM** (all features), **LSTM-NoTiming** (the five round-trip-time features
masked, after the full LSTM missed a real internet scan because it had learned LAN speed), **LSTM-NoTiming-Adv**
(masked, plus random adversarial training) and **LSTM-NoTiming-AdvAdaptive** (trained on the strongest evasion attempt
against itself; deployed). Headline numbers, combined training:

| | CIC-IDS2017 F1 | UNSW-NB15 F1 | Attacks detected at ε = 2 (CIC / UNSW) | UNSW at ε = 1 | Real `scanme.nmap.org` scan, flows flagged | False alarms on the Mac's traffic |
|---|---|---|---|---|---|---|
| LSTM | 0.995 | 0.959 | 44 % / 32 % | 38 % | ≈ 0 % | 0.047 % |
| LSTM-NoTiming | 0.997 | 0.957 | 54 % / 25 % | 36 % | 87 % | 0.016 % |
| LSTM-NoTiming-Adv | 0.993 | 0.959 | **98 %** / 31 % | 37 % | 78 % | **0.014 %** |
| **LSTM-NoTiming-AdvAdaptive** | 0.993 | 0.960 | **98 %** / **51 %** | **97 %** | 78 % (one grouped alert) | 0.028 % |

Details, per-category results and limitations: [docs/REPORT.md §6](docs/REPORT.md).

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

The per-host sequence model needs the releases that keep IPs and timestamps:
* `data/raw/cicids2017_improved/{monday..friday}.csv`: CIC-IDS2017 re-extracted and re-labelled by KU Leuven DistriNet
  (`intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip`)
* `data/raw/unswnb15_full/UNSW-NB15_{1..4}.csv` + `NUSW-NB15_features.csv`: the full UNSW-NB15 records from UNSW's
  share linked at `research.unsw.edu.au/projects/unsw-nb15-dataset`

`.env` needs the database connection string (Supabase → Connect → Session pooler):
```
DATABASE_URL=postgresql://postgres.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres
```

## Run

```bash
.venv/bin/python -m scripts.train                       # ~10 min: experiments + live model + DB push
.venv/bin/python -m scripts.train_sequence              # ~18 min: four LSTM variants + comparison + DB push
sudo .venv/bin/python -m scripts.sensor --iface en0     # live capture (root needed for packet capture)
sudo bash scripts/install_sensor_service.sh             # or: run the sensor permanently via launchd
docker build -t safenet-sensor . && docker run --net=host --cap-add NET_RAW --cap-add NET_ADMIN \
    -e DATABASE_URL=... safenet-sensor                  # more sensors: Linux hosts, gateways, mirror ports
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
* On a second network (172.16.0.0/12), unanswered DNS queries to the local resolver scored 0.93–0.97 "Exploit". The cause is in the
  training labels: UNSW-NB15 marks 57,276 unanswered DNS flows as *Generic* attack and only 55 as normal. These flows
  are now out of scope too (still scored, stored and checked against the blocklists).
* Flows where only the server's packets were captured (QUIC after an idle split) raised low "Anomaly" alerts;
  one-sided flows are now out of scope, and flow direction comes from the TCP handshake or ephemeral ports.
* A real TCP connect scan of `scanme.nmap.org` (a host the Nmap project provides for scan testing) scored *benign*
  flow-by-flow (max 0.013): CIC-IDS2017's scans are LAN nmap SYN scans with a different signature. The behavioural
  rule caught it ("192 distinct ports in 60 s", T1046), and enrichment pulled Linode/AS63949, Shodan's 4 open ports
  and 2 CISA-KEV CVEs.

## Notes and limitations
* CICFlowMeter and Argus define features differently. The shared feature set is the intersection that
  can be computed identically from live packets, so the cross-dataset results measure real transfer, not leakage.
* UNSW-NB15's train/test CSVs carry no port numbers, so its "service" column gives the service one-hot.
* Live alerts are model output on real traffic. Treat them as triage hints, not ground truth.
