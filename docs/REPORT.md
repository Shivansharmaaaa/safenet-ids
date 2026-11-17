# SafeNet: performance report on generalisation and evasion resistance

Model version `1.0.0+202611171028` (trained 2026-11-17). All numbers come from `scripts/train.py` run on the
official dataset releases (raw tables: [`artifacts/reports/report.md`](../artifacts/reports/report.md),
machine-readable: [`results.json`](../artifacts/reports/results.json)) and from the live sensor's database.
Live dashboard: https://safenet-ids.streamlit.app.

## 1. Summary

* **Within one dataset the problem is close to solved.** XGBoost reaches F1 0.997 on CIC-IDS2017 and 0.901 on
  UNSW-NB15, and detects even the rarest attacks (Heartbleed 3/3, SQL injection 6/6, Infiltration 10/11, Worms 44/44).
* **Across datasets it is not.** A model trained on CIC-IDS2017 detects 0.2 % of UNSW-NB15 attacks (F1 0.004);
  the reverse direction flags more than half of all benign CIC traffic (FPR 0.52). High in-dataset scores
  mostly measure how well a model learns *one* testbed.
* **Standard models are easy to evade.** An attacker who only pads, slows down or pads out *their own* packets
  cuts XGBoost's detection rate from 99.4 % to 62.7 % (budget ε = 1). Adversarial training keeps it at 97.0 %,
  with no loss of clean accuracy.
* **Running on real traffic exposed problems no benchmark score shows.** Three alert floods on a real Mac turned out
  to be artifacts of the datasets or of flow measurement, not attacks (section 6). Each was traced with SHAP and
  the training data, then handled by an explicit, documented scope policy rather than by tuning thresholds.

## 2. Setup

| | CIC-IDS2017 | UNSW-NB15 |
|---|---|---|
| Source | UNB CIC, *MachineLearningCSV* (8 days) | UNSW Canberra, official train/test partitions |
| Flow tool | CICFlowMeter | Argus + Bro |
| Rows used | 2,520,798 unique flows → 898,019 train / 756,240 test | 175,341 train / 82,332 test (official split) |
| Attack families | DoS/DDoS, PortScan, brute force (FTP/SSH), web attacks, botnet, infiltration, Heartbleed | Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms |

The two datasets use different flow tools and incompatible feature definitions. SafeNet therefore defines **22
shared features** (packet and byte counts per direction, mean packet sizes, inter-arrival times, duration, rates, byte
ratio, initial TCP windows, service one-hot) that can be computed *identically* from either dataset and from live
packets (`safenet/features.py`, `safenet/flowmeter.py`). Every cross-dataset and live result uses this feature space,
so a transfer result cannot be inflated by tool-specific columns.

Models: Random Forest (150 trees), XGBoost (400 trees, depth 8), and a PyTorch autoencoder trained **only on benign
flows**, which flags a flow when its reconstruction error exceeds the 99th percentile of benign error. All inputs
are log-scaled, and class imbalance is handled with square-root-damped class weights.

## 3. In-dataset performance

| Dataset | Features | Model | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|---|
| CIC-IDS2017 | native | XGBoost | 0.995 | 1.000 | **0.997** | 0.0010 |
| CIC-IDS2017 | shared (22) | XGBoost | 0.983 | 1.000 | 0.991 | 0.0035 |
| CIC-IDS2017 | shared (22) | Autoencoder | 0.917 | 0.553 | 0.690 | 0.0102 |
| UNSW-NB15 | native | XGBoost | 0.835 | 0.979 | **0.901** | 0.2366 |
| UNSW-NB15 | shared (22) | XGBoost | 0.852 | 0.962 | 0.903 | 0.2054 |
| UNSW-NB15 | shared (22) | Autoencoder | 0.964 | 0.442 | 0.606 | 0.0201 |

* Reducing CIC-IDS2017 to the 22 shared features costs only 0.006 F1, so the shared space keeps almost all of the signal.
* UNSW-NB15 is much harder. Its official test partition has a different traffic mix from the training partition,
  and 20–24 % of its normal flows are flagged by every supervised model.
* **Rare attacks** (XGBoost, native features): Heartbleed 3/3, SQL injection 6/6, Infiltration 10/11, XSS 99 %,
  Bot 95 %. On UNSW-NB15: Worms 44/44, Backdoor 100 %, Shellcode 98.9 %, Analysis 95.9 %; the weakest is Fuzzers
  at 85.8 %.
* The autoencoder (shared features) needs no attack labels, which is why it misses attacks that look like normal
  traffic: port scans 1.9 %, bots 2.6 %, web attacks 0 %. It does catch floods and unusual flows: FTP-Patator
  99.9 %, SSH-Patator 92 %, DoS Hulk 84 %, Heartbleed 3/3.

**What the native models rely on (mean |SHAP|).** On CIC-IDS2017 the top feature is *Destination Port*. On
UNSW-NB15 it is *sttl*, the source TTL, which is known to separate attack hosts from normal hosts in the UNSW
testbed. Both are shortcuts tied to how each lab was set up, not to attack behaviour, and section 4 shows they don't
transfer. With the shared features, CIC models rely on TCP windows and HTTP service, UNSW models on byte sizes.

## 4. Generalisation across datasets (shared features)

| Train → Test | Model | Precision | Recall | F1 | FPR | ROC-AUC |
|---|---|---|---|---|---|---|
| CIC → UNSW | Random Forest | 1.000 | 0.0002 | 0.0005 | 0.000 | 0.57 |
| CIC → UNSW | XGBoost | 0.206 | 0.002 | **0.004** | 0.009 | 0.31 |
| CIC → UNSW | Autoencoder | 0.681 | 0.582 | **0.627** | 0.335 | 0.71 |
| UNSW → CIC | Random Forest | 0.182 | 0.602 | 0.279 | 0.550 | 0.51 |
| UNSW → CIC | XGBoost | 0.189 | 0.596 | 0.287 | **0.520** | 0.52 |
| UNSW → CIC | Autoencoder | 0.268 | 0.975 | 0.420 | 0.541 | 0.84 |
| both → CIC | XGBoost | 0.985 | 0.998 | 0.992 | 0.003 | 1.00 |
| both → UNSW | XGBoost | 0.799 | 0.980 | 0.880 | 0.303 | 0.98 |

Findings:

1. **The supervised models do not transfer.** CIC→UNSW XGBoost has ROC-AUC 0.31, *worse than chance*: what it
   learned as "attack" on one testbed tends to look like *normal* traffic on the other. In the reverse direction,
   models trained on UNSW flag more than half of CIC's benign traffic.
2. **The benign-only autoencoder transfers best** (CIC→UNSW F1 0.63 against 0.004 for XGBoost; UNSW→CIC AUC 0.84).
   Learning what normal traffic looks like generalises better than learning one testbed's attacks. The cost is
   a high false-positive rate on the unseen network.
3. **Training on both datasets fixes in-distribution performance but not the underlying problem.** The combined
   XGBoost matches each single-dataset model on its own test set, but UNSW-NB15's FPR stays at 0.30. Data from the
   target network is what matters.
4. Consequence for deployment: the live detector is the combined, adversarially trained XGBoost, backed by the
   autoencoder, multi-flow behavioural rules and threat intelligence. Section 6 shows which live alerts can be trusted.

## 5. Evasion resistance

**Threat model** (`safenet/adversarial.py`). The attacker controls only the packets they send. They can pad payloads
(up to the 1500-byte MTU), slow the attack down (longer duration and gaps), and inject 60-byte dummy packets. They
cannot change the victim's replies. All derived features are recomputed, so every perturbed flow could really occur
on the wire. The attack is **adaptive and black-box**: for each malicious flow it tries 4×4×4 = 64 combinations
within budget ε and keeps the one the target model scores lowest. 30,000 attack flows were drawn from the combined
test set. The table shows the detection rate at threshold 0.5.

| Model | ε = 0 | 0.25 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|
| Random Forest | 99.5 % | 79.5 % | 75.9 % | 66.1 % | 54.6 % |
| XGBoost | 99.4 % | 75.4 % | 74.0 % | 62.7 % | 60.2 % |
| **XGBoost, adversarially trained** | 99.2 % | **97.2 %** | **97.6 %** | **97.0 %** | **87.5 %** |
| Autoencoder | 40.2 % | 26.9 % | 23.0 % | 21.9 % | 21.8 % |

* Even a small change (ε = 0.25, e.g. padding packets by up to 25 %) evades about one in four detections by the
  standard tree models.
* **Adversarial training** added 150,000 perturbed training attacks (random budgets 0.25–2.0). It holds detection at
  about 97 % up to ε = 1 and 87.5 % at ε = 2. Clean performance doesn't suffer: F1 is 0.964 against 0.960, and FPR is
  0.0170 against 0.0197.
* Caveat: the robustness test uses the same family of perturbations that adversarial training used. An attacker with
  new kinds of perturbation, or white-box access, would likely do better. The ε = 2 result shows the defence weakens
  outside its training budget.
* The autoencoder starts low and keeps falling: making an attack look more like normal traffic is exactly what an
  anomaly detector can't see.

## 6. Explainability and live operation

Every ML and threat-intel alert stores a SHAP waterfall: the features that pushed this flow toward "attack" or
"benign". The live detector's most important features globally are byte ratio, backward packet size and the TCP
windows. Behavioural alerts (scan, sweep, brute force) store their rule evidence instead, e.g. "192 distinct ports in
60 s". Alerts are mapped to MITRE ATT&CK. Remote public IPs are matched locally against abuse.ch Feodo Tracker,
Spamhaus DROP and CISA KEV, and enriched with ipwho.is, Shodan InternetDB and GreyNoise.

**First day of live capture** (one MacBook, `en0`, on several Wi-Fi networks, 2026-11-17): 91,557 flows to
1,224 destinations (UDP 72 %, TCP 21 %), stored in Supabase Postgres.

What running on real traffic exposed, and what was done about it:

| Live finding | Root cause (traced with SHAP and the training data) | Resolution |
|---|---|---|
| ICMP, mDNS/SSDP, ICMPv6 flagged as attacks | Protocols absent from both datasets (TCP/UDP flows only) | Out of scope: scored and stored, cannot raise an ML alert (11,070 flows so far) |
| 35 "Exploit" alerts (score 0.93–0.97) on DNS lookups to the local resolver | UNSW-NB15 labels **57,276 unanswered DNS flows *Generic* attack and only 55 normal**; the model learned "no reply = attack" | Unanswered DNS queries out of scope |
| 14 low "Anomaly" alerts on QUIC | Flows where only the server's packets were captured (mid-stream or idle split); every dataset flow starts with a packet from the initiator | One-sided flows out of scope |
| An Apple push connection (port 5223) treated as an attack *from* Apple | Direction guessed wrongly on a mid-stream capture (only ports < 1024 were treated as server ports) | Direction now from the TCP handshake, or else the client's temporary port (≥ 32768) |
| A real connect scan of `scanme.nmap.org` scored benign flow by flow (max 0.013) | CIC-IDS2017's scans are LAN SYN scans with a different signature | Behavioural rule caught it: 192 ports in 60 s, ATT&CK T1046 |
| STUN/WebRTC (UDP 3478) flows scored 0.91–0.99 "Exploit" | WebRTC traffic is absent from both datasets | **Deliberately left visible** as a record of the generalisation gap |
| Long HTTP connections with no server reply (port 80) scored "DoS" | Looks like the slowloris / slowhttptest pattern in CIC-IDS2017 | Left as alerts. The behaviour really does match; an analyst has to decide |

Every scope rule keeps the flow scored, stored, labelled with its reason and checked against the blocklists, so
nothing is hidden. `scripts/apply_policy.py` re-applies the current policy to open alerts after a rule change. They
are marked resolved, not deleted.

## 7. Limitations and future work

* **Datasets are old testbeds.** CIC-IDS2017 and UNSW-NB15 come from 2017 and 2015 lab networks, before QUIC and
  modern WebRTC were common. Section 6 shows the consequence directly. Labelled flows from the deployment network
  would help more than any model change.
* **Per-flow models can't see campaigns.** Scans and brute force spread over many short flows. The behavioural rules
  cover them, but they are hand-set thresholds (25 ports, 25 hosts, 15 sessions in 60 s).
* **No sequence model.** The spec's "LSTM/Autoencoder" option is covered by the autoencoder. A sequence model over
  each host's flows is the natural next step for catching slow multi-flow attacks.
* **Robustness is measured against one attack family** (section 5). White-box and feature-space-agnostic attacks
  were not evaluated.
* **One sensor on one host.** It sees the Mac's own traffic, not a whole network segment, so lateral movement between
  other hosts is invisible.

## 8. Reproducing

```bash
.venv/bin/python -m scripts.train          # all experiments, live model bundle, DB push (545 s for the last full run on this Mac)
.venv/bin/python -m pytest -q tests        # flow meter, dataset mappings, perturbations, rules, policy
```
