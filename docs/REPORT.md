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
* **Context plus adversarial training makes evasion much harder.** A per-host LSTM that reads each flow together
  with the same host's previous 15 flows keeps detecting 97 % of UNSW-NB15 attacks at ε = 0.25, where a single-flow
  XGBoost trained on the same rows falls to 24 %. Trained on the strongest evasion attempts against itself, it still
  detects 98.2 % of CIC-IDS2017 attacks at ε = 2, and 96.7 % of UNSW-NB15 attacks at ε = 1 (51 % at ε = 2) (section 6).
* **The first LSTM had learned the lab's network speed.** It missed a real internet port scan because scan
  flows in the datasets are fast LAN round trips. Removing the five timing features fixed this: the deployed LSTM
  flags 78 % of the real scan's flows (as one grouped alert), with under 0.03 % false alarms on the Mac's normal traffic.
* **Running on real traffic exposed problems no benchmark score shows.** Three alert floods on a real Mac turned out
  to be artifacts of the datasets or of flow measurement, not attacks (section 7). Each was traced with SHAP and
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
   autoencoder, multi-flow behavioural rules and threat intelligence. Section 7 shows which live alerts can be trusted.

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

## 6. Per-host sequence model (LSTM)

**Why.** A port scan, a brute-force run or a slow DoS is many flows that each look ordinary. Section 7's
`scanme.nmap.org` scan scored at most 0.013 flow by flow. A sequence model can see the campaign directly.

**Data.** Per-host sequences need source IPs and start times, which the files used above
(*MachineLearningCSV*, the official UNSW train/test partitions) drop. The sequence model therefore uses:
* **CIC-IDS2017 as re-extracted and re-labelled by Engelen et al.** (KU Leuven DistriNet, 2021–2022): 2,099,976 flows
  with fixed CICFlowMeter bugs and corrected labels. Flows labelled "Attempted" count as benign, as the authors
  recommend.
* **The full UNSW-NB15 records** (`UNSW-NB15_1..4.csv`, 2,540,047 flows) from UNSW's own share.

Only TCP/UDP flows enter histories (4.58 M), exactly as on the live sensor.

**Model** (`safenet/sequence.py`). For every flow, the model reads the same host's last 16 flows (zero-padded at the
start of a host's history). Each step has the 22 shared flow features plus three context features: whole seconds
since the host's previous flow (UNSW-NB15's time resolution, applied everywhere), whether the destination host
repeats, and whether the destination port repeats. Then LayerNorm, a 64-unit LSTM, and a head over [last hidden
state, current flow]. Training: 4 epochs, AdamW, square-root class weights. A unit test checks that the live
`HostHistory` builds windows identical to the training pipeline. Training is deterministic: a re-run reproduced
every metric exactly.

Four variants, all trained on the same rows:

* **LSTM**: all 25 step features.
* **LSTM-NoTiming**: a fixed mask removes the five features that mostly measure network round-trip time (duration,
  bytes/s, packets/s, forward and backward inter-arrival time). The full LSTM **missed a real internet connect scan**
  (p ≈ 0.001): setting only `pkts_per_s` of those flows to a CIC-IDS2017 value flipped the score to 0.93. It had
  learned "a scan is fast because the lab LAN is fast", which does not hold across the internet.
* **LSTM-NoTiming-Adv**: the same mask, trained with adversarial examples. In every batch, half of the attack
  windows are replaced by a random evasion attempt (budget ε ~ U(0, 2), random padding, slowing and packet
  injection on every attacker flow, and the gaps before them stretched). This is the perturbation family of section 5,
  sampled at random.
* **LSTM-NoTiming-AdvAdaptive** (live): as above, but each attack window gets the *strongest* of 8 attempts
  against the model being trained (the 7 corners of a random budget cube [0, ε]³ over padding, slowing and
  injection, plus one random point). The evaluation attacker searches a finer 64-point grid at each budget, so it
  remains the stronger of the two.

**Split.** Random 10-minute time buckets: 70 % train, 10 % validation, 20 % test. **The single-flow baseline is
XGBoost on the same 22 features and exactly the same rows**, so differences come from the host context alone.

| Train → test | LSTM | LSTM-NoTiming | LSTM-NoTiming-Adv | **…-AdvAdaptive** | XGBoost (single flow) |
|---|---|---|---|---|---|
| CIC → CIC | **0.997** | 0.994 | 0.996 | 0.994 | 0.988 |
| UNSW → UNSW | 0.953 | 0.955 | 0.958 | 0.944 | **0.966** |
| CIC → UNSW | **0.225** | 0.022 | 0.082 | 0.104 | 0.038 |
| UNSW → CIC | 0.051 | **0.544** | 0.007 | 0.007 | 0.222 |
| both → CIC | 0.995 | **0.997** | 0.993 | 0.993 | 0.987 |
| both → UNSW | 0.959 | 0.957 | 0.959 | 0.960 | **0.968** |

F1 at threshold 0.5. False-positive rates in-dataset: CIC 0.03 / 0.05 / 0.07 / 0.04 / 0.00 %, UNSW 1.28 / 1.22 /
**0.43** / 1.45 / 0.74 % (same column order); trained on both datasets, the deployed AdvAdaptive model has 0.10 % on
CIC and 0.71 % on UNSW. Per-category detection, precision, recall and ROC-AUC are in
[`sequence_results.json`](../artifacts/reports/sequence_results.json) and the dashboard's *Model lab → LSTM* tab.

* **In-dataset, host context helps on repetitive attacks.** DoS (99.0–99.9 % vs 96.5 % on CIC) and scans on
  UNSW-NB15 (99.7 % vs 95.1 %). It hurts where one attack flow hides among a host's mostly normal flows: Botnet C2
  beacons (12–69 % vs 99 %) and the six CIC Exploit flows (Infiltration, Heartbleed). These are left to the per-flow
  models, which run alongside.
* **Adversarial training cost almost nothing in clean accuracy.** The random variant cut UNSW-NB15 false alarms by
  two thirds (1.22 % → 0.43 %) at a cost in UNSW scan recall (99.4 % → 81.7 % in-dataset). The adaptive variant keeps
  scan recall (97.2 %) but not that false-alarm gain (1.45 %), and it gives up CIC Botnet detection (combined model:
  52 % → 12 %), which the per-flow models cover at 99 %.
* **Cross-dataset results are unstable, not a strength of any variant.** Each variant wins one direction and loses
  the other, and the masked model's 0.54 (UNSW → CIC) fell to 0.007 once adversarial examples were added. Host
  context doesn't close the gap between testbeds.

**Evasion** (adaptive black-box, combined models, 8,000 attack windows per dataset). The attacker controls
*all* their flows in the window: padding, slowing, injecting, *and* spacing the flows further apart. The
single-flow XGBoost here is **not** adversarially trained, so the first rows compare architectures.

| Test set | Model | ε = 0 | 0.25 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|
| CIC-IDS2017 | LSTM | 99.1 % | 97.8 % | 97.4 % | 94.7 % | 44.3 % |
| CIC-IDS2017 | LSTM-NoTiming | 99.5 % | 97.7 % | 87.5 % | 84.4 % | 54.2 % |
| CIC-IDS2017 | LSTM-NoTiming-Adv | 98.8 % | **98.8 %** | **98.8 %** | **98.7 %** | **98.4 %** |
| CIC-IDS2017 | **LSTM-NoTiming-AdvAdaptive** | 98.8 % | 98.7 % | 98.7 % | 98.5 % | 98.2 % |
| CIC-IDS2017 | XGBoost | 97.4 % | 96.1 % | 90.3 % | 71.6 % | 61.5 % |
| UNSW-NB15 | LSTM | 97.8 % | 97.0 % | 89.8 % | 37.6 % | **32.3 %** |
| UNSW-NB15 | LSTM-NoTiming | 98.6 % | 98.4 % | 98.0 % | 35.9 % | 24.8 % |
| UNSW-NB15 | LSTM-NoTiming-Adv | 97.2 % | 96.9 % | 96.6 % | 37.1 % | 30.5 % |
| UNSW-NB15 | **LSTM-NoTiming-AdvAdaptive** | 97.0 % | 96.9 % | 96.9 % | **96.7 %** | **51.2 %** |
| UNSW-NB15 | XGBoost | 98.1 % | 23.5 % | 12.0 % | 6.9 % | 5.2 % |

* On CIC-IDS2017, adversarial training makes the LSTM practically insensitive to this attacker: 98.4 % of attacks
  are still detected at ε = 2, against 54 % without it.
* On UNSW-NB15 random adversarial training does not help at ε ≥ 1 (37 %). **Training on the strongest attempt
  instead lifts detection at ε = 1 from 37 % to 97 %, and at ε = 2 from 31 % to 51 %**, with the same clean F1. At
  ε = 2 (every attacker packet padded and injected by up to 200 %, every flow and gap tripled), half of UNSW's attack
  windows still look like its benign traffic in these features.
* Both robustness results hold for **this** attacker family (padding, slowing, injection, spacing). The adaptive
  variant was trained against points of the same family, so its gain is partly specific to it; other attacks
  (white-box, mimicry of benign hosts) were not evaluated.

**Live.** The sensor keeps a 16-flow history per source host and scores every TCP/UDP flow with the LSTM
(`flows.p_seq`). A flow raises a *sequence* alert when the LSTM scores it ≥ 0.90 and the per-flow models did not
already alert. The same scope policy applies. **A host's sequence alerts within 10 minutes of each other form one
campaign** (at most 1 h): the first flow raises the alert, later flows are flagged and folded into it, and the alert
records the campaign's size, ports and hosts. Each sequence alert stores expected-gradients SHAP values
(`shap.GradientExplainer`), summed per feature and per earlier flow, so the console shows *which of the host's
recent flows* pushed the score.

**Live check** (2026-11-22, the Mac's sensor; replay of all 74,549 stored TCP/UDP flows, plus two fresh 50-port
connect scans of `scanme.nmap.org`):

| Model | Scan flows ≥ 0.90 | Normal in-scope flows ≥ 0.90 (n = 61,075) | ≥ 0.5 |
|---|---|---|---|
| LSTM (first deployment) | ≈ 0 % (p ≈ 0.001) | 27 of 57,814 earlier flows | 0.41 % |
| LSTM-NoTiming | 87.3 % | 10 (0.016 %) | 0.15 % |
| LSTM-NoTiming-Adv | 78.4 % | **8 (0.013 %)** | **0.04 %** |

Re-run on 2026-11-23 for the adaptive variant (the database then held 16,380 flows because of the 3-day retention;
the same 102 scan flows): LSTM-NoTiming-Adv 78.4 % of scan flows, 2 of 14,042 normal flows (0.014 %);
**LSTM-NoTiming-AdvAdaptive 77.5 %, 4 of 14,042 (0.028 %)**. The adaptive model was deployed: the same live behaviour,
far more robust to evasion on UNSW-NB15.

Live, the second scan produced **one** LSTM alert covering 46 flows and 45 ports (peak score 0.999997), next to the
behavioural rule's alert. Before grouping, the same scan produced 43 separate alerts. The per-flow XGBoost scored
those flows 0.00. The few normal flows that do cross 0.90 mostly come right after a scan, while the scan's flows are
still inside the host's 16-flow window; grouping folds them into the scan's campaign.

## 7. Explainability and live operation

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
| The first per-host LSTM also missed the scan (p ≈ 0.001) | It had learned LAN round-trip time (packet rate) as the scan signature | LSTM retrained without timing features: flags 78 % of the scan's flows as one campaign alert (section 6) |
| One scan raised 43 separate LSTM alerts | Every flow of a campaign crosses the threshold | A host's sequence alerts within 10 min are grouped into one alert with campaign evidence |
| The Mac's sensor silently stopped for 15 h overnight (process alive, nothing captured) | The database connection died while the Mac slept; a read on the dead socket blocked forever, so no error ever reached the restart logic | TCP keepalives on every connection (a dead link raises within ~35 s) and a watchdog thread that exits the sensor after 5 min without progress, so launchd / Docker restarts it |
| STUN/WebRTC (UDP 3478) flows scored 0.91–0.99 "Exploit" | WebRTC traffic is absent from both datasets | **Deliberately left visible** as a record of the generalisation gap |
| Long HTTP connections with no server reply (port 80) scored "DoS" | Looks like the slowloris / slowhttptest pattern in CIC-IDS2017 | Left as alerts. The behaviour really does match; an analyst has to decide |

Every scope rule keeps the flow scored, stored, labelled with its reason and checked against the blocklists, so
nothing is hidden. `scripts/apply_policy.py` re-applies the current policy to open alerts after a rule change. They
are marked resolved, not deleted.

## 8. Limitations and future work

* **Datasets are old testbeds.** CIC-IDS2017 and UNSW-NB15 come from 2017 and 2015 lab networks, before QUIC and
  modern WebRTC were common. Section 7 shows the consequence directly. Labelled flows from the deployment network
  would help more than any model change.
* **Per-flow models can't see campaigns.** Scans and brute force spread over many short flows. The behavioural rules
  cover them, but they are hand-set thresholds (25 ports, 25 hosts, 15 sessions in 60 s).
* **The sequence model's robustness is against one attacker family**, which it was also trained on. Half of
  UNSW-NB15's attacks still evade it at ε = 2 (section 6). Botnet beacons and rare single-flow exploits are better left
  to the per-flow models.
* **Robustness is measured against one attack family** (sections 5 and 6). White-box and feature-space-agnostic attacks
  were not evaluated.
* **Coverage depends on where sensors sit.** A sensor sees only its own interface; on Wi-Fi that is the host's own
  traffic. SafeNet supports any number of sensors (macOS, Linux systemd, Docker) reporting to one console, but
  watching a whole segment needs one on the gateway or a switch mirror port ([DEPLOY.md](DEPLOY.md)).

## 9. Reproducing

```bash
.venv/bin/python -m scripts.train          # all experiments, live model bundle, DB push (545 s for the last full run on this Mac)
.venv/bin/python -m scripts.train_sequence # three LSTM variants vs XGBoost, cross-dataset, evasion (1,106 s on this Mac)
.venv/bin/python -m pytest -q tests        # flow meter, dataset mappings, perturbations, rules, policy, sequences
```
