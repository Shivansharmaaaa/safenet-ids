"""Real-world threat intelligence.

Feeds (downloaded, matched LOCALLY against every flow; nothing about your traffic is sent out):
  * abuse.ch Feodo Tracker   botnet C2 IP blocklist   https://feodotracker.abuse.ch/
  * Spamhaus DROP (v4 + v6)  hijacked / criminal netblocks  https://www.spamhaus.org/drop/
  * CISA KEV catalog          CVEs exploited in the wild  https://www.cisa.gov/known-exploited-vulnerabilities-catalog

Per-IP enrichment (called ONLY for the remote, public IP of an alert):
  * ipwho.is                  geolocation + ASN / ISP
  * Shodan InternetDB         open ports, hostnames, CVEs, tags   https://internetdb.shodan.io/
  * GreyNoise Community       internet scanner / benign-service classification  https://viz.greynoise.io/

MITRE ATT&CK mapping for each detection category (https://attack.mitre.org/).
"""
from __future__ import annotations

import ipaddress
import json
import ssl
import threading
import time
import urllib.error
import urllib.request

try:  # python.org builds on macOS ship without a CA store; use Mozilla's bundle via certifi
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL = ssl.create_default_context()

UA = "SafeNet-IDS/1.0 (+https://github.com/Shivansharmaaaa/safenet-ids)"

FEEDS = {
    "feodo": ("abuse.ch Feodo Tracker (botnet C2)", "https://feodotracker.abuse.ch/downloads/ipblocklist.txt"),
    "spamhaus_drop": ("Spamhaus DROP v4", "https://www.spamhaus.org/drop/drop_v4.json"),
    "spamhaus_drop6": ("Spamhaus DROP v6", "https://www.spamhaus.org/drop/drop_v6.json"),
    "cisa_kev": ("CISA Known Exploited Vulnerabilities",
                 "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"),
}

MITRE = {
    "DoS": [("T1498", "Network Denial of Service"), ("T1499", "Endpoint Denial of Service")],
    "Scan": [("T1046", "Network Service Discovery"), ("T1595", "Active Scanning")],
    "BruteForce": [("T1110", "Brute Force")],
    "WebAttack": [("T1190", "Exploit Public-Facing Application")],
    "Botnet": [("T1071", "Application Layer Protocol"), ("T1571", "Non-Standard Port")],
    "Exploit": [("T1210", "Exploitation of Remote Services"), ("T1203", "Exploitation for Client Execution")],
    "ThreatIntel": [("T1071", "Application Layer Protocol"), ("T1583", "Acquire Infrastructure")],
    "Anomaly": [],
}


def mitre_for(label: str) -> list[dict]:
    return [{"id": t, "name": n, "url": f"https://attack.mitre.org/techniques/{t}/"} for t, n in MITRE.get(label, [])]


def _get(url: str, timeout: float = 8.0):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json, text/plain"})
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
        return r.status, r.read()


def is_public(ip: str) -> bool:
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return a.is_global and not a.is_multicast


class ThreatFeeds:
    """Downloads the feeds and refreshes them every ``ttl`` seconds."""

    def __init__(self, ttl: float = 6 * 3600):
        self.ttl = ttl
        self.c2_ips: set[str] = set()
        self.drop_nets: list = []
        self.kev: dict[str, dict] = {}
        self.status: dict[str, dict] = {}
        self.loaded_at = 0.0
        self._lock = threading.Lock()

    def refresh(self, force: bool = False) -> dict[str, dict]:
        if not force and time.time() - self.loaded_at < self.ttl:
            return self.status
        c2, nets, kev, status = set(), [], {}, {}
        for key, (name, url) in FEEDS.items():
            try:
                code, body = _get(url, timeout=30)
                text = body.decode("utf-8", "replace")
                if key == "feodo":
                    c2 = {l.strip() for l in text.splitlines() if l.strip() and not l.startswith("#")}
                    n = len(c2)
                elif key.startswith("spamhaus"):
                    k = 0
                    for line in text.splitlines():
                        line = line.strip()
                        if line.startswith("{") and '"cidr"' in line:
                            nets.append(ipaddress.ip_network(json.loads(line)["cidr"], strict=False))
                            k += 1
                    n = k
                else:
                    data = json.loads(text)
                    kev = {v["cveID"]: {"name": v.get("vulnerabilityName"), "vendor": v.get("vendorProject"),
                                        "product": v.get("product"), "added": v.get("dateAdded"),
                                        "ransomware": v.get("knownRansomwareCampaignUse")}
                           for v in data.get("vulnerabilities", [])}
                    n = len(kev)
                status[key] = {"name": name, "url": url, "ok": True, "entries": n, "fetched_at": time.time()}
            except Exception as e:  # keep the previous copy of a feed if a refresh fails
                prev = self.status.get(key, {})
                status[key] = {"name": name, "url": url, "ok": False, "error": repr(e)[:200],
                               "entries": prev.get("entries", 0), "fetched_at": prev.get("fetched_at")}
        with self._lock:
            if status["feodo"]["ok"]:
                self.c2_ips = c2
            if status["spamhaus_drop"]["ok"] or status["spamhaus_drop6"]["ok"]:
                self.drop_nets = nets
            if status["cisa_kev"]["ok"]:
                self.kev = kev
            self.status = status
            self.loaded_at = time.time()
        return status

    def match(self, ip: str) -> list[dict]:
        """Which blocklists contain this IP (local lookup only)."""
        hits = []
        with self._lock:
            if ip in self.c2_ips:
                hits.append({"feed": "feodo", "name": FEEDS["feodo"][0], "url": "https://feodotracker.abuse.ch/browse/host/" + ip + "/"})
            try:
                a = ipaddress.ip_address(ip)
                for net in self.drop_nets:
                    if a.version == net.version and a in net:
                        hits.append({"feed": "spamhaus_drop", "name": "Spamhaus DROP", "netblock": str(net),
                                     "url": "https://check.spamhaus.org/"})
                        break
            except ValueError:
                pass
        return hits


def enrich_ip(ip: str, kev: dict[str, dict] | None = None) -> dict:
    """Look up a PUBLIC IP in ipwho.is, Shodan InternetDB and GreyNoise Community."""
    out: dict = {"ip": ip, "fetched_at": time.time(), "sources": {}}
    if not is_public(ip):
        out["private"] = True
        return out
    try:
        _, b = _get(f"https://ipwho.is/{ip}")
        g = json.loads(b)
        if g.get("success"):
            conn = g.get("connection") or {}
            out["geo"] = {"country": g.get("country"), "country_code": g.get("country_code"), "city": g.get("city"),
                          "lat": g.get("latitude"), "lon": g.get("longitude"), "asn": conn.get("asn"),
                          "org": conn.get("org"), "isp": conn.get("isp"), "flag": (g.get("flag") or {}).get("emoji")}
        out["sources"]["ipwho.is"] = "ok"
    except Exception as e:
        out["sources"]["ipwho.is"] = f"error: {e!r}"[:120]
    try:
        _, b = _get(f"https://internetdb.shodan.io/{ip}")
        s = json.loads(b)
        vulns = s.get("vulns", []) or []
        out["shodan"] = {"ports": s.get("ports", []), "hostnames": s.get("hostnames", []), "tags": s.get("tags", []),
                         "cpes": s.get("cpes", []), "vulns": vulns,
                         "kev": [v for v in vulns if kev and v in kev]}
        out["sources"]["shodan"] = "ok"
    except urllib.error.HTTPError as e:
        out["sources"]["shodan"] = "no data" if e.code == 404 else f"http {e.code}"
    except Exception as e:
        out["sources"]["shodan"] = f"error: {e!r}"[:120]
    try:
        _, b = _get(f"https://api.greynoise.io/v3/community/{ip}")
        gn = json.loads(b)
        out["greynoise"] = {k: gn.get(k) for k in ("noise", "riot", "classification", "name", "last_seen", "link", "message")}
        out["sources"]["greynoise"] = "ok"
    except urllib.error.HTTPError as e:
        if e.code == 404:
            try:
                gn = json.loads(e.read())
                out["greynoise"] = {k: gn.get(k) for k in ("noise", "riot", "message")}
            except Exception:
                pass
            out["sources"]["greynoise"] = "not observed"
        else:
            out["sources"]["greynoise"] = f"http {e.code}"
    except Exception as e:
        out["sources"]["greynoise"] = f"error: {e!r}"[:120]
    return out


def remote_ip(src: str, dst: str) -> str | None:
    """The public endpoint of a flow (the one that is not this network)."""
    if is_public(dst):
        return dst
    if is_public(src):
        return src
    return None
