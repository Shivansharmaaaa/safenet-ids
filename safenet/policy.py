"""Alerting scope: which flows the ML models may raise alerts on."""
from __future__ import annotations

import ipaddress

from .flowmeter import FlowRecord

# mDNS, LLMNR, SSDP, NetBIOS name/datagram, WS-Discovery: LAN discovery chatter absent from both datasets
DISCOVERY_PORTS = {5353, 5355, 1900, 137, 138, 3702}
DNS_PORT = 53


def out_of_scope(r: FlowRecord) -> str | None:
    """Flows the models were never trained to judge, or whose training labels are a known dataset
    artifact. They are still scored, stored and checked against threat-intel blocklists, but cannot
    raise an ML alert."""
    if r.proto not in ("tcp", "udp"):
        return f"protocol {r.proto} not in training data (TCP/UDP only)"
    try:
        a = ipaddress.ip_address(r.dst_ip)
        if a.is_multicast or r.dst_ip.endswith(".255") or a.is_link_local:
            return "local multicast/broadcast service discovery"
    except ValueError:
        pass
    if r.fwd_pkts == 0:
        # every CIC-IDS2017 / UNSW-NB15 flow starts with a packet from its initiator
        return "one-sided flow: only the responder's packets were captured"
    if r.proto == "udp" and (r.dst_port in DISCOVERY_PORTS or r.src_port in DISCOVERY_PORTS):
        return "local service discovery (mDNS/LLMNR/SSDP/NetBIOS/WS-Discovery)"
    if r.proto == "udp" and r.dst_port == DNS_PORT and r.bwd_pkts == 0:
        # UNSW-NB15 labels 57,276 unanswered DNS flows 'Generic' attack and only 55 normal, so the models
        # learned "no reply = attack". On real networks unanswered lookups (timeouts, loss) are routine.
        return "unanswered DNS query (a UNSW-NB15 labelling artifact: ~99.9% 'Generic' attack there)"
    return None
