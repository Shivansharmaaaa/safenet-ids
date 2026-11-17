"""Alerting scope: which flows the ML models may raise alerts on."""
from __future__ import annotations

import ipaddress

from .flowmeter import FlowRecord

# mDNS, LLMNR, SSDP, NetBIOS name/datagram, WS-Discovery: LAN discovery chatter absent from both datasets
DISCOVERY_PORTS = {5353, 5355, 1900, 137, 138, 3702}


def out_of_scope(r: FlowRecord) -> str | None:
    """Flows the models were never trained to judge. They are still scored, stored and
    checked against threat-intel blocklists, but cannot raise an ML alert."""
    if r.proto not in ("tcp", "udp"):
        return f"protocol {r.proto} not in training data (TCP/UDP only)"
    try:
        a = ipaddress.ip_address(r.dst_ip)
        if a.is_multicast or r.dst_ip.endswith(".255") or a.is_link_local:
            return "local multicast/broadcast service discovery"
    except ValueError:
        pass
    if r.proto == "udp" and (r.dst_port in DISCOVERY_PORTS or r.src_port in DISCOVERY_PORTS):
        return "local service discovery (mDNS/LLMNR/SSDP/NetBIOS/WS-Discovery)"
    return None
