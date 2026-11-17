"""Re-apply the current alert-scope policy (safenet/policy.py) to OPEN ML/autoencoder alerts.

Alerts raised before a policy change on flows that are now out of scope are marked resolved (not deleted),
and their flows get the same `suppressed` reason the live sensor now records. Threat-intel and
behavioural alerts are never touched.

usage: .venv/bin/python -m scripts.apply_policy [--dry-run]
"""
from __future__ import annotations

import argparse
from collections import Counter
from types import SimpleNamespace

from safenet import db
from safenet.config import database_url
from safenet.policy import out_of_scope

QUERY = """
select a.id, a.flow_id, f.src_ip, f.src_port, f.dst_ip, f.dst_port, f.proto, (f.features->>'bwd_pkts')::float
from safenet.alerts a join safenet.flows f on f.id = a.flow_id
where a.status = 'open'
  and coalesce(a.explanation->>'detector', 'ml') in ('ml', 'autoencoder')
  and coalesce(jsonb_array_length(a.intel_hits), 0) = 0
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    with db.connect(database_url()) as conn:
        hits = []
        for aid, fid, sip, sport, dip, dport, proto, bwd in conn.execute(QUERY).fetchall():
            reason = out_of_scope(SimpleNamespace(src_ip=sip, src_port=sport, dst_ip=dip, dst_port=dport,
                                                  proto=proto, bwd_pkts=bwd))
            if reason:
                hits.append((aid, fid, reason))
        for reason, n in Counter(r for _, _, r in hits).items():
            print(f"{n:5d}  {reason}")
        if args.dry_run or not hits:
            print(f"{len(hits)} open alerts now out of scope" + (" (dry run, nothing changed)" if args.dry_run else ""))
            return
        with conn.cursor() as cur:
            cur.executemany("update safenet.flows set suppressed = %s, is_alert = false where id = %s",
                            [(r, f) for _, f, r in hits])
            cur.execute("update safenet.alerts set status = 'resolved' where id = any(%s)", ([a for a, _, _ in hits],))
        conn.commit()
        print(f"resolved {len(hits)} alerts that the current scope policy no longer raises")


if __name__ == "__main__":
    main()
