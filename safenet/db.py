"""Thin PostgreSQL (Supabase) access layer used by training, sensor and dashboard."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import psycopg
from psycopg.types.json import Jsonb

SCHEMA_SQL = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"


@contextmanager
def connect(url: str):
    # prepare_threshold=None keeps us compatible with Supabase's transaction pooler
    with psycopg.connect(url, autocommit=False, prepare_threshold=None, connect_timeout=15) as conn:
        yield conn


def init_schema(url: str) -> None:
    with connect(url) as conn:
        conn.execute(SCHEMA_SQL.read_text())
        conn.commit()


def create_run(conn, version: str, host: str, notes: dict) -> int:
    row = conn.execute(
        """insert into safenet.model_runs(version, host, notes) values (%s,%s,%s)
           on conflict (version) do update set notes = excluded.notes, host = excluded.host
           returning id""",
        (version, host, Jsonb(notes)),
    ).fetchone()
    conn.execute("delete from safenet.eval_results where run_id=%s", (row[0],))
    conn.execute("delete from safenet.feature_importance where run_id=%s", (row[0],))
    return row[0]


def add_eval(conn, run_id: int, experiment: str, model: str, train_ds: str, test_ds: str,
             metrics: dict, setting: str = "") -> None:
    conn.execute(
        """insert into safenet.eval_results(run_id, experiment, model, train_ds, test_ds, setting, metrics)
           values (%s,%s,%s,%s,%s,%s,%s)""",
        (run_id, experiment, model, train_ds, test_ds, setting, Jsonb(_clean(metrics))),
    )


def add_importance(conn, run_id: int, model: str, dataset: str, items: Iterable[tuple[str, float]]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            "insert into safenet.feature_importance(run_id, model, dataset, feature, mean_abs_shap) values (%s,%s,%s,%s,%s)",
            [(run_id, model, dataset, f, float(v)) for f, v in items],
        )


def upsert_sensor(conn, hostname: str, interface: str, model_version: str) -> int:
    row = conn.execute(
        """insert into safenet.sensors(hostname, interface, model_version) values (%s,%s,%s)
           on conflict (hostname, interface) do update
             set model_version = excluded.model_version, started_at = now(), last_seen = now()
           returning id""",
        (hostname, interface, model_version),
    ).fetchone()
    conn.commit()
    return row[0]


def heartbeat(conn, sensor_id: int, packets: int, flows: int) -> None:
    conn.execute(
        "update safenet.sensors set last_seen = now(), packets_seen = %s, flows_scored = %s where id = %s",
        (packets, flows, sensor_id),
    )


def insert_flow(conn, sensor_id: int, rec: dict[str, Any]) -> int:
    row = conn.execute(
        """insert into safenet.flows(sensor_id, ts_start, ts_end, src_ip, src_port, dst_ip, dst_port, proto,
                                     features, pred_label, p_attack, ae_score, ae_anomaly, is_alert, suppressed)
           values (%s, to_timestamp(%s), to_timestamp(%s), %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) returning id""",
        (sensor_id, rec["ts_start"], rec["ts_end"], rec["src_ip"], rec["src_port"], rec["dst_ip"], rec["dst_port"],
         rec["proto"], Jsonb(_clean(rec["features"])), rec["pred_label"], rec["p_attack"], rec["ae_score"],
         rec["ae_anomaly"], rec["is_alert"], rec.get("suppressed")),
    ).fetchone()
    return row[0]


def insert_alert(conn, flow_id: int, a: dict[str, Any]) -> None:
    conn.execute(
        """insert into safenet.alerts(flow_id, severity, label, p_attack, ae_score, src_ip, dst_ip, dst_port, proto,
                                      explanation, summary, remote_ip, mitre, intel_hits)
           values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (flow_id, a["severity"], a["label"], a["p_attack"], a["ae_score"], a["src_ip"], a["dst_ip"], a["dst_port"],
         a["proto"], Jsonb(_clean(a["explanation"])), a["summary"], a.get("remote_ip"),
         Jsonb(_clean(a.get("mitre", []))), Jsonb(_clean(a.get("intel_hits", [])))),
    )


def save_feed_status(conn, status: dict[str, dict]) -> None:
    for key, s in status.items():
        conn.execute(
            """insert into safenet.intel_feeds(key, name, url, ok, entries, error, fetched_at, checked_at)
               values (%s,%s,%s,%s,%s,%s, to_timestamp(%s), now())
               on conflict (key) do update set name=excluded.name, url=excluded.url, ok=excluded.ok,
                 entries=excluded.entries, error=excluded.error,
                 fetched_at=coalesce(excluded.fetched_at, safenet.intel_feeds.fetched_at), checked_at=now()""",
            (key, s["name"], s["url"], s["ok"], s.get("entries", 0), s.get("error"), s.get("fetched_at")),
        )


def ip_intel_fresh(conn, ip: str, max_age_hours: int = 24) -> bool:
    row = conn.execute(
        "select 1 from safenet.ip_intel where ip=%s and fetched_at > now() - make_interval(hours => %s)",
        (ip, max_age_hours),
    ).fetchone()
    return row is not None


def save_ip_intel(conn, ip: str, data: dict) -> None:
    conn.execute(
        """insert into safenet.ip_intel(ip, data, fetched_at) values (%s,%s,now())
           on conflict (ip) do update set data=excluded.data, fetched_at=now()""",
        (ip, Jsonb(_clean(data))),
    )


def prune(conn, keep_days: int = 14) -> None:
    """Keep the free-tier database small: drop non-alert flows older than keep_days."""
    conn.execute(
        "delete from safenet.flows where not is_alert and ts_end < now() - make_interval(days => %s)", (keep_days,)
    )


def _clean(obj):
    """Make numpy scalars / NaN JSON-safe."""
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)).replace("NaN", "null"))
