"""SafeNet SOC console: live flows, explainable alerts, threat intelligence and model evaluation.
Every number on every page is read from the SafeNet PostgreSQL (Supabase) database, which is fed
by the live sensor (real captured traffic) and by the training pipeline (CIC-IDS2017 + UNSW-NB15)."""
from __future__ import annotations

import ipaddress
import os
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg
import streamlit as st

st.set_page_config(page_title="SafeNet · Intrusion Detection", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

# ============================================================================ design tokens
C = dict(bg="#070B14", surface="#0E1524", surface2="#131C2E", border="#1E293B", text="#E2E8F0", muted="#94A3B8",
         accent="#22D3EE", accent2="#818CF8", high="#F43F5E", medium="#F59E0B", low="#38BDF8", ok="#10B981")
SEV = {"high": C["high"], "medium": C["medium"], "low": C["low"]}
SERIES = [C["accent"], C["accent2"], C["ok"], C["medium"], C["high"], "#F472B6", "#A3E635"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: radial-gradient(1200px 600px at 10% -10%, #0F1B33 0%, {C['bg']} 55%) fixed; }}
[data-testid="stSidebar"] {{ background: {C['surface']}; border-right: 1px solid {C['border']}; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }}
h1, h2, h3, h4 {{ letter-spacing: -0.02em; color: {C['text']}; }}
code, .mono {{ font-family: 'JetBrains Mono', monospace !important; font-size: 0.85em; }}
.sn-top {{ display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom: 18px; flex-wrap: wrap; }}
.sn-brand {{ display:flex; align-items:center; gap:12px; }}
.sn-brand .t {{ font-size: 1.55rem; font-weight: 800; color:{C['text']}; line-height:1.1; }}
.sn-brand .s {{ font-size: .82rem; color:{C['muted']}; }}
.sn-pill {{ display:inline-flex; align-items:center; gap:8px; padding:6px 12px; border-radius:999px; font-size:.8rem;
           border:1px solid {C['border']}; background:{C['surface2']}; color:{C['text']}; white-space:nowrap; }}
.sn-dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
.sn-dot.on {{ background:{C['ok']}; box-shadow:0 0 0 0 rgba(16,185,129,.7); animation: pulse 1.8s infinite; }}
.sn-dot.off {{ background:{C['high']}; }}
@keyframes pulse {{ 0% {{ box-shadow:0 0 0 0 rgba(16,185,129,.6); }} 70% {{ box-shadow:0 0 0 9px rgba(16,185,129,0); }} 100% {{ box-shadow:0 0 0 0 rgba(16,185,129,0); }} }}
.sn-kpi {{ background: linear-gradient(180deg, {C['surface2']} 0%, {C['surface']} 100%); border:1px solid {C['border']};
          border-radius:14px; padding:16px 18px; height:100%; }}
.sn-kpi .l {{ color:{C['muted']}; font-size:.74rem; text-transform:uppercase; letter-spacing:.08em; font-weight:600; }}
.sn-kpi .v {{ color:{C['text']}; font-size:1.75rem; font-weight:700; margin-top:6px; line-height:1.15; }}
.sn-kpi .d {{ color:{C['muted']}; font-size:.78rem; margin-top:4px; }}
.sn-kpi.accent {{ border-top:2px solid {C['accent']}; }} .sn-kpi.high {{ border-top:2px solid {C['high']}; }}
.sn-kpi.medium {{ border-top:2px solid {C['medium']}; }} .sn-kpi.ok {{ border-top:2px solid {C['ok']}; }}
.sn-kpi.violet {{ border-top:2px solid {C['accent2']}; }}
.sn-card {{ background:{C['surface']}; border:1px solid {C['border']}; border-radius:14px; padding:16px 18px; margin-bottom:12px; }}
.sn-card h4 {{ margin:0 0 8px 0; font-size:1rem; }}
.sn-badge {{ display:inline-block; padding:2px 9px; border-radius:6px; font-size:.72rem; font-weight:700; letter-spacing:.04em;
            text-transform:uppercase; }}
.sn-badge.high {{ background:rgba(244,63,94,.15); color:{C['high']}; border:1px solid rgba(244,63,94,.35); }}
.sn-badge.medium {{ background:rgba(245,158,11,.15); color:{C['medium']}; border:1px solid rgba(245,158,11,.35); }}
.sn-badge.low {{ background:rgba(56,189,248,.15); color:{C['low']}; border:1px solid rgba(56,189,248,.35); }}
.sn-chip {{ display:inline-block; margin:2px 6px 2px 0; padding:3px 9px; border-radius:999px; font-size:.75rem;
           background:{C['surface2']}; border:1px solid {C['border']}; color:{C['text']}; text-decoration:none !important; }}
.sn-chip:hover {{ border-color:{C['accent']}; color:{C['accent']}; }}
.sn-chip.kev {{ border-color:rgba(244,63,94,.5); color:{C['high']}; }}
.sn-alert {{ display:flex; gap:12px; align-items:flex-start; padding:12px 14px; border:1px solid {C['border']};
            border-left:3px solid var(--sev); border-radius:10px; background:{C['surface']}; margin-bottom:8px; }}
.sn-alert .m {{ color:{C['muted']}; font-size:.78rem; }}
.sn-alert .x {{ color:{C['text']}; font-size:.9rem; margin-top:2px; }}
.sn-sec {{ color:{C['muted']}; font-size:.75rem; text-transform:uppercase; letter-spacing:.1em; font-weight:700; margin: 18px 0 8px; }}
[data-testid="stMetric"] {{ background:{C['surface']}; border:1px solid {C['border']}; border-radius:12px; padding:12px 14px; }}
div[data-testid="stDataFrame"] {{ border:1px solid {C['border']}; border-radius:12px; overflow:hidden; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{ background:{C['surface']}; border:1px solid {C['border']}; border-radius:10px; padding:8px 14px; }}
.stTabs [aria-selected="true"] {{ border-color:{C['accent']} !important; color:{C['accent']} !important; }}
a {{ color:{C['accent']}; }}
</style>
""", unsafe_allow_html=True)

SHIELD = f"""<svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 2l8 3v6c0 5-3.4 9.4-8 11-4.6-1.6-8-6-8-11V5l8-3z" fill="url(#g)" stroke="{C['accent']}" stroke-width="1.2"/>
<path d="M8.5 12.2l2.4 2.4 4.8-5" stroke="#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
<defs><linearGradient id="g" x1="4" y1="2" x2="20" y2="22"><stop stop-color="#0E7490"/><stop offset="1" stop-color="#4338CA"/></linearGradient></defs></svg>"""

FEATURE_LABELS = {
    "duration_s": "Flow duration (s)", "fwd_pkts": "Packets sent by initiator", "bwd_pkts": "Packets sent by responder",
    "fwd_bytes": "Bytes sent by initiator", "bwd_bytes": "Bytes sent by responder",
    "fwd_pkt_mean": "Avg packet size (initiator)", "bwd_pkt_mean": "Avg packet size (responder)",
    "bytes_per_s": "Throughput (bytes/s)", "pkts_per_s": "Packet rate (pkts/s)",
    "fwd_iat_ms": "Initiator packet gap (ms)", "bwd_iat_ms": "Responder packet gap (ms)",
    "fwd_win": "Initiator TCP window", "bwd_win": "Responder TCP window", "is_tcp": "TCP flow",
    "byte_ratio": "Response/request byte ratio", "svc_http": "HTTP service", "svc_tls": "TLS/HTTPS service",
    "svc_dns": "DNS service", "svc_ssh": "SSH service", "svc_ftp": "FTP service", "svc_mail": "Mail service",
    "svc_other": "Other service",
}
DS = {"cicids2017": "CIC-IDS2017", "unswnb15": "UNSW-NB15", "combined": "Both"}


# ============================================================================ auth (Google OIDC)
def _secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)


def _auth_configured() -> bool:
    try:
        return "auth" in st.secrets
    except Exception:
        return False


def require_login() -> None:
    if not _auth_configured():
        return
    if not st.user.is_logged_in:
        _, mid, _ = st.columns([1, 1.3, 1])
        with mid:
            st.markdown(f"<div style='text-align:center;margin-top:12vh'>{SHIELD.replace('40', '72')}"
                        f"<h1 style='margin:12px 0 4px'>SafeNet</h1>"
                        f"<p style='color:{C['muted']}'>Explainable and robust network intrusion detection.<br>"
                        f"Sign in with an authorised Google account to open the SOC console.</p></div>",
                        unsafe_allow_html=True)
            # single [auth] provider in secrets -> st.login() with no provider name
            st.button("Sign in with Google", on_click=st.login, type="primary", width="stretch")
        st.stop()
    allowed = [e.strip().lower() for e in str(_secret("ALLOWED_EMAILS", "")).split(",") if e.strip()]
    if allowed and str(st.user.email).lower() not in allowed:
        st.error(f"{st.user.email} is not authorised to view this console.")
        st.button("Sign out", on_click=st.logout)
        st.stop()


require_login()


# ============================================================================ data access
def _db_url() -> str:
    url = _secret("DATABASE_URL")
    if not url:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        url = os.environ.get("DATABASE_URL")
    if not url:
        st.error("DATABASE_URL is not configured.")
        st.stop()
    return url


@st.cache_resource(show_spinner=False)
def _conn():
    return psycopg.connect(_db_url(), prepare_threshold=None, connect_timeout=15, autocommit=True)


def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    for attempt in (1, 2):
        try:
            cur = _conn().execute(sql, params)
            cols = [d.name for d in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols)
        except (psycopg.OperationalError, psycopg.InterfaceError):
            _conn.clear()  # stale pooled connection: reconnect once
            if attempt == 2:
                raise


def execute(sql: str, params: tuple = ()) -> None:
    _conn().execute(sql, params)


@st.cache_data(ttl=10, show_spinner=False)
def d_sensor():
    return q("select hostname, interface, model_version, started_at, last_seen, packets_seen, flows_scored "
             "from safenet.sensors order by last_seen desc limit 1")


@st.cache_data(ttl=10, show_spinner=False)
def d_kpis():
    return q("""select
      (select count(*) from safenet.flows where ts_end > now() - interval '24 hours') flows_24h,
      (select count(*) from safenet.flows where ts_end > now() - interval '1 hour') flows_1h,
      (select count(*) from safenet.alerts where created_at > now() - interval '24 hours') alerts_24h,
      (select count(*) from safenet.alerts where status = 'open') open_alerts,
      (select count(*) from safenet.alerts where label = 'ThreatIntel' and created_at > now() - interval '7 days') intel_7d,
      (select count(*) from safenet.flows where suppressed is not null and ts_end > now() - interval '24 hours') suppressed_24h,
      (select count(distinct dst_ip) from safenet.flows where ts_end > now() - interval '24 hours') dests_24h""").iloc[0]


@st.cache_data(ttl=10, show_spinner=False)
def d_timeline(hours: int):
    return q("""select date_trunc('minute', ts_end) as t, count(*) flows, sum(is_alert::int) alerts,
                       sum((features->>'fwd_bytes')::float + (features->>'bwd_bytes')::float) bytes
                from safenet.flows where ts_end > now() - make_interval(hours => %s) group by 1 order by 1""", (hours,))


@st.cache_data(ttl=10, show_spinner=False)
def d_flows(limit: int = 1000):
    return q("""select ts_end, src_ip, src_port, dst_ip, dst_port, proto, pred_label, p_attack, ae_score, ae_anomaly,
                       is_alert, suppressed, (features->>'fwd_bytes')::float + (features->>'bwd_bytes')::float as bytes,
                       (features->>'fwd_pkts')::float + (features->>'bwd_pkts')::float as packets,
                       (features->>'duration_s')::float as duration_s
                from safenet.flows order by ts_end desc limit %s""", (limit,))


@st.cache_data(ttl=10, show_spinner=False)
def d_alerts(limit: int = 500):
    return q("""select a.id, a.created_at, a.severity, a.label, a.p_attack, a.ae_score, a.src_ip, a.dst_ip, a.dst_port,
                       a.proto, a.summary, a.explanation, a.status, a.remote_ip, a.mitre, a.intel_hits, f.features
                from safenet.alerts a left join safenet.flows f on f.id = a.flow_id
                order by a.created_at desc limit %s""", (limit,))


@st.cache_data(ttl=60, show_spinner=False)
def d_feeds():
    return q("select key, name, url, ok, entries, error, fetched_at, checked_at from safenet.intel_feeds order by key")


@st.cache_data(ttl=30, show_spinner=False)
def d_ip_intel():
    return q("""select i.ip, i.data, i.fetched_at, count(a.id) alerts, max(a.created_at) last_alert
                from safenet.ip_intel i left join safenet.alerts a on a.remote_ip = i.ip
                group by i.ip, i.data, i.fetched_at order by alerts desc, i.fetched_at desc""")


@st.cache_data(ttl=600, show_spinner=False)
def d_eval():
    run = q("select id, version, created_at, notes from safenet.model_runs order by created_at desc limit 1")
    if run.empty:
        return run, pd.DataFrame(), pd.DataFrame()
    rid = int(run.iloc[0]["id"])
    ev = q("select experiment, model, train_ds, test_ds, setting, metrics from safenet.eval_results where run_id=%s", (rid,))
    imp = q("select model, dataset, feature, mean_abs_shap from safenet.feature_importance where run_id=%s", (rid,))
    return run, ev, imp


# ============================================================================ ui helpers
def fig_style(fig, h=320, legend=True):
    fig.update_layout(legend_title_text="", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", color=C["text"], size=12), height=h,
                      margin=dict(l=8, r=8, t=36, b=8), colorway=SERIES, showlegend=legend,
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"),
                      title=dict(font=dict(size=14, color=C["text"]), x=0, xanchor="left"),
                      hoverlabel=dict(bgcolor=C["surface2"], bordercolor=C["border"], font_family="Inter"))
    fig.update_xaxes(gridcolor="rgba(148,163,184,.08)", zeroline=False, linecolor=C["border"])
    fig.update_yaxes(gridcolor="rgba(148,163,184,.08)", zeroline=False, linecolor=C["border"])
    return fig


def kpi(col, label, value, sub="", tone="accent"):
    col.markdown(f"<div class='sn-kpi {tone}'><div class='l'>{label}</div><div class='v'>{value}</div>"
                 f"<div class='d'>{sub}</div></div>", unsafe_allow_html=True)


def section(title: str):
    st.markdown(f"<div class='sn-sec'>{title}</div>", unsafe_allow_html=True)


def ago(ts) -> str:
    if ts is None or pd.isna(ts):
        return "never"
    s = (datetime.now(timezone.utc) - pd.Timestamp(ts).to_pydatetime()).total_seconds()
    if s < 60:
        return f"{int(s)}s ago"
    if s < 3600:
        return f"{int(s // 60)}m ago"
    if s < 86400:
        return f"{int(s // 3600)}h ago"
    return f"{int(s // 86400)}d ago"


def fmt_n(x) -> str:
    x = float(x or 0)
    for unit, d in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= d:
            return f"{x / d:.1f}{unit}"
    return f"{x:,.0f}"


def is_public(ip: str) -> bool:
    try:
        a = ipaddress.ip_address(ip)
        return a.is_global and not a.is_multicast
    except ValueError:
        return False


def header(title: str, subtitle: str):
    s = d_sensor()
    if s.empty:
        pill = "<span class='sn-pill'><span class='sn-dot off'></span>No sensor yet</span>"
    else:
        r = s.iloc[0]
        online = (datetime.now(timezone.utc) - pd.Timestamp(r["last_seen"]).to_pydatetime()).total_seconds() < 90
        dot = "on" if online else "off"
        state = "Sensor online" if online else f"Sensor offline · {ago(r['last_seen'])}"
        pill = (f"<span class='sn-pill'><span class='sn-dot {dot}'></span>{state}</span>"
                f"<span class='sn-pill mono'>{r['hostname']} · {r['interface']}</span>"
                f"<span class='sn-pill mono'>model {r['model_version']}</span>")
    st.markdown(f"<div class='sn-top'><div class='sn-brand'>{SHIELD}<div><div class='t'>{title}</div>"
                f"<div class='s'>{subtitle}</div></div></div><div style='display:flex;gap:8px;flex-wrap:wrap'>{pill}</div></div>",
                unsafe_allow_html=True)


def alert_card(r) -> str:
    sev = r["severity"]
    mitre = "".join(f"<a class='sn-chip' href='{m['url']}' target='_blank'>{m['id']} {m['name']}</a>"
                    for m in (r.get("mitre") or []))
    intel = "".join(f"<a class='sn-chip kev' href='{h['url']}' target='_blank'>⚑ {h['name']}</a>"
                    for h in (r.get("intel_hits") or []))
    return (f"<div class='sn-alert' style='--sev:{SEV.get(sev, C['low'])}'>"
            f"<div style='min-width:74px'><span class='sn-badge {sev}'>{sev}</span></div><div style='flex:1'>"
            f"<div class='m'>#{r['id']} · {r['label']} · {ago(r['created_at'])} · "
            f"<span class='mono'>{r['src_ip']} → {r['dst_ip']}:{r['dst_port']}/{r['proto']}</span></div>"
            f"<div class='x'>{r['summary']}</div><div style='margin-top:6px'>{intel}{mitre}</div></div></div>")


# ============================================================================ pages
def page_overview():
    header("Security overview", "Real-time view of traffic captured by the SafeNet sensor")

    @st.fragment(run_every=10)
    def live():
        k = d_kpis()
        s = d_sensor()
        pk = int(s.iloc[0]["packets_seen"]) if not s.empty else 0
        c = st.columns(6)
        kpi(c[0], "Flows · 24h", fmt_n(k.flows_24h), f"{fmt_n(k.flows_1h)} in the last hour", "accent")
        kpi(c[1], "Packets this session", fmt_n(pk), "captured on the wire", "violet")
        kpi(c[2], "Alerts · 24h", f"{int(k.alerts_24h):,}", "ML + threat intel", "high" if k.alerts_24h else "ok")
        kpi(c[3], "Open alerts", f"{int(k.open_alerts):,}", "awaiting triage", "medium" if k.open_alerts else "ok")
        kpi(c[4], "Threat-intel hits · 7d", f"{int(k.intel_7d):,}", "Feodo / Spamhaus matches", "high" if k.intel_7d else "ok")
        kpi(c[5], "Destinations · 24h", f"{int(k.dests_24h):,}", f"{int(k.suppressed_24h)} out-of-scope suppressed", "accent")

        tl = d_timeline(3)
        left, right = st.columns([2.2, 1])
        with left:
            if tl.empty:
                st.info("Waiting for the first flows from the sensor…")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=tl["t"], y=tl["flows"], name="Flows / min", mode="lines",
                                         line=dict(color=C["accent"], width=2), fill="tozeroy",
                                         fillcolor="rgba(34,211,238,.12)"))
                al = tl[tl["alerts"] > 0]
                fig.add_trace(go.Scatter(x=al["t"], y=al["flows"], name="Minutes with alerts", mode="markers",
                                         marker=dict(color=C["high"], size=9, line=dict(color="#fff", width=1))))
                st.plotly_chart(fig_style(fig.update_layout(title="Traffic · last 3 hours (UTC)"), 300), width="stretch")
        with right:
            fl = d_flows(1500)
            if not fl.empty:
                mix = fl.groupby("proto").size().reset_index(name="n")
                fig = px.pie(mix, names="proto", values="n", hole=.62, title="Protocol mix (recent)")
                fig.update_traces(textinfo="label+percent", marker=dict(line=dict(color=C["bg"], width=2)))
                st.plotly_chart(fig_style(fig, 300, legend=False), width="stretch")

        left, right = st.columns([1.4, 1])
        with left:
            section("Latest alerts")
            al = d_alerts(8)
            if al.empty:
                st.markdown(f"<div class='sn-card'>✅ No alerts. Every flow scored so far looks benign to the models "
                            f"and no contacted IP is on the Feodo Tracker or Spamhaus DROP blocklists.</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown("".join(alert_card(r) for _, r in al.iterrows()), unsafe_allow_html=True)
        with right:
            section("Top remote endpoints (recent flows)")
            fl = d_flows(1500)
            if not fl.empty:
                fl = fl.assign(remote=fl.apply(lambda r: r.dst_ip if is_public(r.dst_ip) else (
                    r.src_ip if is_public(r.src_ip) else None), axis=1)).dropna(subset=["remote"])
                top = (fl.groupby("remote").agg(flows=("remote", "size"), bytes=("bytes", "sum"),
                                                max_score=("p_attack", "max")).sort_values("bytes", ascending=False)
                       .head(10).reset_index())
                st.dataframe(top, hide_index=True, width="stretch", height=380, column_config={
                    "remote": st.column_config.TextColumn("Remote IP"),
                    "bytes": st.column_config.NumberColumn("Bytes", format="compact"),
                    "max_score": st.column_config.ProgressColumn("Max attack score", min_value=0, max_value=1, format="%.3f"),
                })

    live()


def page_traffic():
    header("Live traffic", "Every flow the sensor has scored, newest first")

    @st.fragment(run_every=10)
    def body():
        fl = d_flows(2000)
        if fl.empty:
            st.info("No flows yet.")
            return
        f1, f2, f3 = st.columns([1.2, 1, 1])
        protos = f1.multiselect("Protocol", sorted(fl["proto"].unique()), default=sorted(fl["proto"].unique()))
        view = f2.segmented_control("Show", ["All", "Alerts", "Suppressed"], default="All")
        minp = f3.slider("Min attack score", 0.0, 1.0, 0.0, 0.05)
        v = fl[fl["proto"].isin(protos) & (fl["p_attack"] >= minp)]
        if view == "Alerts":
            v = v[v["is_alert"]]
        elif view == "Suppressed":
            v = v[v["suppressed"].notna()]
        c = st.columns(4)
        kpi(c[0], "Flows shown", f"{len(v):,}")
        kpi(c[1], "Bytes", fmt_n(v["bytes"].sum()), tone="violet")
        kpi(c[2], "Median attack score", f"{v['p_attack'].median():.4f}" if len(v) else "–", tone="ok")
        kpi(c[3], "Autoencoder anomalies", f"{int(v['ae_anomaly'].sum()):,}", tone="medium")
        st.dataframe(v, hide_index=True, width="stretch", height=430, column_config={
            "ts_end": st.column_config.DatetimeColumn("Time (UTC)", format="HH:mm:ss"),
            "p_attack": st.column_config.ProgressColumn("Attack score", min_value=0, max_value=1, format="%.4f"),
            "ae_score": st.column_config.NumberColumn("AE error", format="%.3f"),
            "bytes": st.column_config.NumberColumn("Bytes", format="compact"),
            "packets": st.column_config.NumberColumn("Pkts", format="%d"),
            "duration_s": st.column_config.NumberColumn("Dur (s)", format="%.2f"),
            "is_alert": st.column_config.CheckboxColumn("Alert"),
            "ae_anomaly": st.column_config.CheckboxColumn("AE anomaly"),
            "suppressed": st.column_config.TextColumn("Suppressed (out of model scope)"),
        })
        a, b = st.columns(2)
        fig = px.histogram(v, x="p_attack", nbins=40, color="proto", log_y=True, title="Attack-score distribution")
        a.plotly_chart(fig_style(fig, 300), width="stretch")
        fig = px.scatter(v, x="ae_score", y="p_attack", color="proto", log_x=True, opacity=.7,
                         hover_data=["src_ip", "dst_ip", "dst_port"], title="Supervised score vs autoencoder error")
        b.plotly_chart(fig_style(fig, 300), width="stretch")

    body()


def ip_intel_panel(ip: str | None):
    if not ip:
        st.caption("Both endpoints are on the local network, so no external lookup was made.")
        return
    it = d_ip_intel()
    row = it[it["ip"] == ip]
    if row.empty:
        st.caption(f"Intelligence for {ip} is being collected by the sensor (ipwho.is, Shodan InternetDB, GreyNoise)…")
        return
    d = row.iloc[0]["data"]
    geo, sh, gn = d.get("geo") or {}, d.get("shodan") or {}, d.get("greynoise") or {}
    st.markdown(f"**{geo.get('flag', '')} {ip}** · {geo.get('city') or '?'}, {geo.get('country') or '?'} · "
                f"AS{geo.get('asn') or '?'} {geo.get('org') or ''}")
    chips = "".join(f"<span class='sn-chip'>port {p}</span>" for p in sh.get("ports", [])[:20])
    chips += "".join(f"<span class='sn-chip'>{h}</span>" for h in sh.get("hostnames", [])[:6])
    chips += "".join(f"<span class='sn-chip'>#{t}</span>" for t in sh.get("tags", []))
    kev = set(sh.get("kev", []))
    all_v = sh.get("vulns", [])
    vulns = sorted(v for v in all_v if v in kev) + sorted((v for v in all_v if v not in kev), reverse=True)  # KEV first
    chips += "".join(f"<a class='sn-chip {'kev' if v in kev else ''}' target='_blank' "
                     f"href='https://nvd.nist.gov/vuln/detail/{v}'>{'⚠ CISA KEV · ' if v in kev else ''}{v}</a>"
                     for v in vulns[:30])
    st.markdown(chips or "<span class='sn-chip'>Shodan: no exposed services recorded</span>", unsafe_allow_html=True)
    if vulns:
        st.caption(f"Shodan lists {len(vulns)} CVEs for software on this host"
                   + (f", {len(kev)} of them in CISA's Known Exploited Vulnerabilities catalog" if kev else "")
                   + (f" (showing 30)" if len(vulns) > 30 else "") + ".")
    if gn:
        cls = "known benign service (RIOT)" if gn.get("riot") else ("internet scanner (noise)" if gn.get("noise")
                                                                     else "not seen scanning the internet")
        st.markdown(f"GreyNoise: **{cls}**{' · ' + gn['name'] if gn.get('name') else ''} · "
                    f"[view](https://viz.greynoise.io/ip/{ip}) · [Shodan](https://www.shodan.io/host/{ip}) · "
                    f"[AbuseIPDB](https://www.abuseipdb.com/check/{ip})")
    for h in d.get("blocklists") or []:
        st.error(f"Listed on {h['name']} {h.get('netblock', '')} · [details]({h['url']})")
    src = " · ".join(f"{k}: {v}" for k, v in (d.get("sources") or {}).items())
    st.caption(f"Looked up {ago(row.iloc[0]['fetched_at'])} · {src}")


def page_alerts():
    header("Alert triage", "Explainable detections with SHAP, MITRE ATT&CK and threat intelligence")
    al = d_alerts(500)
    if al.empty:
        st.markdown("<div class='sn-card'><h4>No alerts yet</h4>The sensor raises an alert when the adversarially "
                    "trained XGBoost scores a TCP/UDP flow ≥ 0.90, when the autoencoder reconstruction error is "
                    "≥ 5× its benign threshold, when a behavioural rule fires (port scan, sweep, brute force) or when "
                    "the remote IP appears on the abuse.ch Feodo Tracker or Spamhaus DROP lists.</div>", unsafe_allow_html=True)
        return
    f1, f2, f3 = st.columns([1, 1, 1.4])
    sev = f1.pills("Severity", ["high", "medium", "low"], default=["high", "medium", "low"], selection_mode="multi")
    status = f2.pills("Status", ["open", "acknowledged", "resolved"], default=["open"], selection_mode="multi")
    labels = f3.multiselect("Type", sorted(al["label"].unique()), default=sorted(al["label"].unique()))
    v = al[al["severity"].isin(sev or []) & al["status"].isin(status or []) & al["label"].isin(labels)]
    c = st.columns(4)
    kpi(c[0], "In queue", f"{len(v):,}")
    kpi(c[1], "High", f"{int((v['severity'] == 'high').sum())}", tone="high")
    kpi(c[2], "Medium", f"{int((v['severity'] == 'medium').sum())}", tone="medium")
    kpi(c[3], "Threat intel", f"{int((v['label'] == 'ThreatIntel').sum())}", tone="violet")
    if v.empty:
        st.success("Queue is empty for this filter.")
        return
    left, right = st.columns([1, 1.25])
    with left:
        section("Queue")
        table = v[["id", "created_at", "severity", "label", "p_attack", "remote_ip", "dst_port", "proto", "status"]]
        ev = st.dataframe(table, hide_index=True, width="stretch", height=560, on_select="rerun",
                          selection_mode="single-row", column_config={
                              "created_at": st.column_config.DatetimeColumn("Time (UTC)", format="MMM D HH:mm"),
                              "p_attack": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.3f"),
                              "remote_ip": "Remote IP"})
        rows = ev.selection.rows if ev and ev.selection else []
    a = v.iloc[rows[0]] if rows else v.iloc[0]
    with right:
        section(f"Alert #{a['id']}")
        st.markdown(alert_card(a), unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        ex = a["explanation"] or {}
        if ex.get("detector") == "behavioral":
            evd = ex.get("evidence", {})
            key, val = next(((k, v) for k, v in evd.items() if k in ("distinct_ports", "distinct_hosts", "connections")),
                            ("events", "–"))
            m1.metric("Detector", "Behavioural rule")
            m2.metric(key.replace("_", " ").capitalize(), f"{val}", help=f"threshold {evd.get('threshold')} in "
                                                                           f"{int(evd.get('window_s', 60))} s")
        else:
            m1.metric("XGBoost attack probability", f"{a['p_attack']:.3f}")
            m2.metric("Autoencoder error", f"{a['ae_score']:.3f}",
                      help=f"benign threshold {ex.get('ae_threshold', float('nan')):.3f}")
        m3.metric("Status", a["status"])
        t1, t2, t3 = st.tabs(["Why", "Remote IP intelligence", "Flow features"])
        with t1:
            expl = a["explanation"] or {}
            if expl.get("detector") == "behavioral":
                st.markdown(f"<div class='sn-card'><h4>Behavioural rule · {expl.get('rule')}</h4>"
                            f"This alert comes from SafeNet's multi-flow detector, which aggregates flows from the "
                            f"same source over a {int(expl.get('evidence', {}).get('window_s', 60))} s window. Per-flow "
                            f"models see each probe as an ordinary short connection.</div>", unsafe_allow_html=True)
                evd = dict(expl.get("evidence", {}))
                samples = evd.pop("sample_ports", None) or evd.pop("sample_hosts", None) or []
                st.dataframe(pd.DataFrame([{"evidence": k.replace("_", " "), "value": str(v)} for k, v in evd.items()]),
                             hide_index=True, width="stretch")
                if samples:
                    st.markdown("".join(f"<span class='sn-chip mono'>{x}</span>" for x in samples), unsafe_allow_html=True)
            contrib = pd.DataFrame(expl.get("contributions", []))
            if not contrib.empty:
                contrib["label"] = contrib.apply(lambda r: f"{FEATURE_LABELS.get(r['feature'], r['feature'])} = "
                                                           f"{r['value']:,.4g}", axis=1)
                base = (a["explanation"] or {}).get("base_value", 0.0)
                fig = go.Figure(go.Waterfall(
                    orientation="h", y=["Model baseline"] + contrib["label"].tolist() + ["This flow"],
                    x=[base] + contrib["shap"].tolist() + [0], measure=["absolute"] + ["relative"] * len(contrib) + ["total"],
                    increasing=dict(marker=dict(color=C["high"])), decreasing=dict(marker=dict(color=C["low"])),
                    totals=dict(marker=dict(color=C["accent2"])), connector=dict(line=dict(color=C["border"]))))
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_style(fig.update_layout(title="Contribution to attack log-odds"), 360, False),
                                width="stretch")
                st.caption("Red bars push the flow towards *attack*, blue towards *benign* (top features by |SHAP|).")
        with t2:
            ip_intel_panel(a["remote_ip"])
        with t3:
            feats = a["features"] or {}
            st.dataframe(pd.DataFrame([{"feature": FEATURE_LABELS.get(k, k), "value": v_} for k, v_ in feats.items()]),
                         hide_index=True, width="stretch", height=300)
        b1, b2, _ = st.columns([1, 1, 2])
        if a["status"] == "open" and b1.button("Acknowledge", type="primary", width="stretch"):
            execute("update safenet.alerts set status='acknowledged' where id=%s", (int(a["id"]),))
            d_alerts.clear(); d_kpis.clear(); st.rerun()
        if a["status"] != "resolved" and b2.button("Resolve", width="stretch"):
            execute("update safenet.alerts set status='resolved' where id=%s", (int(a["id"]),))
            d_alerts.clear(); d_kpis.clear(); st.rerun()


def page_intel():
    header("Threat intelligence", "Live blocklists and enrichment from public threat-intel services")
    fd = d_feeds()
    section("Feeds matched locally against every flow")
    if fd.empty:
        st.info("The sensor has not downloaded the feeds yet.")
    else:
        cols = st.columns(len(fd))
        for col, (_, f) in zip(cols, fd.iterrows()):
            tone = "ok" if f["ok"] else "high"
            col.markdown(f"<div class='sn-kpi {tone}'><div class='l'>{f['name']}</div><div class='v'>{int(f['entries']):,}</div>"
                         f"<div class='d'>{'updated ' + ago(f['fetched_at']) if f['ok'] else 'stale: ' + str(f['error'])[:60]}"
                         f" · <a href='{f['url']}' target='_blank'>source</a></div></div>", unsafe_allow_html=True)
    it = d_ip_intel()
    section("Remote IPs from alerts, enriched via ipwho.is · Shodan InternetDB · GreyNoise")
    if it.empty:
        st.markdown("<div class='sn-card'>No alerted public IPs yet. When an alert involves a public IP the sensor looks "
                    "it up once a day and caches the result here.</div>", unsafe_allow_html=True)
        return
    rows = []
    for _, r in it.iterrows():
        d = r["data"]
        g, s, gn = d.get("geo") or {}, d.get("shodan") or {}, d.get("greynoise") or {}
        rows.append(dict(ip=r["ip"], alerts=int(r["alerts"]), country=g.get("country"), city=g.get("city"),
                         lat=g.get("lat"), lon=g.get("lon"), org=g.get("org"), asn=g.get("asn"),
                         open_ports=len(s.get("ports", [])), cves=len(s.get("vulns", [])), kev=len(s.get("kev", [])),
                         greynoise=("RIOT" if gn.get("riot") else "noise" if gn.get("noise") else "quiet"),
                         blocklisted=bool(d.get("blocklists"))))
    df = pd.DataFrame(rows)
    geo = df.dropna(subset=["lat", "lon"])
    if not geo.empty:
        geo = geo.assign(status=geo["blocklisted"].map({True: "On a blocklist", False: "Not blocklisted"}))
        fig = px.scatter_geo(geo, lat="lat", lon="lon", size=geo["alerts"].clip(lower=1), color="status",
                             hover_name="ip", hover_data=["org", "country", "open_ports", "cves"],
                             color_discrete_map={"On a blocklist": C["high"], "Not blocklisted": C["accent"]},
                             projection="natural earth")
        fig.update_geos(bgcolor="rgba(0,0,0,0)", showcountries=True, countrycolor="#334155", showland=True,
                        landcolor=C["surface2"], showocean=True, oceancolor=C["bg"], coastlinecolor="#334155")
        st.plotly_chart(fig_style(fig.update_layout(title="Where alerted traffic went"), 420), width="stretch")
    st.dataframe(df.drop(columns=["lat", "lon"]), hide_index=True, width="stretch", column_config={
        "kev": st.column_config.NumberColumn("CISA KEV CVEs"), "cves": "Shodan CVEs", "open_ports": "Open ports"})
    pick = st.selectbox("Inspect IP", df["ip"].tolist())
    with st.container(border=True):
        ip_intel_panel(pick)


def _metrics(ev: pd.DataFrame, exp: str) -> pd.DataFrame:
    rows = []
    for _, r in ev[ev["experiment"] == exp].iterrows():
        m = {k: v for k, v in r["metrics"].items() if isinstance(v, (int, float))}
        rows.append(dict(model=r["model"], train=DS.get(r["train_ds"], r["train_ds"]),
                         test=DS.get(r["test_ds"], r["test_ds"]), setting=r["setting"], **m))
    return pd.DataFrame(rows)


def page_models():
    header("Model lab", "How well SafeNet detects, generalises and resists evasion: measured on real datasets")
    run, ev, imp = d_eval()
    if run.empty:
        st.info("No training run has been pushed yet. Run `python -m scripts.train`.")
        return
    r0, notes = run.iloc[0], run.iloc[0]["notes"]
    ind, cr, rb, at = (_metrics(ev, e) for e in ("in_dataset", "cross_dataset", "robustness", "adv_training"))

    def pick(df, **kw):
        s = df
        for k, v in kw.items():
            s = s[s[k] == v]
        return s.iloc[0] if not s.empty else None

    c = st.columns(4)
    x = pick(ind, model="XGBoost", train="CIC-IDS2017", setting="native")
    kpi(c[0], "In-dataset F1 · CIC-IDS2017", f"{x['f1']:.3f}" if x is not None else "–", "XGBoost, native features", "ok")
    x = pick(ind, model="XGBoost", train="UNSW-NB15", setting="native")
    kpi(c[1], "In-dataset F1 · UNSW-NB15", f"{x['f1']:.3f}" if x is not None else "–", "XGBoost, native features", "accent")
    x = pick(cr, model="XGBoost", train="CIC-IDS2017", test="UNSW-NB15")
    kpi(c[2], "Cross-dataset F1", f"{x['f1']:.3f}" if x is not None else "–", "trained CIC → tested UNSW", "high")
    a1 = pick(rb, model="XGBoost", setting="eps=1.0")
    a2 = pick(rb, model="XGBoost-AdvTrained", setting="eps=1.0")
    kpi(c[3], "Detection under attack (ε=1)",
        f"{a2['detection_rate']:.0%}" if a2 is not None else "–",
        f"adv-trained vs {a1['detection_rate']:.0%} standard" if a1 is not None else "", "violet")
    st.caption(f"Model `{r0['version']}` · trained {pd.Timestamp(r0['created_at']):%Y-%m-%d %H:%M} UTC · "
               f"CIC-IDS2017 {notes.get('cic_train', 0):,} train / {notes.get('cic_test', 0):,} test flows · "
               f"UNSW-NB15 official split {notes.get('unsw_train', 0):,} / {notes.get('unsw_test', 0):,}")

    t1, t2, t3, t4, t5 = st.tabs(["In-dataset", "Cross-dataset", "Rare attacks", "Adversarial robustness", "Explainability"])
    with t1:
        d = ind[ind["f1"].notna()] if "f1" in ind else ind
        cols = [k for k in ["train", "setting", "model", "precision", "recall", "f1", "fpr", "roc_auc", "macro_f1_category"] if k in d]
        st.dataframe(d[cols].rename(columns={"train": "dataset", "setting": "features"}), hide_index=True, width="stretch",
                     column_config={k: st.column_config.ProgressColumn(k, min_value=0, max_value=1, format="%.4f")
                                    for k in ["precision", "recall", "f1", "roc_auc"]})
        fig = px.bar(d, x="model", y="f1", color="setting", barmode="group", facet_col="train", range_y=[0, 1],
                     title="F1 by model · native vs shared feature set")
        st.plotly_chart(fig_style(fig, 340), width="stretch")
    with t2:
        metric = st.segmented_control("Metric", ["f1", "recall", "precision", "roc_auc", "fpr"], default="f1")
        cc = st.columns(3)
        for i, mdl in enumerate(["RandomForest", "XGBoost", "Autoencoder"]):
            piv = cr[cr["model"] == mdl].pivot(index="train", columns="test", values=metric)
            fig = px.imshow(piv, text_auto=".3f", zmin=0, zmax=1, aspect="auto",
                            color_continuous_scale="Teal" if metric != "fpr" else "Reds", title=mdl)
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(title="tested on")
            fig.update_yaxes(title="trained on")
            cc[i].plotly_chart(fig_style(fig, 300, False), width="stretch")
        st.caption("Diagonal = same dataset. Off-diagonal = a model trained on one network tested on a different one; "
                   "the drop is the real-world generalisation gap SafeNet measures.")
    with t3:
        rows = []
        for _, r in ev[ev["experiment"] == "in_dataset"].iterrows():
            for lab, dd in (r["metrics"].get("per_label_detection") or {}).items():
                rows.append(dict(dataset=DS.get(r["train_ds"]), features=r["setting"], model=r["model"], label=lab,
                                 n_test=dd["n"], detected=dd["detected"]))
        if rows:
            pl = pd.DataFrame(rows)
            a, b = st.columns(2)
            dsel = a.segmented_control("Dataset", sorted(pl["dataset"].dropna().unique()), default="CIC-IDS2017")
            fsel = b.segmented_control("Features", sorted(pl["features"].unique()), default="native")
            sub = pl[(pl["dataset"] == dsel) & (pl["features"] == fsel)].sort_values("n_test", ascending=False)
            fig = px.bar(sub, x="detected", y="label", color="model", barmode="group", orientation="h",
                         hover_data=["n_test"], range_x=[0, 1], title="Detection rate per attack label (rarest at top)")
            st.plotly_chart(fig_style(fig, max(380, 26 * sub["label"].nunique() * 2)), width="stretch")
            st.caption("For BENIGN / Normal the bar is the false-alarm rate. Hover a bar to see how many test flows it has.")
    with t4:
        if not rb.empty:
            fig = px.line(rb.sort_values("eps"), x="eps", y="detection_rate", color="model", markers=True,
                          labels={"eps": "attacker budget ε (pad / slow down / dummy packets)",
                                  "detection_rate": "attacks still detected"},
                          title="Adaptive black-box evasion attack")
            st.plotly_chart(fig_style(fig.update_layout(yaxis_range=[0, 1.02]), 380), width="stretch")
        if not at.empty:
            st.markdown("**Clean-traffic cost of adversarial training** (combined test set)")
            st.dataframe(at.drop(columns=["train", "test", "setting"]), hide_index=True, width="stretch")
    with t5:
        if not imp.empty:
            a, b = st.columns(2)
            which = a.selectbox("Model", imp["model"].unique().tolist())
            d2 = b.selectbox("Dataset", imp[imp["model"] == which]["dataset"].unique().tolist(),
                             format_func=lambda x: DS.get(x, x))
            sub = imp[(imp["model"] == which) & (imp["dataset"] == d2)].nlargest(15, "mean_abs_shap")
            sub = sub.assign(feature=sub["feature"].map(lambda f: FEATURE_LABELS.get(f, f)))
            fig = px.bar(sub.iloc[::-1], x="mean_abs_shap", y="feature", orientation="h",
                         title="Mean |SHAP| (global feature importance)")
            fig.update_traces(marker_color=C["accent"])
            st.plotly_chart(fig_style(fig, 460, False), width="stretch")


def page_about():
    header("About SafeNet", "Architecture, data sources and integrations")
    st.markdown(f"""
<div class='sn-card'><h4>Pipeline</h4>
Packets captured on a real interface → bidirectional flows (5-tuple, FIN/RST, 15 s idle / 120 s active timeout,
TSO super-packets re-split to wire frames) → 22 shared flow features computable identically from CIC-IDS2017,
UNSW-NB15 and live traffic → <b>XGBoost (adversarially trained)</b> attack probability, <b>XGBoost</b> attack type,
<b>PyTorch autoencoder</b> anomaly score → SHAP explanation → MITRE ATT&CK mapping → threat-intel match and
enrichment → PostgreSQL (Supabase) → this console (Streamlit, Google sign-in).</div>
<div class='sn-card'><h4>Data and services</h4>
<a class='sn-chip' target='_blank' href='https://www.unb.ca/cic/datasets/ids-2017.html'>CIC-IDS2017 · UNB</a>
<a class='sn-chip' target='_blank' href='https://research.unsw.edu.au/projects/unsw-nb15-dataset'>UNSW-NB15 · UNSW Canberra</a>
<a class='sn-chip' target='_blank' href='https://feodotracker.abuse.ch/'>abuse.ch Feodo Tracker</a>
<a class='sn-chip' target='_blank' href='https://www.spamhaus.org/drop/'>Spamhaus DROP</a>
<a class='sn-chip' target='_blank' href='https://www.cisa.gov/known-exploited-vulnerabilities-catalog'>CISA KEV</a>
<a class='sn-chip' target='_blank' href='https://internetdb.shodan.io/'>Shodan InternetDB</a>
<a class='sn-chip' target='_blank' href='https://viz.greynoise.io/'>GreyNoise Community</a>
<a class='sn-chip' target='_blank' href='https://ipwho.is/'>ipwho.is</a>
<a class='sn-chip' target='_blank' href='https://attack.mitre.org/'>MITRE ATT&CK</a>
<a class='sn-chip' target='_blank' href='https://supabase.com/'>Supabase Postgres</a>
<a class='sn-chip' target='_blank' href='https://github.com/Shivansharmaaaa/safenet-ids'>Source on GitHub</a>
</div>
<div class='sn-card'><h4>Alert policy</h4>
A flow raises an alert when the adversarially-trained XGBoost scores it ≥ 0.90, when the autoencoder error is ≥ 5×
its benign threshold, or when the remote IP is on a blocklist. Behavioural rules over a 60 s window add multi-flow
detections the per-flow models cannot see: vertical port scans (≥ 25 ports on one host), horizontal sweeps
(≥ 25 hosts on one port) and brute force (≥ 15 short sessions to SSH/RDP/FTP/SMB/database ports). ICMP and other non-TCP/UDP protocols, and local
multicast/broadcast discovery (mDNS, SSDP), are outside what the models were trained on: they are still scored
and stored (marked <i>suppressed</i>) and still checked against the blocklists, but cannot raise an ML alert.</div>
<div class='sn-card'><h4>Privacy</h4>
Blocklist matching is done locally on the sensor. Only the public IP of an <i>alerted</i> flow is ever sent to the
enrichment services (cached for 24 h). Captured traffic lives in a private <code>safenet</code> schema with
row-level security, not exposed through Supabase's public API, and this console requires Google sign-in.</div>
""", unsafe_allow_html=True)


# ============================================================================ navigation
with st.sidebar:
    st.markdown(f"<div style='display:flex;align-items:center;gap:10px;margin:4px 0 14px'>{SHIELD.replace('40', '30')}"
                f"<div style='font-weight:800;font-size:1.1rem'>SafeNet</div></div>", unsafe_allow_html=True)

pages = [
    st.Page(page_overview, title="Overview", icon=":material/space_dashboard:", default=True),
    st.Page(page_traffic, title="Live traffic", icon=":material/lan:", url_path="traffic"),
    st.Page(page_alerts, title="Alerts", icon=":material/notification_important:", url_path="alerts"),
    st.Page(page_intel, title="Threat intel", icon=":material/travel_explore:", url_path="intel"),
    st.Page(page_models, title="Model lab", icon=":material/science:", url_path="models"),
    st.Page(page_about, title="About", icon=":material/info:", url_path="about"),
]
nav = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.divider()
    if _auth_configured() and st.user.is_logged_in:
        st.caption(f"Signed in as **{st.user.email}**")
        st.button("Sign out", on_click=st.logout, width="stretch")
    st.caption("Data: Supabase Postgres · refreshes every 10 s")

nav.run()
