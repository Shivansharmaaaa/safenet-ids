"""SafeNet dashboard: live flows, explainable alerts and the evaluation report.
Every number on this page is read from the SafeNet PostgreSQL (Supabase) database."""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg
import streamlit as st

st.set_page_config(page_title="SafeNet — Network Intrusion Detection", page_icon="🛡️", layout="wide")

FEATURE_LABELS = {
    "duration_s": "Flow duration (s)", "fwd_pkts": "Packets sent by initiator", "bwd_pkts": "Packets sent by responder",
    "fwd_bytes": "Bytes sent by initiator", "bwd_bytes": "Bytes sent by responder",
    "fwd_pkt_mean": "Avg packet size from initiator", "bwd_pkt_mean": "Avg packet size from responder",
    "bytes_per_s": "Throughput (bytes/s)", "pkts_per_s": "Packet rate (pkts/s)",
    "fwd_iat_ms": "Gap between initiator packets (ms)", "bwd_iat_ms": "Gap between responder packets (ms)",
    "fwd_win": "Initiator TCP window", "bwd_win": "Responder TCP window", "is_tcp": "TCP flow",
    "byte_ratio": "Response/request byte ratio", "svc_http": "HTTP service", "svc_tls": "TLS/HTTPS service",
    "svc_dns": "DNS service", "svc_ssh": "SSH service", "svc_ftp": "FTP service", "svc_mail": "Mail service",
    "svc_other": "Other service",
}
DS_NAMES = {"cicids2017": "CIC-IDS2017", "unswnb15": "UNSW-NB15", "combined": "Both"}


# ----------------------------------------------------------------------------- auth
def _secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)


def require_login() -> None:
    try:
        auth_configured = "auth" in st.secrets
    except Exception:
        auth_configured = False
    if not auth_configured:
        return  # local development without OAuth configured
    if not st.user.is_logged_in:
        st.title("🛡️ SafeNet")
        st.write("Explainable and robust network intrusion detection. Sign in to view the live dashboard.")
        st.button("Sign in with Google", on_click=st.login, args=("google",), type="primary")
        st.stop()
    allowed = [e.strip().lower() for e in str(_secret("ALLOWED_EMAILS", "")).split(",") if e.strip()]
    if allowed and str(st.user.email).lower() not in allowed:
        st.error(f"{st.user.email} is not authorised to view this dashboard.")
        st.button("Sign out", on_click=st.logout)
        st.stop()


require_login()


# ----------------------------------------------------------------------------- data access
@st.cache_resource
def _conn_url() -> str:
    url = _secret("DATABASE_URL")
    if not url:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            url = os.environ.get("DATABASE_URL")
        except ImportError:
            pass
    if not url:
        st.error("DATABASE_URL is not configured.")
        st.stop()
    return url


def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    with psycopg.connect(_conn_url(), prepare_threshold=None, connect_timeout=15) as conn:
        cur = conn.execute(sql, params)
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


@st.cache_data(ttl=10)
def live_stats():
    sensors = q("select hostname, interface, model_version, started_at, last_seen, packets_seen, flows_scored "
                "from safenet.sensors order by last_seen desc")
    kpi = q("""select
                 (select count(*) from safenet.flows where ts_end > now() - interval '24 hours') as flows_24h,
                 (select count(*) from safenet.alerts where created_at > now() - interval '24 hours') as alerts_24h,
                 (select count(*) from safenet.alerts where status = 'open') as open_alerts,
                 (select count(distinct src_ip) from safenet.flows where ts_end > now() - interval '24 hours') as hosts_24h""")
    timeline = q("""select date_trunc('minute', ts_end) as minute, count(*) as flows,
                           sum(case when is_alert then 1 else 0 end) as alerts
                    from safenet.flows where ts_end > now() - interval '3 hours' group by 1 order by 1""")
    recent = q("""select ts_end, src_ip, src_port, dst_ip, dst_port, proto, pred_label, p_attack, ae_score, is_alert
                  from safenet.flows order by ts_end desc limit 200""")
    proto = q("""select proto, count(*) n from safenet.flows where ts_end > now() - interval '24 hours'
                 group by 1 order by 2 desc""")
    return sensors, kpi, timeline, recent, proto


@st.cache_data(ttl=10)
def alerts_df():
    return q("""select id, created_at, severity, label, p_attack, ae_score, src_ip, dst_ip, dst_port, proto,
                       summary, explanation, status, flow_id
                from safenet.alerts order by created_at desc limit 500""")


@st.cache_data(ttl=300)
def eval_data():
    run = q("select id, version, created_at, notes from safenet.model_runs order by created_at desc limit 1")
    if run.empty:
        return run, pd.DataFrame(), pd.DataFrame()
    rid = int(run.iloc[0]["id"])
    ev = q("select experiment, model, train_ds, test_ds, setting, metrics from safenet.eval_results where run_id=%s",
           (rid,))
    imp = q("select model, dataset, feature, mean_abs_shap from safenet.feature_importance where run_id=%s", (rid,))
    return run, ev, imp


# ----------------------------------------------------------------------------- layout
st.title("🛡️ SafeNet")
st.caption("Explainable and robust network intrusion detection: Random Forest, XGBoost and an Autoencoder "
           "trained on CIC-IDS2017 and UNSW-NB15, scoring real traffic captured by the SafeNet sensor.")
if st.user.is_logged_in if hasattr(st.user, "is_logged_in") else False:
    with st.sidebar:
        st.write(f"Signed in as **{st.user.email}**")
        st.button("Sign out", on_click=st.logout)

tab_live, tab_alerts, tab_eval, tab_about = st.tabs(["📡 Live monitor", "🚨 Alerts", "📊 Model performance", "ℹ️ About"])


@st.fragment(run_every=10)
def live_panel():
    sensors, kpi, timeline, recent, proto = live_stats()
    if sensors.empty:
        st.info("No sensor has reported yet. Start one with `sudo .venv/bin/python -m scripts.sensor`.")
        return
    now = pd.Timestamp.now(tz="UTC")
    s = sensors.iloc[0]
    online = (now - pd.Timestamp(s["last_seen"])).total_seconds() < 60
    st.markdown(f"**Sensor** `{s['hostname']}` on `{s['interface']}` — "
                + ("🟢 online" if online else f"🔴 offline (last seen {pd.Timestamp(s['last_seen']):%Y-%m-%d %H:%M UTC})")
                + f" · model `{s['model_version']}` · {int(s['packets_seen']):,} packets · {int(s['flows_scored']):,} flows")
    k = kpi.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flows (24h)", f"{int(k.flows_24h):,}")
    c2.metric("Alerts (24h)", f"{int(k.alerts_24h):,}")
    c3.metric("Open alerts", f"{int(k.open_alerts):,}")
    c4.metric("Active hosts (24h)", f"{int(k.hosts_24h):,}")
    if not timeline.empty:
        fig = go.Figure()
        fig.add_bar(x=timeline["minute"], y=timeline["flows"], name="Flows", marker_color="#4C78A8")
        fig.add_bar(x=timeline["minute"], y=timeline["alerts"], name="Alerts", marker_color="#E45756")
        fig.update_layout(barmode="overlay", height=280, margin=dict(l=10, r=10, t=30, b=10),
                          title="Flows scored per minute (last 3 hours)", legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
    left, right = st.columns([3, 1])
    with left:
        st.subheader("Most recent flows")
        r = recent.copy()
        r["p_attack"] = r["p_attack"].round(4)
        r["ae_score"] = r["ae_score"].round(3)
        st.dataframe(r, width="stretch", height=380, hide_index=True)
    with right:
        st.subheader("Protocols (24h)")
        if not proto.empty:
            st.plotly_chart(px.pie(proto, names="proto", values="n", hole=0.5).update_layout(
                height=300, margin=dict(l=0, r=0, t=0, b=0), showlegend=True), width="stretch")


with tab_live:
    live_panel()


def shap_chart(expl: dict):
    c = pd.DataFrame(expl.get("contributions", []))
    if c.empty:
        return None
    c["label"] = c.apply(lambda r: f"{r['description']} = {r['value']:,.4g}", axis=1)
    c = c.iloc[::-1]
    fig = go.Figure(go.Bar(x=c["shap"], y=c["label"], orientation="h",
                           marker_color=["#E45756" if v > 0 else "#4C78A8" for v in c["shap"]]))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10),
                      title="SHAP contribution to 'attack' score (red pushes towards attack)")
    return fig


with tab_alerts:
    al = alerts_df()
    if al.empty:
        st.success("No alerts recorded yet.")
    else:
        f1, f2 = st.columns(2)
        sev = f1.multiselect("Severity", ["high", "medium", "low"], default=["high", "medium", "low"])
        status = f2.multiselect("Status", ["open", "acknowledged"], default=["open", "acknowledged"])
        view = al[al["severity"].isin(sev) & al["status"].isin(status)]
        st.dataframe(view.drop(columns=["explanation", "flow_id"]), width="stretch", height=300, hide_index=True)
        if not view.empty:
            pick = st.selectbox("Explain alert", view["id"].tolist(),
                                format_func=lambda i: f"#{i} · " + view.set_index("id").loc[i, "summary"][:110])
            a = view.set_index("id").loc[pick]
            st.markdown(f"### Alert #{pick}: {a['label']} ({a['severity']})")
            st.write(a["summary"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Attack probability (XGBoost)", f"{a['p_attack']:.3f}")
            m2.metric("Autoencoder score", f"{a['ae_score']:.3f}",
                      help=f"threshold {a['explanation'].get('ae_threshold', float('nan')):.3f}")
            m3.metric("Flow", f"{a['src_ip']} → {a['dst_ip']}:{a['dst_port']}/{a['proto']}")
            fig = shap_chart(a["explanation"])
            if fig:
                st.plotly_chart(fig, width="stretch")
            if a["status"] == "open" and st.button("Acknowledge alert"):
                with psycopg.connect(_conn_url(), prepare_threshold=None) as conn:
                    conn.execute("update safenet.alerts set status='acknowledged' where id=%s", (int(pick),))
                alerts_df.clear()
                st.rerun()


def _metrics_frame(ev: pd.DataFrame, experiment: str) -> pd.DataFrame:
    sub = ev[ev["experiment"] == experiment]
    rows = []
    for _, r in sub.iterrows():
        m = {k: v for k, v in r["metrics"].items() if isinstance(v, (int, float))}
        rows.append(dict(model=r["model"], train=DS_NAMES.get(r["train_ds"], r["train_ds"]),
                         test=DS_NAMES.get(r["test_ds"], r["test_ds"]), setting=r["setting"], **m))
    return pd.DataFrame(rows)


with tab_eval:
    run, ev, imp = eval_data()
    if run.empty:
        st.info("No training run has been pushed to the database yet.")
    else:
        r0 = run.iloc[0]
        notes = r0["notes"]
        st.markdown(f"Model version `{r0['version']}`, trained {pd.Timestamp(r0['created_at']):%Y-%m-%d %H:%M UTC}. "
                    f"CIC-IDS2017: {notes.get('cic_train', 0):,} train / {notes.get('cic_test', 0):,} test flows · "
                    f"UNSW-NB15 official split: {notes.get('unsw_train', 0):,} / {notes.get('unsw_test', 0):,}.")

        st.subheader("1 · In-dataset detection")
        ind = _metrics_frame(ev, "in_dataset")
        ind = ind[ind.get("f1").notna()] if "f1" in ind else ind
        cols = [c for c in ["train", "setting", "model", "precision", "recall", "f1", "fpr", "roc_auc", "macro_f1_category"]
                if c in ind]
        st.dataframe(ind[cols].rename(columns={"train": "dataset", "setting": "features"}).round(4),
                     width="stretch", hide_index=True)

        st.subheader("2 · Cross-dataset generalisation")
        cr = _metrics_frame(ev, "cross_dataset")
        if not cr.empty:
            metric = st.radio("Metric", ["f1", "recall", "precision", "roc_auc", "fpr"], horizontal=True)
            cc = st.columns(3)
            for i, mdl in enumerate(["RandomForest", "XGBoost", "Autoencoder"]):
                piv = cr[cr["model"] == mdl].pivot(index="train", columns="test", values=metric)
                fig = px.imshow(piv, text_auto=".3f", color_continuous_scale="RdYlGn" if metric != "fpr" else "RdYlGn_r",
                                zmin=0, zmax=1, aspect="auto", title=mdl)
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), coloraxis_showscale=False)
                fig.update_xaxes(title="tested on")
                fig.update_yaxes(title="trained on")
                cc[i].plotly_chart(fig, width="stretch")

        st.subheader("3 · Rare attacks: detection rate per original label")
        rows = []
        for _, r in ev[(ev["experiment"] == "in_dataset")].iterrows():
            for lab, d in (r["metrics"].get("per_label_detection") or {}).items():
                rows.append(dict(dataset=DS_NAMES.get(r["train_ds"]), features=r["setting"], model=r["model"],
                                 label=lab, n_test=d["n"], detected=d["detected"]))
        if rows:
            pl = pd.DataFrame(rows)
            ds = st.selectbox("Dataset", sorted(pl["dataset"].unique()))
            feat = st.selectbox("Feature set", sorted(pl["features"].unique()))
            sub = pl[(pl["dataset"] == ds) & (pl["features"] == feat)]
            fig = px.bar(sub.sort_values("n_test"), x="detected", y="label", color="model", barmode="group",
                         orientation="h", hover_data=["n_test"], range_x=[0, 1])
            fig.update_layout(height=max(350, 28 * sub["label"].nunique() * 1.6), margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("For BENIGN/Normal the bar is the false-alarm rate; for attacks it is recall.")

        st.subheader("4 · Robustness against adaptive evasion")
        rb = _metrics_frame(ev, "robustness")
        if not rb.empty:
            fig = px.line(rb.sort_values("eps"), x="eps", y="detection_rate", color="model", markers=True,
                          labels={"eps": "attacker budget ε (padding / slow-down / dummy-packet ratio)",
                                  "detection_rate": "attacks still detected"})
            fig.update_layout(height=360, yaxis_range=[0, 1.02], margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
        at = _metrics_frame(ev, "adv_training")
        if not at.empty:
            st.markdown("Clean-traffic performance before vs after adversarial training (combined test set):")
            st.dataframe(at.drop(columns=["train", "test", "setting"]).round(4), width="stretch", hide_index=True)

        st.subheader("5 · What the models rely on (global SHAP)")
        if not imp.empty:
            which = st.selectbox("Model", imp["model"].unique().tolist() if True else [],
                                 format_func=lambda m: m)
            dsel = imp[imp["model"] == which]["dataset"].unique().tolist()
            d = st.selectbox("Dataset ", dsel, format_func=lambda x: DS_NAMES.get(x, x))
            sub = imp[(imp["model"] == which) & (imp["dataset"] == d)].nlargest(15, "mean_abs_shap")
            sub["feature"] = sub["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
            fig = px.bar(sub.iloc[::-1], x="mean_abs_shap", y="feature", orientation="h")
            fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")

with tab_about:
    st.markdown("""
**Pipeline.** The SafeNet sensor captures packets on a real network interface, groups them into bidirectional
flows (5-tuple, FIN/RST, 15 s idle / 120 s active timeouts) and computes 22 flow features that can be derived
identically from CIC-IDS2017, UNSW-NB15 and live traffic. Each finished flow is scored by:

* **XGBoost (adversarially trained)**: attack probability, alert when ≥ 0.90
* **XGBoost category model**: attack type (DoS, Scan, BruteForce, WebAttack, Botnet, Exploit)
* **Autoencoder** trained on benign flows only: reconstruction error flags unusual traffic, including attacks never seen in training

Every alert stores a **SHAP** explanation of which flow properties pushed the score towards "attack".

**Evaluation.** Section 1 shows standard in-dataset results; Section 2 trains on one dataset and tests on the
other; Section 4 runs a black-box adaptive attacker that pads payloads, slows the attack and injects dummy
packets (only on the attacker's side of the flow) to evade detection, and shows how adversarial training helps.
""")
