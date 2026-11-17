-- SafeNet database schema (PostgreSQL / Supabase).
-- Lives in its own schema so Supabase's auto-generated REST API (which only
-- exposes "public") never exposes captured traffic. RLS is enabled as defence
-- in depth; the sensor/dashboard connect as the database owner.

create schema if not exists safenet;

create table if not exists safenet.model_runs (
    id            bigserial primary key,
    created_at    timestamptz not null default now(),
    version       text not null unique,
    host          text,
    notes         jsonb not null default '{}'::jsonb
);

-- Every offline evaluation number shown on the dashboard.
create table if not exists safenet.eval_results (
    id          bigserial primary key,
    run_id      bigint not null references safenet.model_runs(id) on delete cascade,
    experiment  text not null,        -- in_dataset | cross_dataset | robustness | adv_training | per_class
    model       text not null,
    train_ds    text not null,
    test_ds     text not null,
    setting     text not null default '',
    metrics     jsonb not null,
    created_at  timestamptz not null default now()
);
create index if not exists eval_results_run_idx on safenet.eval_results(run_id, experiment);

create table if not exists safenet.feature_importance (
    id            bigserial primary key,
    run_id        bigint not null references safenet.model_runs(id) on delete cascade,
    model         text not null,
    dataset       text not null,
    feature       text not null,
    mean_abs_shap double precision not null
);

create table if not exists safenet.sensors (
    id             bigserial primary key,
    hostname       text not null,
    interface      text not null,
    model_version  text,
    started_at     timestamptz not null default now(),
    last_seen      timestamptz not null default now(),
    packets_seen   bigint not null default 0,
    flows_scored   bigint not null default 0,
    unique (hostname, interface)
);

create table if not exists safenet.flows (
    id          bigserial primary key,
    sensor_id   bigint references safenet.sensors(id) on delete set null,
    ts_start    timestamptz not null,
    ts_end      timestamptz not null,
    src_ip      text not null,
    src_port    integer,
    dst_ip      text not null,
    dst_port    integer,
    proto       text not null,
    features    jsonb not null,
    pred_label  text not null,
    p_attack    double precision not null,
    ae_score    double precision not null,
    ae_anomaly  boolean not null,
    is_alert    boolean not null default false
);
create index if not exists flows_ts_idx on safenet.flows(ts_end desc);
create index if not exists flows_alert_idx on safenet.flows(is_alert, ts_end desc);

create table if not exists safenet.alerts (
    id          bigserial primary key,
    flow_id     bigint references safenet.flows(id) on delete cascade,
    created_at  timestamptz not null default now(),
    severity    text not null,         -- high | medium | low
    label       text not null,         -- predicted attack category or 'Anomaly'
    p_attack    double precision not null,
    ae_score    double precision not null,
    src_ip      text not null,
    dst_ip      text not null,
    dst_port    integer,
    proto       text not null,
    explanation jsonb not null,        -- SHAP top contributions
    summary     text not null,
    status      text not null default 'open'
);
create index if not exists alerts_ts_idx on safenet.alerts(created_at desc);

alter table safenet.model_runs         enable row level security;
alter table safenet.eval_results       enable row level security;
alter table safenet.feature_importance enable row level security;
alter table safenet.sensors            enable row level security;
alter table safenet.flows              enable row level security;
alter table safenet.alerts             enable row level security;

-- ---------------------------------------------------------------- threat intelligence
alter table safenet.alerts add column if not exists remote_ip text;
alter table safenet.alerts add column if not exists mitre jsonb not null default '[]'::jsonb;
alter table safenet.alerts add column if not exists intel_hits jsonb not null default '[]'::jsonb;

-- status of the downloaded feeds (Feodo Tracker, Spamhaus DROP, CISA KEV)
create table if not exists safenet.intel_feeds (
    key         text primary key,
    name        text not null,
    url         text not null,
    ok          boolean not null,
    entries     integer not null default 0,
    error       text,
    fetched_at  timestamptz,
    checked_at  timestamptz not null default now()
);

-- cached enrichment of public IPs that raised alerts (ipwho.is, Shodan InternetDB, GreyNoise)
create table if not exists safenet.ip_intel (
    ip          text primary key,
    data        jsonb not null,
    fetched_at  timestamptz not null default now()
);

alter table safenet.intel_feeds enable row level security;
alter table safenet.ip_intel    enable row level security;

-- why a high-scoring flow did not raise an alert (outside the models' training scope)
alter table safenet.flows add column if not exists suppressed text;
