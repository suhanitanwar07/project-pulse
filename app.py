
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from datetime import datetime

st.set_page_config(
    page_title="PULSE | Infrastructure Intelligence",
    page_icon="☁️",
    layout="wide"
)

# ---------- DARK DASHBOARD THEME ----------
st.markdown("""
<style>
.stApp {
    background: #0b1020;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background: #10172a;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: #f8fafc !important;
}
div[data-testid="stMetric"] {
    background: #141d32;
    border: 1px solid #27344f;
    padding: 18px;
    border-radius: 12px;
}
div[data-testid="stMetricValue"] {
    color: #38bdf8;
}
.pulse-card {
    background: #141d32;
    border: 1px solid #27344f;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 12px;
}
.small-muted {
    color: #94a3b8;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.markdown("# ☁️ PULSE")
st.sidebar.caption("INFRASTRUCTURE INTELLIGENCE")
st.sidebar.divider()

st.sidebar.markdown("### Environment")
environment = st.sidebar.selectbox(
    "Monitoring environment",
    ["Production", "Staging", "Development"]
)

window = st.sidebar.slider(
    "History window (minutes)",
    min_value=30,
    max_value=120,
    value=60,
    step=15
)

contamination = st.sidebar.slider(
    "Expected anomaly fraction",
    min_value=0.02,
    max_value=0.15,
    value=0.07,
    step=0.01,
    help="Expected fraction of unusual observations. This is a model setting, not a measured incident rate."
)

if st.sidebar.button("↻ Refresh simulation", use_container_width=True):
    st.rerun()

st.sidebar.divider()
st.sidebar.caption("DEMO MODE · SYNTHETIC METRICS")
st.sidebar.caption("No real servers are connected.")

# ---------- GENERATE SIMULATED DATA ----------
@st.cache_data
def generate_data(n=120):
    rng = np.random.default_rng(42)

    data = pd.DataFrame({
        "CPU": np.clip(
            rng.normal(42, 10, n), 5, 95
        ),
        "Memory": np.clip(
            rng.normal(57, 8, n), 15, 95
        ),
        "Latency": np.clip(
            rng.normal(110, 22, n), 30, 400
        ),
        "Network": np.clip(
            rng.normal(48, 12, n), 5, 100
        ),
        "Errors": rng.poisson(2, n)
    })

    # Inject a few unusual events for demonstration.
    for i in [n - 28, n - 27, n - 9, n - 8, n - 7]:
        data.loc[i, "CPU"] = rng.uniform(88, 99)
        data.loc[i, "Latency"] = rng.uniform(260, 390)
        data.loc[i, "Errors"] = rng.integers(12, 25)

    data["Timestamp"] = pd.date_range(
        end=datetime.now(),
        periods=n,
        freq="min"
    )

    return data


data = generate_data()
data = data.tail(window).reset_index(drop=True)

# ---------- MACHINE LEARNING ----------
features = ["CPU", "Memory", "Latency", "Network", "Errors"]

model = IsolationForest(
    n_estimators=150,
    contamination=contamination,
    random_state=42
)

data["Prediction"] = model.fit_predict(data[features])
data["Anomaly"] = data["Prediction"] == -1

# ---------- HEADER ----------
st.markdown(
    "<p style='color:#38bdf8;letter-spacing:3px;'>"
    "CLOUD OPERATIONS / MONITORING"
    "</p>",
    unsafe_allow_html=True
)

st.title("Infrastructure Control Center")
st.markdown(
    "Monitor system behavior, identify anomalies, "
    "and investigate unusual metric patterns."
)

st.caption(
    f"Environment: {environment}  ·  "
    "Data source: Synthetic simulation  ·  "
    "ML engine: Isolation Forest"
)

st.divider()

# ---------- TOP METRICS ----------
latest = data.iloc[-1]
anomaly_count = int(data["Anomaly"].sum())

health = max(
    0,
    min(
        100,
        int(100 - 4 * anomaly_count
            - max(0, latest["CPU"] - 70) * 0.3
            - max(0, latest["Memory"] - 80) * 0.3)
    )
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "CPU Utilization",
    f"{latest['CPU']:.1f}%",
    help="Latest simulated CPU reading"
)
c2.metric(
    "Memory Usage",
    f"{latest['Memory']:.1f}%"
)
c3.metric(
    "Response Latency",
    f"{latest['Latency']:.0f} ms"
)
c4.metric(
    "Health Indicator",
    f"{health}/100"
)

st.write("")

left, right = st.columns([1.65, 1])

# ---------- METRIC CHART ----------
with left:
    st.subheader("Resource utilization")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data["Timestamp"],
        y=data["CPU"],
        name="CPU %",
        mode="lines",
        line=dict(color="#38bdf8", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=data["Timestamp"],
        y=data["Memory"],
        name="Memory %",
        mode="lines",
        line=dict(color="#a78bfa", width=2)
    ))

    fig.update_layout(
        height=350,
        template="plotly_dark",
        paper_bgcolor="#0b1020",
        plot_bgcolor="#141d32",
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.12),
        xaxis_title="Time",
        yaxis_title="Usage (%)"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------- ANOMALY PANEL ----------
with right:
    st.subheader("Anomaly detection")

    if anomaly_count == 0:
        st.success("No anomalies flagged in this window.")
    else:
        st.error(f"{anomaly_count} unusual observations flagged")

    st.markdown(
        "<div class='pulse-card'>"
        "<span class='small-muted'>MODEL</span><br>"
        "<b>Isolation Forest</b><br><br>"
        "<span class='small-muted'>FEATURES ANALYZED</span><br>"
        "<b>5 infrastructure metrics</b><br><br>"
        "<span class='small-muted'>OBSERVATIONS ANALYZED</span><br>"
        f"<b>{len(data)}</b>"
        "</div>",
        unsafe_allow_html=True
    )

    st.caption(
        "An anomaly is an observation the model considers "
        "unusual. It is not automatically a confirmed incident."
    )

# ---------- LATENCY CHART ----------
st.subheader("Response latency & network traffic")

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=data["Timestamp"],
    y=data["Latency"],
    name="Latency (ms)",
    mode="lines",
    line=dict(color="#fb923c", width=2)
))

fig2.add_trace(go.Scatter(
    x=data["Timestamp"],
    y=data["Network"],
    name="Network utilization (%)",
    mode="lines",
    line=dict(color="#34d399", width=2),
    yaxis="y2"
))

fig2.update_layout(
    height=300,
    template="plotly_dark",
    paper_bgcolor="#0b1020",
    plot_bgcolor="#141d32",
    margin=dict(l=10, r=10, t=20, b=10),
    legend=dict(orientation="h", y=1.12),
    yaxis=dict(title="Latency (ms)"),
    yaxis2=dict(
        title="Network (%)",
        overlaying="y",
        side="right"
    )
)

st.plotly_chart(fig2, use_container_width=True)

# ---------- INCIDENT TIMELINE ----------
st.subheader("Incident investigation")

incidents = data[data["Anomaly"]].sort_values(
    "Timestamp", ascending=False
)

if incidents.empty:
    st.info("No unusual observations to investigate in this window.")
else:
    for _, row in incidents.head(10).iterrows():
        reasons = []

        if row["CPU"] > 80:
            reasons.append("high CPU utilization")
        if row["Memory"] > 80:
            reasons.append("high memory usage")
        if row["Latency"] > 220:
            reasons.append("elevated response latency")
        if row["Errors"] > 8:
            reasons.append("increased error count")

        explanation = (
            ", ".join(reasons)
            if reasons
            else "unusual combination of metrics"
        )

        st.markdown(
            f"""
            <div class="pulse-card">
                <span class="small-muted">
                    {row['Timestamp'].strftime('%H:%M:%S')}
                </span>
                <br>
                <b style="color:#fb7185;">UNUSUAL OBSERVATION</b>
                <br>
                {explanation.capitalize()}
                <br><br>
                <span class="small-muted">
                    CPU: {row['CPU']:.1f}% |
                    Memory: {row['Memory']:.1f}% |
                    Latency: {row['Latency']:.0f} ms |
                    Errors: {int(row['Errors'])}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------- RAW DATA ----------
with st.expander("Inspect underlying monitoring data"):
    display_data = data.copy()
    display_data["Status"] = np.where(
        display_data["Anomaly"], "Anomaly", "Normal"
    )
    st.dataframe(
        display_data.drop(columns=["Prediction"]),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download metrics as CSV",
        data=display_data.to_csv(index=False),
        file_name="pulse_metrics.csv",
        mime="text/csv"
    )

st.divider()
st.caption(
    "PULSE prototype · Synthetic monitoring data · "
    "Anomaly detection is experimental"
)

# --- PROJECT PULSE: Real Data Explorer ---
import pandas as pd
import streamlit as st

st.divider()
st.header("Real-World Metrics Explorer")
st.write(
    "Upload a CSV collected by PROJECT PULSE to explore actual "
    "system measurements. Uploaded data is processed by this app."
)

uploaded_metrics = st.file_uploader(
    "Upload your metrics CSV",
    type=["csv"],
    key="real_metrics_upload",
)

if uploaded_metrics is not None:
    try:
        real_df = pd.read_csv(uploaded_metrics)
        st.write("Dataset preview")
        st.dataframe(real_df.head(20), use_container_width=True)

        numeric_columns = real_df.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:
            selected_metric = st.selectbox(
                "Choose a metric to chart",
                numeric_columns,
                key="real_metric_choice",
            )
            st.line_chart(real_df[selected_metric])

        st.caption(
            f"Rows: {len(real_df)} | "
            f"Columns: {len(real_df.columns)}"
        )
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")

st.divider()
st.header("ML Evaluation — AWS CloudWatch Benchmark")
st.write(
    "Offline Isolation Forest experiments using the public "
    "Numenta Anomaly Benchmark (NAB) AWS CPU time series."
)

evaluation_results = pd.DataFrame({
    "Training-score percentile": ["99th", "97th", "95th", "90th", "80th"],
    "Precision": [0.389, 0.433, 0.437, 0.426, 0.374],
    "Recall": [0.017, 0.065, 0.095, 0.229, 0.353],
    "F1 score": [0.033, 0.113, 0.155, 0.298, 0.363],
})

st.dataframe(evaluation_results, hide_index=True, use_container_width=True)
st.warning(
    "Experimental results, not production performance. "
    "The 80th-percentile threshold had the highest F1 among "
    "the five tested thresholds. More validation is needed."
)
