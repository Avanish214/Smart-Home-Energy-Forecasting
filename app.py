from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from energy_model import (
    FORECAST_STEPS,
    LOOKBACK_STEPS,
    download_uci_dataset,
    forecast_lstm,
    load_uci_dataset,
    make_demo_dataset,
    train_lstm,
)

st.set_page_config(page_title="Wattwise | Home energy", page_icon="⌂", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#182522; --muted:#73817b; --paper:#f5f6f1; --line:#e0e5dc; --green:#287a56; --lime:#c8ef72; --red:#b44d3e; }
    html, body, [class*="css"] { font-family:'Manrope', sans-serif; color:var(--ink); }
    .stApp { background:var(--paper); }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:#202d29; }
    [data-testid="stSidebar"] * { color:#edf2eb; }
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] { background:#34443e; border:1px solid #53625a; }
    .block-container { max-width:1440px; padding-top:2.2rem; padding-bottom:3rem; }
    .topline { display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid var(--line); padding-bottom:18px; margin-bottom:26px; }
    .brand { font-size:15px; font-weight:800; letter-spacing:.02em; }
    .brand-mark { display:inline-grid; place-items:center; width:30px; height:30px; margin-right:9px; background:var(--lime); color:var(--ink); border-radius:9px; }
    .eyebrow { color:var(--green); font:500 11px 'DM Mono', monospace; text-transform:uppercase; letter-spacing:.13em; }
    .title { font-size:30px; line-height:1.15; font-weight:700; margin:5px 0 4px; }
    .subtle { color:var(--muted); font-size:13px; }
    .status { display:inline-block; color:#52635a; border:1px solid var(--line); border-radius:20px; padding:7px 11px; font-size:11px; background:#fff; }
    .metric { background:#fff; border:1px solid var(--line); border-radius:8px; padding:17px 19px 15px; min-height:112px; }
    .metric-label { color:var(--muted); text-transform:uppercase; font:500 10px 'DM Mono', monospace; letter-spacing:.09em; }
    .metric-value { font-size:27px; font-weight:700; line-height:1.35; margin-top:6px; }
    .metric-foot { color:var(--muted); font-size:11px; }
    .section-head { display:flex; align-items:baseline; justify-content:space-between; margin:24px 0 10px; }
    .section-title { font-size:15px; font-weight:700; }
    .chart-panel { background:#fff; border:1px solid var(--line); border-radius:8px; padding:12px 14px 0; }
    .notice { background:#fff6e9; color:#76501a; border-left:3px solid #e8b453; padding:12px 15px; border-radius:3px; font-size:13px; margin:14px 0; }
    .notice-alert { background:#fff0ed; color:#8c392d; border-left-color:#d97060; }
    .recommendation { border-top:1px solid var(--line); padding:13px 0; font-size:13px; line-height:1.55; }
    .recommendation strong { color:var(--ink); }
    .small-note { color:var(--muted); font-size:11px; }
    div[data-testid="stButton"] button[kind="primary"] { background:var(--lime); color:var(--ink); border:0; font-weight:700; }
    div[data-testid="stButton"] button[kind="primary"]:hover { background:#b9e65a; color:var(--ink); }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    if "readings" not in st.session_state:
        st.session_state.readings = make_demo_dataset()
        st.session_state.is_demo = True
    if "model" not in st.session_state:
        st.session_state.model = None


def show_metric(label: str, value: str, footnote: str) -> None:
    st.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-foot">{footnote}</div></div>',
        unsafe_allow_html=True,
    )


def build_chart(history: pd.Series, prediction: pd.Series | None) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            name="Measured",
            line={"color": "#287a56", "width": 2},
            hovertemplate="%{x|%b %d, %H:%M}<br>%{y:.2f} kW<extra>Measured</extra>",
        )
    )
    if prediction is not None:
        edge_x = [history.index[-1], prediction.index[0]]
        edge_y = [history.iloc[-1], prediction.iloc[0]]
        figure.add_trace(go.Scatter(x=edge_x, y=edge_y, showlegend=False, line={"color": "#b6d65c", "width": 2, "dash": "dot"}))
        figure.add_trace(
            go.Scatter(
                x=prediction.index,
                y=prediction.values,
                name="LSTM forecast" if st.session_state.model is not None else "Illustrative",
                line={"color": "#a7c747" if not st.session_state.is_demo else "#9ca9a0", "width": 2.5, "dash": "dot" if st.session_state.is_demo else "solid"},
                fill="tozeroy",
                fillcolor="rgba(200,239,114,0.14)",
                hovertemplate="%{x|%b %d, %H:%M}<br>%{y:.2f} kW<extra>Forecast</extra>",
            )
        )
    figure.update_layout(
        height=340,
        margin={"l": 4, "r": 12, "t": 10, "b": 5},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "DM Mono, monospace", "size": 10, "color": "#73817b"},
        legend={"orientation": "h", "y": 1.12, "x": 0.01},
        hovermode="x unified",
        xaxis={"showgrid": False, "zeroline": False},
        yaxis={"title": "kW", "gridcolor": "#edf0ea", "rangemode": "tozero", "zeroline": False},
    )
    return figure


initialize_state()
readings: pd.Series = st.session_state.readings

with st.sidebar:
    st.markdown("<div class='brand'><span class='brand-mark'>⌂</span>WATTWISE</div>", unsafe_allow_html=True)
    st.markdown("<p class='eyebrow' style='margin-top:32px;color:#c8ef72'>DATA SOURCE</p>", unsafe_allow_html=True)
    st.caption("UCI Individual Household Electric Power Consumption")
    st.caption("1-minute readings · 2006–2010")
    if st.button("Load UCI dataset", width="stretch"):
        try:
            with st.spinner("Downloading and processing the UCI archive…"):
                archive_path = download_uci_dataset(Path("data/household_power_consumption.zip"))
                st.session_state.readings = load_uci_dataset(archive_path)
                st.session_state.is_demo = False
                st.session_state.model = None
            st.rerun()
        except Exception as error:
            st.error(f"Could not load the UCI dataset: {error}")
    if st.button("Use sample data", width="stretch"):
        st.session_state.readings = make_demo_dataset()
        st.session_state.is_demo = True
        st.session_state.model = None
        st.rerun()

    st.markdown("<p class='eyebrow' style='margin-top:28px;color:#c8ef72'>FORECAST MODEL</p>", unsafe_allow_html=True)
    epochs = st.slider("Training epochs", min_value=2, max_value=20, value=8)
    if st.button("Train LSTM", type="primary", width="stretch"):
        try:
            with st.spinner("Training multi-step LSTM…"):
                fitted = train_lstm(readings, epochs=epochs)
                st.session_state.model = fitted[:3]
                st.session_state.model_mae = fitted[3][-1]
            st.success(f"Training complete · validation MAE {st.session_state.model_mae:.2f} kW")
        except ModuleNotFoundError:
            st.error("TensorFlow is missing. Install project dependencies with `pip install -r requirements.txt`.")
        except Exception as error:
            st.error(f"Training could not complete: {error}")
    st.markdown("<p class='small-note' style='margin-top:14px'>Input window: previous 24 hours<br>Output horizon: next 4 hours</p>", unsafe_allow_html=True)


current_kw = float(readings.iloc[-1])
today = readings[readings.index.date == readings.index[-1].date()]
daily_kwh = float(today.sum() * 0.25)
peak_kw = float(today.max())
model_info = st.session_state.model
forecast: pd.Series | None = None
if model_info is not None:
    forecast = forecast_lstm(model_info[0], readings, model_info[1], model_info[2])
elif st.session_state.is_demo:
    demo_values = []
    last_time = readings.index[-1]
    for step in range(1, FORECAST_STEPS + 1):
        time = last_time + pd.Timedelta(15 * step, unit="min")
        hour = time.hour + time.minute / 60
        baseline = 0.48 + 0.34 * np.exp(-((hour - 7.5) / 2.1) ** 2) + 1.05 * np.exp(-((hour - 19.5) / 2.8) ** 2)
        demo_values.append(max(0.2, 0.72 * current_kw + 0.28 * baseline))
    forecast = pd.Series(demo_values, index=pd.date_range(readings.index[-1] + pd.Timedelta(15, unit="min"), periods=FORECAST_STEPS, freq="15min"))

st.markdown(
    "<div class='topline'><div class='brand'><span class='brand-mark'>⌂</span>WATTWISE</div>"
    f"<span class='status'>{'SYNTHETIC DEMO DATA' if st.session_state.is_demo else 'UCI HOUSEHOLD DATA'}</span></div>",
    unsafe_allow_html=True,
)
st.markdown("<div class='eyebrow'>HOME ENERGY / OVERVIEW</div>", unsafe_allow_html=True)
st.markdown("<div class='title'>Your home, at a glance.</div>", unsafe_allow_html=True)
st.markdown("<div class='subtle'>Household active power · readings aggregated to 15-minute intervals</div>", unsafe_allow_html=True)

metric_columns = st.columns(4, gap="medium")
with metric_columns[0]:
    show_metric("Current draw", f"{current_kw:.2f} <span style='font-size:14px;font-weight:500'>kW</span>", "Most recent 15-minute average")
with metric_columns[1]:
    next_value = float(forecast.iloc[0]) if forecast is not None else current_kw
    show_metric("Next 15 minutes", f"{next_value:.2f} <span style='font-size:14px;font-weight:500'>kW</span>", "LSTM forecast" if model_info else "Illustrative estimate" if st.session_state.is_demo else "Train the LSTM to forecast")
with metric_columns[2]:
    show_metric("Today so far", f"{daily_kwh:.1f} <span style='font-size:14px;font-weight:500'>kWh</span>", "Estimated from 15-minute mean power")
with metric_columns[3]:
    show_metric("Today's peak", f"{peak_kw:.2f} <span style='font-size:14px;font-weight:500'>kW</span>", "Highest 15-minute average")

if st.session_state.is_demo and model_info is None:
    st.markdown("<div class='notice'>Sample data and its outlook are synthetic illustrations, not household readings or a trained LSTM forecast. Load the UCI dataset and train the model for real predictions.</div>", unsafe_allow_html=True)
elif model_info is None:
    st.markdown("<div class='notice'>UCI readings loaded. Train the LSTM from the sidebar to generate a four-hour forecast.</div>", unsafe_allow_html=True)

if current_kw >= 3.5 or (forecast is not None and float(forecast.max()) >= 3.5):
    st.markdown("<div class='notice notice-alert'><strong>High consumption detected.</strong> Power is above 3.5 kW now or is forecast to cross that level. Check for several high-draw appliances running together.</div>", unsafe_allow_html=True)

left, right = st.columns([1.8, 1], gap="large")
with left:
    st.markdown("<div class='section-head'><span class='section-title'>Consumption &amp; outlook</span><span class='small-note'>Previous 48 hours · next 4 hours</span></div>", unsafe_allow_html=True)
    history = readings.tail(LOOKBACK_STEPS * 2)
    st.markdown("<div class='chart-panel'>", unsafe_allow_html=True)
    st.plotly_chart(build_chart(history, forecast), width="stretch", config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)
    if model_info:
        st.markdown(f"<div class='small-note'>Held-out validation mean absolute error: {st.session_state.model_mae:.3f} kW · horizon: 4 hours</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='section-head'><span class='section-title'>Next hour</span><span class='small-note'>15-minute steps</span></div>", unsafe_allow_html=True)
    if forecast is None:
        st.markdown("<div class='recommendation'>Forecast will appear after you train the model.</div>", unsafe_allow_html=True)
    else:
        for offset, value in zip((15, 30, 45, 60), forecast.iloc[:4]):
            label = "Illustrative" if st.session_state.is_demo and model_info is None else "Predicted"
            st.markdown(f"<div class='recommendation'><span class='small-note'>+{offset} MIN · {label.upper()}</span><br><strong>{value:.2f} kW</strong></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-head'><span class='section-title'>Small ways to save</span></div>", unsafe_allow_html=True)
    recommendations: list[str] = []
    if current_kw >= 3.5 or (forecast is not None and float(forecast.max()) >= 3.5):
        recommendations.append("<strong>Stagger heavy appliances.</strong> Running heating, laundry, or cooking loads together can create a short-lived peak.")
    if readings.index[-1].hour < 6 and current_kw > 1.0:
        recommendations.append("<strong>Check overnight standby.</strong> A sustained overnight load may come from equipment that can be scheduled or switched off.")
    if readings.index[-1].hour < 17:
        recommendations.append("<strong>Shift flexible use.</strong> If your tariff varies by time, consider scheduling laundry or dishwashing for a lower-rate period.")
    if not recommendations:
        recommendations.append("<strong>Keep an eye on the evening peak.</strong> Compare tonight's usage with the daily profile and stagger high-draw appliances where practical.")
    for recommendation in recommendations[:2]:
        st.markdown(f"<div class='recommendation'>{recommendation}</div>", unsafe_allow_html=True)
    st.markdown("<div class='small-note'>Suggestions are general guidance, not a tariff-specific savings estimate.</div>", unsafe_allow_html=True)

st.markdown("<div style='border-top:1px solid #e0e5dc;margin-top:32px;padding-top:14px' class='small-note'>Source: UCI Machine Learning Repository · Individual Household Electric Power Consumption. Model uses global active power only.</div>", unsafe_allow_html=True)
