"""Streamlit app visualizing PAC automation bloom-week simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


SECONDS_PER_SIMULATION = 30
DEFAULT_PERIOD_MINUTES = 15
SIMULATION_DAYS = 8


@dataclass(frozen=True)
class ControllerParams:
    pac_base: float = 1.0
    pac_min: float = 1.0
    pac_max: float = 20.0
    max_delta_per_hour: float = 3.0
    k1: float = 100.0
    k2: float = 0.03
    k3: float = 0.15
    chl_enter: float = 8.0
    chl_exit: float = 5.0
    uv_enter: float = 0.06
    uv_exit: float = 0.05
    fdom_enter: float = 80.0
    fdom_exit: float = 60.0


def _easing_profile(length: int, midpoint: float, growth: float = 4.0) -> np.ndarray:
    """Smooth curve that rises and falls around a midpoint."""
    x = np.linspace(0.0, 1.0, length)
    rise = 1.0 / (1.0 + np.exp(-growth * (x - midpoint)))
    fall = 1.0 / (1.0 + np.exp(-growth * ((1 - x) - (1 - midpoint))))
    profile = rise * fall
    return (profile - profile.min()) / (profile.max() - profile.min())


def generate_synthetic_signals(params: ControllerParams) -> pd.DataFrame:
    """Generate 8 days of 15-minute synthetic bloom signals."""
    periods = int((24 * 60 / DEFAULT_PERIOD_MINUTES) * SIMULATION_DAYS)
    index = pd.date_range("2024-06-01", periods=periods, freq=f"{DEFAULT_PERIOD_MINUTES}min")

    bloom_profile = _easing_profile(periods, midpoint=0.45, growth=6.5)
    bloom_profile = bloom_profile ** 1.2

    chl_base = 3.5
    chl_peak = 12.0
    chlorophyll = chl_base + (chl_peak - chl_base) * bloom_profile
    chlorophyll += np.random.normal(0, 0.35, size=periods)

    lag_steps = int(6 * 60 / DEFAULT_PERIOD_MINUTES)
    fdom = 60 + 50 * np.clip(np.roll(bloom_profile, lag_steps), 0, 1)
    fdom += np.random.normal(0, 1.5, size=periods)

    uv_base = 0.04
    uv = uv_base + 0.03 * np.clip(np.roll(bloom_profile, lag_steps // 2), 0, 1)
    uv += np.random.normal(0, 0.002, size=periods)

    turbidity = 1.3 + 0.8 * np.clip(np.roll(bloom_profile, lag_steps // 3), 0, 1)
    turbidity += np.random.normal(0, 0.05, size=periods)

    spike_center = int(0.7 * periods)
    spike_profile = np.exp(-0.5 * ((np.arange(periods) - spike_center) / 3) ** 2)
    turbidity += 0.45 * spike_profile
    uv -= 0.004 * spike_profile

    sdi = 3.0 + 1.2 * np.clip(np.roll(bloom_profile, lag_steps // 4), 0, 1)
    sdi += np.random.normal(0, 0.08, size=periods)

    df = pd.DataFrame(
        {
            "timestamp": index,
            "chlorophyll_ug_per_l": chlorophyll,
            "fdom_rfu": fdom,
            "uv254_1_per_cm": uv,
            "turbidity_ntu": turbidity,
            "sdi_index": sdi,
        }
    )
    return df


def apply_controller_logic(df: pd.DataFrame, params: ControllerParams) -> pd.DataFrame:
    """Apply bloom detection, dosing, and alert logic."""
    df = df.copy()

    bloom_on = np.zeros(len(df), dtype=bool)
    pac = np.empty(len(df))
    pac[:] = params.pac_base
    alert = np.zeros(len(df), dtype=int)
    fault = np.zeros(len(df), dtype=int)

    pac_prev = params.pac_base
    max_delta_per_step = params.max_delta_per_hour * (DEFAULT_PERIOD_MINUTES / 60)

    for i, row in df.iterrows():
        if bloom_on[i - 1] if i > 0 else False:
            prev_mode = True
        else:
            prev_mode = False

        enter_condition = (
            (row["chlorophyll_ug_per_l"] >= params.chl_enter)
            or (
                row["uv254_1_per_cm"] >= params.uv_enter
                and row["fdom_rfu"] >= params.fdom_enter
            )
        )
        exit_condition = (
            row["chlorophyll_ug_per_l"] <= params.chl_exit
            and row["uv254_1_per_cm"] <= params.uv_exit
            and row["fdom_rfu"] <= params.fdom_exit
        )

        if prev_mode and not exit_condition:
            bloom_on[i] = True
        elif not prev_mode and enter_condition:
            bloom_on[i] = True
        else:
            bloom_on[i] = False

        if bloom_on[i]:
            pac_target = (
                params.pac_base
                + params.k1 * row["uv254_1_per_cm"]
                + params.k2 * row["fdom_rfu"]
                + params.k3 * max(0.0, row["chlorophyll_ug_per_l"] - params.chl_exit)
            )
        else:
            pac_target = params.pac_base

        pac_clamped = float(np.clip(pac_target, params.pac_min, params.pac_max))
        pac_rate = pac_prev

        delta = pac_clamped - pac_prev
        if delta > max_delta_per_step:
            pac_rate = pac_prev + max_delta_per_step
        elif delta < -max_delta_per_step:
            pac_rate = pac_prev - max_delta_per_step
        else:
            pac_rate = pac_clamped

        if i > 0:
            turbidity_trend = row["turbidity_ntu"] - df.loc[i - 1, "turbidity_ntu"]
            uv_trend = row["uv254_1_per_cm"] - df.loc[i - 1, "uv254_1_per_cm"]
        else:
            turbidity_trend = 0.0
            uv_trend = 0.0

        if turbidity_trend > 0.25 and uv_trend < -0.002:
            pac_rate = pac_prev
            alert[i] = 1
        else:
            alert[i] = 0

        pac[i] = pac_rate
        pac_prev = pac_rate

    df["mode_bloom"] = bloom_on.astype(int)
    df["pac_setpoint_mg_per_l"] = pac
    df["alert_flag"] = alert
    df["sensor_fault"] = fault

    uv254_ro = df["uv254_1_per_cm"] * (1 - 0.02 * df["pac_setpoint_mg_per_l"]) + np.random.normal(0, 0.002, len(df))
    df["ro_influent_uv254"] = np.clip(uv254_ro, 0, None)

    return df


def generate_simulation(params: ControllerParams) -> pd.DataFrame:
    sensors = generate_synthetic_signals(params)
    return apply_controller_logic(sensors, params)


def _frame_caption(frame_idx: int, total_frames: int) -> str:
    pct = frame_idx / max(total_frames - 1, 1)
    if pct <= 0.15:
        return "Sensors online. Baseline stable."
    if pct <= 0.30:
        return "Day 2: Algae signals rising."
    if pct <= 0.45:
        return "Bloom detected. Controller activates."
    if pct <= 0.60:
        return "Dosing adjusts with rate limits."
    if pct <= 0.75:
        return "Guardrails active: turbidity spike detected."
    if pct <= 0.90:
        return "Bloom subsides. Controller returns to baseline."
    return "Outcome: RO feedwater cleaner, system stable."


def _box_annotations(state: Dict[str, bool], caption: str) -> List[dict]:
    base_color = "#f2f0ec"
    active_colors = {
        "sensors": "#ffd7ba",
        "logic": "#f8c6ff",
        "dose": "#ffb3c1",
        "safety": "#ffe29a",
    }
    labels = [
        ("SENSORS", "sensors"),
        ("LOGIC", "logic"),
        ("DOSE", "dose"),
        ("SAFETY", "safety"),
    ]
    annotations = []
    for idx, (label, key) in enumerate(labels):
        x = 0.13 + idx * 0.24
        annotations.append(
            dict(
                x=x,
                y=1.18,
                xref="paper",
                yref="paper",
                text=label,
                showarrow=False,
                font=dict(size=14, color="#333"),
                align="center",
                bordercolor="#b8b3a7",
                borderwidth=1,
                borderpad=6,
                bgcolor=active_colors[key] if state[key] else base_color,
                opacity=0.95,
            )
        )
        if idx < len(labels) - 1:
            annotations.append(
                dict(
                    x=x + 0.12,
                    y=1.18,
                    xref="paper",
                    yref="paper",
                    text="→",
                    showarrow=False,
                    font=dict(size=20, color="#888"),
                )
            )
    annotations.append(
        dict(
            x=0.5,
            y=1.32,
            xref="paper",
            yref="paper",
            text=caption,
            font=dict(size=16, color="#2f2c28"),
            showarrow=False,
            align="center",
            bgcolor="rgba(255,255,255,0.7)",
            borderpad=10,
        )
    )
    return annotations


def _build_animation(df: pd.DataFrame, params: ControllerParams) -> go.Figure:
    total_frames = 120
    frame_indices = np.unique(np.linspace(0, len(df) - 1, total_frames, dtype=int))
    total_frames = len(frame_indices)

    fig = make_subplots(rows=2, cols=1, shared_x=True, vertical_spacing=0.1)

    colors = {
        "chlorophyll_ug_per_l": "#ff9966",
        "fdom_rfu": "#d4a5ff",
        "uv254_1_per_cm": "#3bc5ce",
        "turbidity_ntu": "#6d8c80",
        "pac_setpoint_mg_per_l": "#ff6f6f",
        "pac_base": "#f5a3a3",
        "pac_max": "#f28484",
        "ro_influent_uv254": "#6ec5ff",
    }

    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="rgba(255,176,148,0.25)", width=0),
            fill="tozeroy",
            fillcolor="rgba(255,176,148,0.25)",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    for column in ["chlorophyll_ug_per_l", "fdom_rfu", "uv254_1_per_cm", "turbidity_ntu"]:
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color=colors[column], width=3, shape="spline"),
                name=column,
            ),
            row=1,
            col=1,
        )

    for name, dash, width in [
        ("pac_setpoint_mg_per_l", None, 4),
        ("pac_base", "dot", 2),
        ("pac_max", "dash", 2),
        ("ro_influent_uv254", None, 2),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color=colors.get(name, "#999"), width=width, dash=dash, shape="spline"),
                name=name,
            ),
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#f0a500"),
            name="alert",
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Timestamp", row=2, col=1)
    fig.update_yaxes(title_text="Sensor Signals", row=1, col=1)
    fig.update_yaxes(title_text="PAC & RO UV254", row=2, col=1)

    frames = []
    pac_base_series = pd.Series(params.pac_base, index=df.index)
    pac_max_series = pd.Series(params.pac_max, index=df.index)

    for pos, idx in enumerate(frame_indices):
        subset = df.iloc[: idx + 1]
        current = df.iloc[idx]
        caption = _frame_caption(pos, total_frames)
        state = {
            "sensors": bool(
                current["chlorophyll_ug_per_l"] > params.chl_exit + 0.3
                or current["uv254_1_per_cm"] > params.uv_exit + 0.002
                or current["fdom_rfu"] > params.fdom_exit + 3
            ),
            "logic": bool(current["mode_bloom"]),
            "dose": current["pac_setpoint_mg_per_l"] > params.pac_base + 0.1,
            "safety": bool(current["alert_flag"] or current["sensor_fault"]),
        }
        shading = np.where(subset["mode_bloom"] > 0, subset["chlorophyll_ug_per_l"] * 0.9, 0)

        frame = go.Frame(
            data=[
                go.Scatter(
                    x=subset["timestamp"],
                    y=shading,
                ),
                go.Scatter(
                    x=subset["timestamp"],
                    y=subset["chlorophyll_ug_per_l"],
                ),
                go.Scatter(
                    x=subset["timestamp"],
                    y=subset["fdom_rfu"],
                ),
                go.Scatter(
                    x=subset["timestamp"],
                    y=subset["uv254_1_per_cm"],
                ),
                go.Scatter(
                    x=subset["timestamp"],
                    y=subset["turbidity_ntu"],
                ),
                go.Scatter(
                    x=subset["timestamp"],
                    y=subset["pac_setpoint_mg_per_l"],
                ),
                go.Scatter(
                    x=subset["timestamp"],
                    y=pac_base_series.iloc[: idx + 1],
                ),
                go.Scatter(
                    x=subset["timestamp"],
                    y=pac_max_series.iloc[: idx + 1],
                ),
                go.Scatter(
                    x=subset["timestamp"],
                    y=subset["ro_influent_uv254"],
                ),
                go.Scatter(
                    x=subset.loc[subset["alert_flag"] == 1, "timestamp"],
                    y=subset.loc[subset["alert_flag"] == 1, "pac_setpoint_mg_per_l"],
                ),
            ],
            name=str(idx),
            layout=dict(annotations=_box_annotations(state, caption)),
        )
        frames.append(frame)

    fig.frames = frames

    if frames:
        fig.update(data=frames[0].data)
        fig.update_layout(annotations=frames[0].layout["annotations"])

    fig.update_layout(
        template="plotly_white",
        height=720,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0),
        margin=dict(t=160, b=60, l=70, r=40),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "▶ Run",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": (SECONDS_PER_SIMULATION / max(total_frames, 1)) * 1000, "redraw": False},
                                "fromcurrent": True,
                            },
                        ],
                    }
                ],
                "x": 0.05,
                "y": 1.05,
            }
        ],
        sliders=[
            {
                "active": 0,
                "pad": {"t": 60},
                "steps": [
                    {
                        "label": subset_idx.strftime("%b %d %H:%M"),
                        "method": "animate",
                        "args": [[str(idx)], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
                    }
                    for subset_idx, idx in zip(df.loc[frame_indices, "timestamp"], frame_indices)
                ],
            }
        ],
        paper_bgcolor="#f7f3ef",
        plot_bgcolor="#fffdf8",
    )

    fig.add_hline(y=params.chl_enter, line_dash="dash", line_color="#ffb995", row=1, col=1, annotation_text="Chl enter")
    fig.add_hline(y=params.chl_exit, line_dash="dot", line_color="#ffb995", row=1, col=1, annotation_text="Chl exit")
    fig.add_hline(y=params.uv_enter, line_dash="dash", line_color="#3bc5ce", row=1, col=1, annotation_text="UV enter")
    fig.add_hline(y=params.uv_exit, line_dash="dot", line_color="#3bc5ce", row=1, col=1, annotation_text="UV exit")
    fig.add_hline(y=params.fdom_enter, line_dash="dash", line_color="#d4a5ff", row=1, col=1, annotation_text="FDOM enter")
    fig.add_hline(y=params.fdom_exit, line_dash="dot", line_color="#d4a5ff", row=1, col=1, annotation_text="FDOM exit")

    return fig


def _sidebar_params() -> ControllerParams:
    st.sidebar.header("Control Panel")
    st.sidebar.caption("Adjust values, then press **Run Simulation** to replay.")

    def slider(label: str, value: float, min_value: float, max_value: float, step: float) -> float:
        return st.sidebar.slider(label, min_value=min_value, max_value=max_value, value=value, step=step)

    pac_max = slider("PAC max", 20.0, 5.0, 30.0, 0.5)
    k1 = slider("k1 (UV weight)", 100.0, 50.0, 180.0, 5.0)
    k2 = slider("k2 (FDOM weight)", 0.03, 0.0, 0.08, 0.005)
    k3 = slider("k3 (Chl bias)", 0.15, 0.0, 0.5, 0.01)
    max_delta = slider("Max delta per hr", 3.0, 0.5, 5.0, 0.1)

    st.sidebar.divider()
    chl_enter = slider("Chl enter", 8.0, 4.0, 12.0, 0.5)
    chl_exit = slider("Chl exit", 5.0, 2.0, 8.0, 0.5)
    uv_enter = slider("UV enter", 0.06, 0.03, 0.09, 0.002)
    uv_exit = slider("UV exit", 0.05, 0.02, 0.08, 0.002)
    fdom_enter = slider("FDOM enter", 80.0, 40.0, 120.0, 2.0)
    fdom_exit = slider("FDOM exit", 60.0, 30.0, 100.0, 2.0)

    params = ControllerParams(
        pac_max=pac_max,
        k1=k1,
        k2=k2,
        k3=k3,
        max_delta_per_hour=max_delta,
        chl_enter=chl_enter,
        chl_exit=chl_exit,
        uv_enter=uv_enter,
        uv_exit=uv_exit,
        fdom_enter=fdom_enter,
        fdom_exit=fdom_exit,
    )

    if "params" not in st.session_state:
        st.session_state["params"] = params

    if st.sidebar.button("Run Simulation", type="primary"):
        st.session_state["params"] = params
        st.session_state["run_id"] = st.session_state.get("run_id", 0) + 1

    return st.session_state.get("params", params)


def main() -> None:
    st.set_page_config(page_title="PAC Automation Bloom Week", layout="wide")
    st.title("PAC Automation: Bloom Week Simulation")
    st.subheader("Explainable dosing controller responding to an algal bloom event")

    params = _sidebar_params()
    run_id = st.session_state.get("run_id", 0)

    with st.spinner("Generating simulation..."):
        df = generate_simulation(params)

    st.caption(
        "This narrative animation compresses an eight-day bloom scenario into 30 seconds, highlighting detection, dosing, and safety guardrails."
    )
    fig = _build_animation(df, params)
    st.plotly_chart(fig, use_container_width=True, key=f"figure-{run_id}")

    st.markdown("### Controller Timeline Snapshot")
    st.dataframe(
        df[[
            "timestamp",
            "chlorophyll_ug_per_l",
            "fdom_rfu",
            "uv254_1_per_cm",
            "turbidity_ntu",
            "pac_setpoint_mg_per_l",
            "mode_bloom",
            "alert_flag",
        ]]
        .tail(20)
        .reset_index(drop=True)
    )

    st.markdown(
        "*Safety guardrails hold the PAC dose steady when turbidity spikes while UV254 drops, demonstrating transparent operational logic.*"
    )


if __name__ == "__main__":
    main()
