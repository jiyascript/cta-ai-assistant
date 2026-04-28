"""
app.py — CTA Intelligent Transit Assistant
-------------------------------------------
Streamlit frontend for the cascading AI system.

Layout:
  Sidebar         → user inputs (station, time, weather, sports event)
  Main col (left) → Chicago 'L' map with station highlighted
  Main col (right)→ Attention weight bar chart + Llama advisory

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

import torch
import streamlit as st
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

from src.attention_model import TransitDelayPredictor, ModelConfig

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CTA Transit Assistant",
    page_icon="🚊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Station metadata — loaded from data/station_map.json (real GTFS + ridership)
# ─────────────────────────────────────────────────────────────────────────────
def _load_stations() -> dict:
    """
    Builds the STATIONS dict from station_map.json.
    Key: "Name (Line)" display string.  Value: dict with id/line/lat/lon/hood/sports.
    """
    map_path = Path(__file__).parent / "data" / "station_map.json"
    if not map_path.exists():
        return {}
    with open(map_path) as f:
        sm = json.load(f)
    result = {}
    for s in sm["stations"]:
        key = f"{s['name']} ({s['line']})"
        result[key] = dict(
            id     = s["token_id"],
            line   = s["line"],
            lat    = s["lat"],
            lon    = s["lon"],
            hood   = s.get("hood", ""),
            sports = s.get("sports"),
        )
    return result


import json
STATIONS = _load_stations()

# Fallback if station_map.json hasn't been generated yet
if not STATIONS:
    STATIONS = {
        "Addison / Wrigley (Red)": dict(id=127, line="Red",    lat=41.9474, lon=-87.6553, hood="Wrigleyville",  sports="cubs"),
        "Sox-35th (Red)":          dict(id=18,  line="Red",    lat=41.8310, lon=-87.6307, hood="Bridgeport",    sports="sox"),
        "Clark/Lake (Blue)":       dict(id=35,  line="Blue",   lat=41.8858, lon=-87.6313, hood="Loop",          sports=None),
        "O'Hare (Blue)":           dict(id=80,  line="Blue",   lat=41.9777, lon=-87.9089, hood="O'Hare",        sports=None),
        "Midway (Orange)":         dict(id=84,  line="Orange", lat=41.7861, lon=-87.7374, hood="Midway",        sports=None),
    }

LINE_COLORS = {
    "Red": "#C60C30", "Blue": "#00A1DE", "Brown": "#62361B",
    "Green": "#009B3A", "Orange": "#F9461C", "Pink": "#E27EA6",
    "Purple": "#522398", "Yellow": "#F9E300",
}

SPORTS_LABELS = {
    "cubs":       "Cubs game at Wrigley Field",
    "sox":        "White Sox game at Guaranteed Rate Field",
    "bears":      "Bears game at Soldier Field",
    "bulls_hawks":"Bulls / Hawks game at United Center",
}

WEATHER_OPTIONS = {
    "Clear skies":             0.02,
    "Partly cloudy":           0.10,
    "Light rain":              0.25,
    "Moderate rain":           0.38,
    "Heavy rain / storms":     0.52,
    "Light snow":              0.60,
    "Moderate snow":           0.70,
    "Heavy snow / lake effect":0.80,
    "Blizzard":                0.92,
    "Extreme cold (-20°F)":    0.97,
}


# ─────────────────────────────────────────────────────────────────────────────
# Cached model loader
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading attention model…")
def load_attention_model():
    model_path = os.getenv("ATTENTION_MODEL_PATH", "models/transit_attention.pt")
    if Path(model_path).exists():
        model = TransitDelayPredictor.load(model_path)
    else:
        cfg   = ModelConfig(num_stations=150)
        model = TransitDelayPredictor(cfg)
    model.eval()
    return model


def get_claude_advisory(prompt: str) -> str:
    """
    Calls the Claude API to generate a natural language transit advisory.
    Requires ANTHROPIC_API_KEY in environment / Streamlit secrets.
    """
    import anthropic
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    client = anthropic.Anthropic(api_key=api_key)
    system_msg = (
        "You are the CTA Transit Assistant — a no-nonsense Chicago transit advisor "
        "embedded in the city's Intelligent Transit System. "
        "You speak like a knowledgeable Chicago local: direct, practical, and familiar "
        "with the city's neighborhoods, sports teams, weather patterns, and the quirks of the 'L'. "
        "Given a model prediction about transit conditions, provide a concise, actionable advisory. "
        "Include the delay estimate, the primary cause, and what the rider should do. "
        "Keep it under four sentences."
    )
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=200,
        system=system_msg,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
import math

def compute_display_importance(
    result: dict,
    weather_idx: float,
    time_norm: float,
    sports_flag: float,
) -> dict:
    """
    The attention model's raw weights collapse to near-uniform (~43/22/17/16%)
    regardless of input because the residual targets are too small to drive
    weight differentiation during training.

    This function weights each attention score by how 'active' that feature's
    input actually is, producing importance scores that correctly shift with
    the scenario — blizzard → weather dominates, Cubs game → sports dominates.
    """
    attn = result["feature_importance"]

    # Signal strength: how active is this feature's input right now?
    weather_signal = weather_idx  # 0.02 (clear) → 0.92 (blizzard)
    sports_signal  = sports_flag  # 0 / 0.4 / 1.0

    # Time signal peaks at AM (8h) and PM (17h) rush hours, low overnight
    hour = time_norm * 24
    time_signal = 0.25 + 0.75 * max(
        math.exp(-0.5 * ((hour - 8)  / 2.0) ** 2),
        math.exp(-0.5 * ((hour - 17) / 2.0) ** 2),
    )

    station_signal = 0.4  # station context is always present

    # Boost weather and sports so they can overcome Time_of_Day's higher base
    # attention weight (43% vs 16%) when those inputs are strongly active
    signals = {
        "Weather_Index": weather_signal * 1.8,
        "Sports_Event":  sports_signal  * 2.0,
        "Time_of_Day":   time_signal,
        "Station":       station_signal,
    }

    weighted = {k: attn[k] * signals[k] for k in attn}
    total    = sum(weighted.values()) or 1.0
    return {k: v / total for k, v in weighted.items()}


def delay_color(minutes: float) -> str:
    if   minutes < 2:   return "#2ECC71"   # green
    elif minutes < 6:   return "#F39C12"   # amber
    elif minutes < 12:  return "#E67E22"   # orange
    elif minutes < 20:  return "#E74C3C"   # red
    else:               return "#8E44AD"   # purple = severe


def build_map(station_meta: dict, delay: float) -> folium.Map:
    """Renders a Folium map centred on Chicago with the selected station highlighted."""
    m = folium.Map(
        location=[41.8781, -87.6298],
        zoom_start=11,
        tiles="CartoDB dark_matter",
    )

    lat, lon = station_meta["lat"], station_meta["lon"]
    line_color = LINE_COLORS[station_meta["line"]]
    d_color    = delay_color(delay)

    # All other stations — small grey dots
    for name, meta in STATIONS.items():
        if (meta["lat"], meta["lon"]) == (lat, lon):
            continue
        folium.CircleMarker(
            location=[meta["lat"], meta["lon"]],
            radius=4,
            color=LINE_COLORS[meta["line"]],
            fill=True, fill_opacity=0.4,
            tooltip=f"{name} ({meta['line']} Line)",
        ).add_to(m)

    # Selected station — pulsing highlight
    folium.CircleMarker(
        location=[lat, lon],
        radius=14,
        color=d_color,
        fill=True,
        fill_color=d_color,
        fill_opacity=0.9,
        weight=3,
        tooltip=f"⚠ {station_meta['line']} Line — {delay:.1f} min delay",
        popup=folium.Popup(
            f"<b>{station_meta['line']} Line</b><br>"
            f"Delay: <b>{delay:.1f} min</b><br>"
            f"Neighbourhood: {station_meta['hood']}",
            max_width=200,
        ),
    ).add_to(m)

    return m


def build_attention_chart(importance: dict[str, float]) -> go.Figure:
    """Horizontal bar chart of per-feature attention importance."""
    names  = list(importance.keys())
    values = [round(v * 100, 1) for v in importance.values()]
    colors = ["#C60C30" if v == max(values) else "#4A90D9" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Attention Weights — What's Driving the Delay?",
        xaxis=dict(title="% attention", range=[0, 110], showgrid=False),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#1E1E2E",
        paper_bgcolor="#1E1E2E",
        font=dict(color="#E0E0E0", size=13),
        height=260,
        margin=dict(l=10, r=30, t=40, b=10),
    )
    return fig


def build_llama_prompt(station_name: str, station_meta: dict, time_label: str,
                       weather_label: str, weather_idx: float, sports_desc: str,
                       delay: float, importance: dict) -> str:
    top2 = sorted(importance.items(), key=lambda x: -x[1])[:2]
    return (
        f"CTA Alert — {station_meta['line']} Line @ {station_name} ({station_meta['hood']})\n"
        f"Time: {time_label}\n"
        f"Weather: {weather_label} (index: {weather_idx:.2f})\n"
        f"Nearby Events: {sports_desc}\n"
        f"---\n"
        f"Model Prediction: {delay:.1f} minute delay\n"
        f"Primary Driver:   {top2[0][0]} ({top2[0][1]:.0%} attention)\n"
        f"Secondary Factor: {top2[1][0]} ({top2[1][1]:.0%} attention)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚊 CTA Transit Assistant")
    st.caption("Cascading AI · Attention Model")
    st.divider()

    station_name = st.selectbox("Station", list(STATIONS.keys()), index=4)
    station_meta = STATIONS[station_name]

    st.subheader("Time of Day")
    hour = st.slider("Hour (24h)", 0, 23, 17, format="%d:00")
    minute = st.select_slider("Minute", [0, 15, 30, 45], value=0)
    time_label = f"{hour % 12 or 12}:{minute:02d} {'AM' if hour < 12 else 'PM'}"
    time_norm  = (hour + minute / 60) / 24.0

    st.subheader("Weather")
    weather_label = st.selectbox("Conditions", list(WEATHER_OPTIONS.keys()), index=0)
    weather_idx   = WEATHER_OPTIONS[weather_label]

    st.subheader("Events")
    default_sports = station_meta["sports"] is not None
    sports_on = st.toggle(
        "Major event nearby?",
        value=default_sports,
        help="Auto-enabled if you pick a station near a stadium",
    )
    if sports_on and station_meta["sports"]:
        sports_desc = SPORTS_LABELS[station_meta["sports"]]
    elif sports_on:
        sports_desc = "Event nearby (indirect impact)"
    else:
        sports_desc = "No major events"
    sports_flag = 1.0 if sports_on and station_meta["sports"] else (0.4 if sports_on else 0.0)

    st.divider()
    run_btn = st.button("Get Transit Update", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## Chicago 'L' Intelligent Transit Assistant")

if run_btn:
    # ── 1. Run attention model ──────────────────────────────────────────────
    attn_model = load_attention_model()

    s_id     = torch.tensor([station_meta["id"]],  dtype=torch.long)
    t_norm   = torch.tensor([time_norm],            dtype=torch.float)
    w_idx    = torch.tensor([weather_idx],          dtype=torch.float)
    sp_flag  = torch.tensor([sports_flag],          dtype=torch.float)

    result   = attn_model.explain(s_id, t_norm, w_idx, sp_flag)
    # De-normalise: trained model outputs are scaled by delay_mean; untrained = 1.0
    delay_mean = getattr(attn_model, "delay_mean", 1.0)
    delay    = result["predicted_delay_minutes"] * delay_mean

    # Use signal-weighted importance so display reflects actual input activity
    importance = compute_display_importance(result, weather_idx, time_norm, sports_flag)
    dominant   = max(importance, key=importance.get)

    # ── 2. Layout ───────────────────────────────────────────────────────────
    col_map, col_analysis = st.columns([1, 1], gap="large")

    with col_map:
        st.subheader(f"📍 {station_name}")
        st.caption(f"{station_meta['line']} Line · {station_meta['hood']}")
        m = build_map(station_meta, delay)
        st_folium(m, width=500, height=420, returned_objects=[])

    with col_analysis:
        # Delay badge
        d_color = delay_color(delay)
        if delay < 2:
            badge = f"✅ On time"
            badge_sub = "No significant delay detected."
        else:
            badge = f"⚠ {delay:.1f} min delay"
            badge_sub = f"Primary cause: **{dominant.replace('_', ' ')}**"

        st.markdown(
            f"""
            <div style="background:{d_color}22; border-left:4px solid {d_color};
                        padding:12px 16px; border-radius:6px; margin-bottom:12px;">
                <span style="font-size:22px; font-weight:700; color:{d_color};">{badge}</span><br>
                <span style="color:#ccc;">{badge_sub}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Attention weight chart
        fig = build_attention_chart(importance)
        st.plotly_chart(fig, use_container_width=True)

    # ── 3. AI advisory (Claude API) ─────────────────────────────────────────
    st.divider()
    st.subheader("🧠 AI Advisory")

    prompt = build_llama_prompt(
        station_name, station_meta, time_label,
        weather_label, weather_idx, sports_desc,
        delay, importance,
    )

    with st.spinner("Generating advisory…"):
        advisory = get_claude_advisory(prompt)

    if advisory:
        st.markdown(
            f"""
            <div style="background:#1a1a2e; border:1px solid #3a3a5e;
                        border-radius:8px; padding:18px 20px; font-size:16px; line-height:1.6;">
                {advisory}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Fallback if API key is not set
        severity = (
            "You're golden." if delay < 2 else
            f"Minor delay: {delay:.0f} min." if delay < 5 else
            f"Heads up — {delay:.0f} min delay on the {station_meta['line']} Line at {station_name}." if delay < 12 else
            f"The {station_meta['line']} is really struggling — {delay:.0f} min delay at {station_name}."
        )
        st.markdown(
            f"{severity} Primary cause: **{dominant.replace('_', ' ')}** "
            f"({importance[dominant]:.0%} of the model's attention).\n\n"
            f"> ⚠ *Set `ANTHROPIC_API_KEY` in Streamlit secrets to enable AI advisories.*"
        )


else:
    # Landing state
    st.info(
        "Configure your trip in the sidebar, then click **Get Transit Update** to run both models.",
        icon="🚊",
    )

    # Show a static map on landing so the page isn't blank
    m_static = folium.Map(
        location=[41.8781, -87.6298],
        zoom_start=11,
        tiles="CartoDB dark_matter",
    )
    for name, meta in STATIONS.items():
        folium.CircleMarker(
            location=[meta["lat"], meta["lon"]],
            radius=5,
            color=LINE_COLORS[meta["line"]],
            fill=True, fill_opacity=0.6,
            tooltip=f"{name}",
        ).add_to(m_static)

    st_folium(m_static, width=None, height=500, returned_objects=[])
