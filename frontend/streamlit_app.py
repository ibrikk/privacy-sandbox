# frontend/streamlit_app.py

import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime, date, timedelta
from typing import List, Optional, Any, cast
import numpy as np

# Plotly for visualizations
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

API_URL = "http://localhost:8000"

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Synthetic Persona Generator",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1f77b4;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #1f77b4;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .question-text {
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.3rem;
    }
    .help-text {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .sidebar-section {
        padding: 0.5rem 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "current_section" not in st.session_state:
    st.session_state.current_section = 0

if "survey_data" not in st.session_state:
    st.session_state.survey_data = {}

if "generation_result" not in st.session_state:
    st.session_state.generation_result = None

if "view_mode" not in st.session_state:
    st.session_state.view_mode = "survey"  # "survey" or "results"

# ============================================================
# CONSTANTS (Match Enums in models.py)
# ============================================================

AGE_OPTIONS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

AREA_TYPE_OPTIONS = {
    "Urban (city center)": "urban",
    "Suburban": "suburban",
    "Rural": "rural",
}

WAKE_TIME_OPTIONS = {
    "Before 6:00 AM": "before_6am",
    "6:00 – 7:59 AM": "6am-8am",
    "8:00 – 9:59 AM": "8am-10am",
    "10:00 – 11:59 AM": "10am-12pm",
    "12:00 PM or later": "after_12pm",
}

SLEEP_TIME_OPTIONS = {
    "Before 10:00 PM": "before_10pm",
    "10:00 – 11:59 PM": "10pm-12am",
    "12:00 – 1:59 AM": "12am-2am",
    "2:00 – 3:59 AM": "2am-4am",
    "4:00 AM or later": "after_4am",
}

CHRONOTYPE_OPTIONS = {
    "Definitely a morning person": "definitely_morning",
    "More a morning person than an evening person": "more_morning",
    "Neither": "neither",
    "More an evening person than a morning person": "more_evening",
    "Definitely an evening person": "definitely_evening",
}

PEAK_USAGE_OPTIONS = {
    "Morning (6 AM – 12 PM)": "morning",
    "Afternoon (12 PM – 6 PM)": "afternoon",
    "Evening (6 PM – 12 AM)": "evening",
    "Late night (12 AM – 6 AM)": "late_night",
    "Evenly distributed throughout the day": "evenly_distributed",
}

ROUTINE_OPTIONS = {
    "Very structured (consistent daily schedule)": "very_structured",
    "Somewhat structured": "somewhat_structured",
    "Mixed": "mixed",
    "Somewhat unstructured": "somewhat_unstructured",
    "Very unstructured (highly variable)": "very_unstructured",
}

COMMUTE_DAYS_OPTIONS = {
    "0 days (fully remote/stay home)": "0",
    "1-2 days": "1-2",
    "3-4 days": "3-4",
    "5 or more days": "5+",
}

COMMUTE_MODE_OPTIONS = {
    "Driving": "driving",
    "Public transit": "public_transit",
    "Walking or biking": "walking_biking",
    "Mixed modes": "mixed",
    "I don't commute regularly": "stay_home",
}

PHYSICAL_ACTIVITY_OPTIONS = {
    "0 days": "0",
    "1-2 days": "1-2",
    "3-4 days": "3-4",
    "5-6 days": "5-6",
    "Every day": "every_day",
}

COMMUTE_TIME_OPTIONS = {
    "Almost none": "almost_none",
    "Less than 30 minutes": "less_than_30min",
    "30 minutes – 1 hour": "30-60min",
    "1 – 2 hours": "1-2hours",
    "More than 2 hours": "more_than_2hours",
}

SCREEN_TIME_OPTIONS = {
    "Less than 1 hour": "less_than_1hour",
    "1 – 2 hours": "1-2hours",
    "2 – 4 hours": "2-4hours",
    "4 – 6 hours": "4-6hours",
    "More than 6 hours": "more_than_6hours",
}

CHECKING_FREQUENCY_OPTIONS = {
    "Less than once per hour": "less_than_1_per_hour",
    "1-2 times per hour": "1-2_per_hour",
    "3-4 times per hour": "3-4_per_hour",
    "5-6 times per hour": "5-6_per_hour",
    "More than 6 times per hour": "more_than_6_per_hour",
}

SESSION_TYPE_OPTIONS = {
    "Mostly very short checks (a few seconds)": "very_short_checks",
    "Mostly short checks (under 1 minute)": "short_checks",
    "Mostly medium sessions (1-5 minutes)": "medium_sessions",
    "Mostly longer sessions (more than 5 minutes)": "long_sessions",
    "A mix of all types": "mixed",
}

GLANCE_FREQUENCY_OPTIONS = {
    "Rarely": "rarely",
    "Sometimes": "sometimes",
    "Often": "often",
    "Very often": "very_often",
}

WORK_RESTRICTION_OPTIONS = {
    "I can use it freely": "use_freely",
    "I use it occasionally when I have a moment": "occasionally",
    "Only briefly when necessary": "briefly_when_necessary",
    "Not applicable (I don't work in an office/workplace)": "not_applicable",
}

EVENING_CHANGE_OPTIONS = {
    "Much shorter than during the day": "much_shorter",
    "Somewhat shorter": "somewhat_shorter",
    "About the same": "about_same",
    "Somewhat longer": "somewhat_longer",
    "Much longer than during the day": "much_longer",
}

APP_CATEGORY_OPTIONS = {
    "Messaging (WhatsApp, Messenger, Telegram)": "messaging",
    "Social media (Instagram, Facebook, TikTok, X/Twitter)": "social_media",
    "Music/Audio (Spotify, Apple Music, podcasts)": "music_audio",
    "Video streaming (YouTube, Netflix, TikTok)": "video_streaming",
    "Maps/Navigation (Google Maps, Waze)": "maps_navigation",
    "Shopping (Amazon, eBay, etc.)": "shopping",
    "News/Reading (news apps, Reddit, blogs)": "news_reading",
    "Productivity/Work (email, calendar, docs)": "productivity_work",
    "Fitness/Health (workout apps, health tracking)": "fitness_health",
    "Games": "games",
    "Other": "other",
}

USAGE_REASON_OPTIONS = {
    "Messaging or talking with others": "messaging_talking",
    "Browsing social media": "social_media",
    "Watching videos": "watching_videos",
    "Listening to music or podcasts": "listening_music_podcasts",
    "Reading news or articles": "reading_news",
    "Navigation/maps": "navigation_maps",
    "Shopping": "shopping",
    "Work or productivity": "work_productivity",
    "Gaming": "gaming",
    "Other": "other",
}

EXPLORATION_OPTIONS = {
    "I mostly stick to familiar apps and content": "mostly_familiar",
    "I sometimes explore but prefer what I know": "sometimes_explore",
    "I enjoy exploring new apps/content regularly": "both_equally",
}

# ============================================================
# API FUNCTIONS
# ============================================================


def api_health_check() -> bool:
    """Check if API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def fetch_persona_by_id(persona_id: str) -> Optional[dict]:
    """Fetch a persona by ID from the API."""
    try:
        response = requests.get(f"{API_URL}/persona/{persona_id}", timeout=30)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to API server")
        return None
    except Exception as e:
        st.error(f"Error fetching persona: {e}")
        return None


def generate_persona_api(payload: dict) -> Optional[dict]:
    """Call the generate endpoint."""
    try:
        response = requests.post(
            f"{API_URL}/generate",
            json=payload,
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Could not connect to the API server. Make sure `server.py` is running on port 8000."
        )
        return None
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. The server may be overloaded.")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def get_selection_value(display_to_value: dict, display_key: str) -> str:
    """Convert display text to API value."""
    return display_to_value.get(display_key, display_key)


def multi_select_to_values(display_to_value: dict, selected_displays: list) -> list:
    """Convert list of display texts to API values."""
    return [display_to_value.get(d, d) for d in selected_displays]


def normalize_persona_data(data: dict) -> dict:
    """
    Normalize persona data from different sources.
    """
    if "survey" in data and "dimensions" in data and "parameters" in data:
        return {
            "persona_id": data.get("persona_id", "unknown"),
            "dimensions": data.get("dimensions", {}),
            "parameters": data.get("parameters", {}),
            "schedule": data.get("schedule"),
            "survey_summary": {
                "city": data.get("survey", {}).get("city", "N/A"),
                "occupation": data.get("survey", {}).get("occupation", "N/A"),
                "age_range": data.get("survey", {}).get("age_range", "N/A"),
            },
        }
    return data


def get_sessions_df(schedule: dict) -> pd.DataFrame:
    """Extract sessions from schedule into a DataFrame."""
    if not schedule or "sessions" not in schedule:
        return pd.DataFrame()

    sessions = schedule.get("sessions", [])
    if not sessions:
        return pd.DataFrame()

    df = pd.DataFrame(sessions)

    # Parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["minute"] = df["timestamp"].dt.minute
        df["time_decimal"] = df["hour"] + df["minute"] / 60

    return df


def normalize_context_label(value) -> str:
    """Collapse commute subtypes into a single display label."""
    if value is None:
        return "unknown"

    text = str(value).strip().lower()
    if text.startswith("commute_"):
        return "commute"
    return text


# ============================================================
# SIDEBAR
# ============================================================


def render_sidebar():
    """Render the sidebar with navigation and load options."""

    st.sidebar.title("📱 Persona Generator")

    # API Status
    api_ok = api_health_check()
    if api_ok:
        st.sidebar.success("✅ API Connected", icon="🟢")
    else:
        st.sidebar.error("❌ API Offline", icon="🔴")
        st.sidebar.caption("Start server: `python server.py`")

    st.sidebar.divider()

    # ========== LOAD BY ID SECTION ==========
    st.sidebar.subheader("🔍 Load Persona by ID")

    query_params = st.query_params
    url_persona_id = query_params.get("id", "")

    persona_id_input = st.sidebar.text_input(
        "Persona ID",
        value=url_persona_id,
        placeholder="e.g., abc123-def456-...",
        key="sidebar_persona_id",
        label_visibility="collapsed",
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        load_clicked = st.button("Load", width="stretch", key="load_btn")
    with col2:
        clear_clicked = st.button("Clear", width="stretch", key="clear_btn")

    if load_clicked and persona_id_input:
        with st.sidebar.status("Loading...", expanded=True):
            data = fetch_persona_by_id(persona_id_input.strip())
            if data:
                st.session_state.generation_result = normalize_persona_data(data)
                st.session_state.view_mode = "results"
                st.query_params["id"] = persona_id_input.strip()
                st.rerun()
            else:
                st.sidebar.error(f"Persona not found: {persona_id_input[:20]}...")

    if clear_clicked:
        st.session_state.generation_result = None
        st.session_state.view_mode = "survey"
        st.query_params.clear()
        st.rerun()

    if url_persona_id and st.session_state.generation_result is None:
        data = fetch_persona_by_id(url_persona_id.strip())
        if data:
            st.session_state.generation_result = normalize_persona_data(data)
            st.session_state.view_mode = "results"

    st.sidebar.divider()

    # ========== NAVIGATION ==========
    st.sidebar.subheader("🧭 Navigation")

    if st.sidebar.button(
        "📝 New Survey", width="stretch", type="secondary", key="nav_survey"
    ):
        st.session_state.view_mode = "survey"
        st.session_state.generation_result = None
        st.query_params.clear()
        st.rerun()

    if st.session_state.generation_result:
        if st.sidebar.button(
            "📊 View Results",
            width="stretch",
            type="primary",
            key="nav_results",
        ):
            st.session_state.view_mode = "results"
            st.rerun()


# ============================================================
# NEW DASHBOARD VISUALIZATIONS
# ============================================================


def render_glance_vs_engaged_pie(df: pd.DataFrame) -> go.Figure:
    """Pie chart comparing glances vs engaged sessions."""
    if df.empty or "is_glance" not in df.columns:
        return go.Figure()

    glance_counts = df["is_glance"].value_counts()
    labels = ["Engaged Sessions", "Quick Glances"]
    values = [glance_counts.get(False, 0), glance_counts.get(True, 0)]
    colors = ["#3498db", "#e74c3c"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker=dict(colors=colors),
                textinfo="label+percent",
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title="Glances vs Engaged Sessions",
        height=350,
        margin=dict(t=60, b=20, l=20, r=20),
        showlegend=True,
    )

    return fig


def render_glance_duration_comparison(df: pd.DataFrame) -> go.Figure:
    """Box plot comparing duration of glances vs non-glances."""
    if df.empty or "is_glance" not in df.columns:
        return go.Figure()

    df_plot = df.copy()
    df_plot["Session Type"] = df_plot["is_glance"].map(
        lambda x: "Glance" if x else "Engaged"
    )

    fig = go.Figure()

    for session_type, color in [("Glance", "#e74c3c"), ("Engaged", "#3498db")]:
        data = df_plot[df_plot["Session Type"] == session_type]["duration_seconds"]
        fig.add_trace(
            go.Box(y=data, name=session_type, marker_color=color, boxmean=True)
        )

    fig.update_layout(
        title="Session Duration: Glances vs Engaged",
        yaxis_title="Duration (seconds)",
        height=350,
        showlegend=False,
    )

    return fig


def render_glance_rate_by_app(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing glance rate by app category."""
    if df.empty or "app_category" not in df.columns:
        return go.Figure()

    stats = (
        df.groupby("app_category").agg({"is_glance": ["sum", "count"]}).reset_index()
    )
    stats.columns = ["app_category", "glances", "total"]
    stats["glance_rate"] = (stats["glances"] / stats["total"] * 100).round(1)
    stats = stats.sort_values("glance_rate", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=stats["glance_rate"],
            y=stats["app_category"].str.replace("_", " ").str.title(),
            orientation="h",
            marker_color=stats["glance_rate"],
            marker_colorscale="RdYlGn_r",
            text=stats["glance_rate"].astype(str) + "%",
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Glance Rate by App Category",
        xaxis_title="Glance Rate (%)",
        height=400,
        margin=dict(l=150),
    )

    return fig


def render_glance_timeline(df: pd.DataFrame) -> go.Figure:
    """Timeline showing glances vs engaged sessions throughout the day."""
    if df.empty or "hour" not in df.columns:
        return go.Figure()

    hourly = df.groupby(["hour", "is_glance"]).size().unstack(fill_value=0)
    hourly.columns = ["Engaged", "Glance"] if False in hourly.columns else ["Glance"]

    fig = go.Figure()

    if "Engaged" in hourly.columns:
        fig.add_trace(
            go.Bar(
                x=hourly.index,
                y=hourly["Engaged"],
                name="Engaged",
                marker_color="#3498db",
            )
        )

    if "Glance" in hourly.columns:
        fig.add_trace(
            go.Bar(
                x=hourly.index,
                y=hourly["Glance"],
                name="Glance",
                marker_color="#e74c3c",
            )
        )

    fig.update_layout(
        title="Glances vs Engaged Sessions by Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Number of Sessions",
        barmode="stack",
        height=350,
        xaxis=dict(tickmode="linear", tick0=0, dtick=2),
    )

    return fig


def render_session_duration_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of session durations."""
    if df.empty or "duration_seconds" not in df.columns:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df["duration_seconds"], nbinsx=30, marker_color="#9b59b6", opacity=0.75
        )
    )

    # Add mean line
    mean_dur = df["duration_seconds"].mean()
    fig.add_vline(
        x=mean_dur,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_dur:.1f}s",
    )

    fig.update_layout(
        title="Distribution of Session Durations",
        xaxis_title="Duration (seconds)",
        yaxis_title="Count",
        height=350,
    )

    return fig


def render_duration_by_app_box(df: pd.DataFrame) -> go.Figure:
    """Box plot of duration by app category."""
    if df.empty:
        return go.Figure()

    fig = px.box(
        df,
        x="app_category",
        y="duration_seconds",
        color="app_category",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    fig.update_layout(
        title="Session Duration by App Category",
        xaxis_title="",
        yaxis_title="Duration (seconds)",
        showlegend=False,
        height=400,
        xaxis_tickangle=-45,
    )

    return fig


def render_app_category_sessions_bar(df: pd.DataFrame) -> go.Figure:
    """Bar chart of session count by app category."""
    if df.empty or "app_category" not in df.columns:
        return go.Figure()

    counts = df["app_category"].value_counts().sort_values(ascending=True)

    fig = go.Figure(
        go.Bar(
            x=counts.values,
            y=counts.index.str.replace("_", " ").str.title(),
            orientation="h",
            marker_color=px.colors.qualitative.Plotly[: len(counts)],
            text=counts.values,
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Number of Sessions by App Category",
        xaxis_title="Session Count",
        height=400,
        margin=dict(l=150),
    )

    return fig


def render_app_total_time_bar(df: pd.DataFrame) -> go.Figure:
    """Bar chart of total time by app category."""
    if (
        df.empty
        or "app_category" not in df.columns
        or "duration_seconds" not in df.columns
    ):
        return go.Figure()

    grouped = df.groupby("app_category")["duration_seconds"].sum()
    time_by_app: pd.Series = pd.Series(grouped, dtype="float64") / 60.0

    fig = go.Figure(
        go.Bar(
            x=time_by_app.values,
            y=time_by_app.index.astype(str).str.replace("_", " ").str.title(),
            orientation="h",
            marker_color=px.colors.qualitative.Set3[: len(time_by_app)],
            text=[f"{v:.1f} min" for v in time_by_app.values],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Total Time by App Category",
        xaxis_title="Time (minutes)",
        height=400,
        margin=dict(l=150),
    )

    return fig


def render_app_heatmap_by_hour(df: pd.DataFrame) -> go.Figure:
    """Heatmap of app usage by hour."""
    if df.empty or "hour" not in df.columns:
        return go.Figure()

    pivot = df.groupby(["hour", "app_category"]).size().unstack(fill_value=0)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values.T,
            x=pivot.index,
            y=[cat.replace("_", " ").title() for cat in pivot.columns],
            colorscale="Blues",
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title="App Usage Heatmap by Hour",
        xaxis_title="Hour of Day",
        yaxis_title="App Category",
        height=450,
        xaxis=dict(tickmode="linear", tick0=0, dtick=2),
    )

    return fig


def render_context_distribution_pie(df: pd.DataFrame) -> go.Figure:
    """Pie chart of total time by context."""
    if df.empty or "context" not in df.columns or "duration_seconds" not in df.columns:
        return go.Figure()

    df_plot = df.copy()
    df_plot["context_group"] = df_plot["context"].apply(normalize_context_label)

    context_values = df_plot["context_group"].tolist()
    duration_values_raw = df_plot["duration_seconds"].tolist()

    duration_values: list[float] = []
    for value in duration_values_raw:
        try:
            duration_values.append(float(value))
        except (TypeError, ValueError):
            duration_values.append(0.0)

    context_base = pd.DataFrame(
        {
            "context_group": context_values,
            "duration_seconds": duration_values,
        }
    )

    context_minutes_df = (
        context_base.groupby("context_group", dropna=False)
        .agg(total_seconds=("duration_seconds", "sum"))
        .reset_index()
    )

    context_minutes_df["total_minutes"] = [
        float(v) / 60.0 for v in context_minutes_df["total_seconds"].tolist()
    ]

    context_minutes_df = context_minutes_df.sort_values(
        by="total_seconds", ascending=False
    )

    labels = [
        str(c).replace("_", " ").title()
        for c in context_minutes_df["context_group"].tolist()
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=context_minutes_df["total_minutes"].tolist(),
                hole=0.3,
                marker=dict(colors=px.colors.qualitative.Pastel),
            )
        ]
    )

    fig.update_layout(title="Total Time by Context", height=350)

    return fig


def render_activity_breakdown_bar(df: pd.DataFrame) -> go.Figure:
    """Stacked bar of app usage by activity using total minutes, not raw session count."""
    if (
        df.empty
        or "activity" not in df.columns
        or "app_category" not in df.columns
        or "duration_seconds" not in df.columns
    ):
        return go.Figure()

    pivot = (
        df.groupby(["activity", "app_category"])["duration_seconds"]
        .sum()
        .unstack(fill_value=0)
        / 60.0
    )

    fig = go.Figure()
    activity_labels = [str(a).replace("_", " ").title() for a in pivot.index]

    for col in pivot.columns:
        fig.add_trace(
            go.Bar(
                name=str(col).replace("_", " ").title(),
                x=activity_labels,
                y=pivot[col],
            )
        )

    fig.update_layout(
        title="App Usage by Activity",
        xaxis_title="Activity",
        yaxis_title="Total Time (min)",
        barmode="stack",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def render_user_initiated_pie(df: pd.DataFrame) -> go.Figure:
    """Pie chart of user-initiated vs system-triggered sessions."""
    if df.empty or "is_user_initiated" not in df.columns:
        return go.Figure()

    counts = df["is_user_initiated"].value_counts()

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["User Initiated", "System Triggered"],
                values=[counts.get(True, 0), counts.get(False, 0)],
                hole=0.4,
                marker=dict(colors=["#27ae60", "#f39c12"]),
            )
        ]
    )

    fig.update_layout(title="User Initiated vs System Triggered", height=300)

    return fig


def render_user_initiated_by_app(df: pd.DataFrame) -> go.Figure:
    """Bar chart of user-initiated rate by app category."""
    if df.empty:
        return go.Figure()
    stats = (
        df.groupby("app_category")
        .agg({"is_user_initiated": ["sum", "count"]})
        .reset_index()
    )
    stats.columns = ["app_category", "user_initiated", "total"]
    stats["user_rate"] = (stats["user_initiated"] / stats["total"] * 100).round(1)
    stats = stats.sort_values("user_rate", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=stats["user_rate"],
            y=stats["app_category"].str.replace("_", " ").str.title(),
            orientation="h",
            marker_color=stats["user_rate"],
            marker_colorscale="RdYlGn_r",
            text=stats["user_rate"].astype(str) + "%",
            textposition="outside",
        )
    )

    fig.update_layout(
        title="User-Initiated Rate by App Category",
        xaxis_title="User-Initiated Rate (%)",
        height=400,
        margin=dict(l=150),
    )

    return fig


def render_sessions_by_hour_line(df: pd.DataFrame) -> go.Figure:
    """Line chart showing session count by hour."""
    if df.empty or "hour" not in df.columns:
        return go.Figure()

    hourly = df.groupby("hour").size().reindex(range(24), fill_value=0)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hourly.index,
            y=hourly.values,
            mode="lines+markers",
            fill="tozeroy",
            marker=dict(size=8, color="#3498db"),
            line=dict(width=3, color="#3498db"),
            fillcolor="rgba(52, 152, 219, 0.3)",
        )
    )

    # Mark peak hour
    peak_hour = hourly.idxmax()
    fig.add_annotation(
        x=peak_hour,
        y=hourly[peak_hour],
        text=f"Peak: {peak_hour}:00",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#e74c3c",
        font=dict(color="#e74c3c", size=12),
    )

    fig.update_layout(
        title="Session Frequency Throughout the Day",
        xaxis_title="Hour of Day",
        yaxis_title="Number of Sessions",
        height=350,
        xaxis=dict(tickmode="linear", tick0=0, dtick=2),
    )

    return fig


def render_time_between_sessions(df: pd.DataFrame) -> go.Figure:
    """Histogram of time gaps between consecutive sessions."""
    if df.empty or "timestamp" not in df.columns or len(df) < 2:
        return go.Figure()

    df_sorted = df.sort_values("timestamp")
    gaps = df_sorted["timestamp"].diff().dt.total_seconds() / 60  # Convert to minutes
    gaps = gaps.dropna()
    gaps = gaps[gaps < 120]  # Filter out gaps > 2 hours for better visualization

    if len(gaps) == 0:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Histogram(x=gaps, nbinsx=30, marker_color="#2ecc71", opacity=0.75))

    median_gap = gaps.median()
    fig.add_vline(
        x=median_gap,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Median: {median_gap:.1f} min",
    )

    fig.update_layout(
        title="Time Between Consecutive Sessions",
        xaxis_title="Gap (minutes)",
        yaxis_title="Count",
        height=350,
    )

    return fig


def render_location_sessions_bar(df: pd.DataFrame) -> go.Figure:
    """Bar chart of sessions by location."""
    if df.empty or "location_label" not in df.columns:
        return go.Figure()

    loc_counts = df["location_label"].value_counts()

    fig = go.Figure(
        go.Bar(
            x=loc_counts.values,
            y=loc_counts.index.str.replace("_", " ").str.title(),
            orientation="h",
            marker_color=px.colors.qualitative.Safe[: len(loc_counts)],
            text=loc_counts.values,
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Sessions by Location",
        xaxis_title="Session Count",
        height=350,
        margin=dict(l=120),
    )

    return fig


def render_location_map(df: pd.DataFrame) -> go.Figure:
    """Scatter mapbox of session locations."""
    if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        return go.Figure()

    # Filter out null coordinates
    df_map = df.dropna(subset=["latitude", "longitude"])
    if "location_label" in df_map.columns:
        df_map = df_map[df_map["location_label"] != "commute"]

    if df_map.empty:
        return go.Figure()

    # Aggregate by location
    loc_agg = (
        df_map.groupby(["latitude", "longitude", "location_label"])
        .agg({"session_id": "count", "duration_seconds": "sum"})
        .reset_index()
    )
    loc_agg.columns = [
        "latitude",
        "longitude",
        "location_label",
        "session_count",
        "total_duration",
    ]
    loc_agg["total_minutes"] = loc_agg["total_duration"] / 60

    fig = go.Figure(
        go.Scattermapbox(
            lat=loc_agg["latitude"],
            lon=loc_agg["longitude"],
            mode="markers",
            marker=dict(
                size=np.sqrt(loc_agg["session_count"]) * 8 + 6,
                color=loc_agg["session_count"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Sessions"),
            ),
            text=loc_agg.apply(
                lambda r: f"{r['location_label']}<br>Sessions: {r['session_count']}<br>Time: {r['total_minutes']:.1f} min",
                axis=1,
            ),
            hoverinfo="text",
        )
    )

    center_lat = loc_agg["latitude"].mean()
    center_lon = loc_agg["longitude"].mean()

    fig.update_layout(
        mapbox=dict(
            style="carto-positron", center=dict(lat=center_lat, lon=center_lon), zoom=11
        ),
        title="Geographic Distribution of Phone Usage",
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def render_activity_time_distribution(df: pd.DataFrame) -> go.Figure:
    """Stacked area chart showing activity distribution over time."""
    if df.empty or "hour" not in df.columns or "activity" not in df.columns:
        return go.Figure()

    pivot = df.groupby(["hour", "activity"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(range(24), fill_value=0)

    fig = go.Figure()

    colors = px.colors.qualitative.Set2
    for i, col in enumerate(pivot.columns):
        fig.add_trace(
            go.Scatter(
                x=pivot.index,
                y=pivot[col],
                name=col.replace("_", " ").title(),
                mode="lines",
                stackgroup="one",
                line=dict(width=0.5),
                fillcolor=colors[i % len(colors)],
            )
        )

    fig.update_layout(
        title="Activity Distribution Throughout the Day",
        xaxis_title="Hour of Day",
        yaxis_title="Number of Sessions",
        height=400,
        xaxis=dict(tickmode="linear", tick0=0, dtick=2),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


def render_dimension_gauge(
    dimension_name: str, value: float, low_label: str, high_label: str
) -> go.Figure:
    """Create a gauge chart for a behavioral dimension."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            domain=dict(x=[0, 1], y=[0, 1]),
            title=dict(
                text=dimension_name.replace("_", " ").title(), font=dict(size=14)
            ),
            number=dict(font=dict(size=24)),
            gauge=dict(
                axis=dict(range=[0, 1], tickwidth=1, tickcolor="darkgray"),
                bar=dict(color="#3498db"),
                bgcolor="white",
                borderwidth=2,
                bordercolor="gray",
                steps=[
                    dict(range=[0, 0.33], color="#ffebee"),
                    dict(range=[0.33, 0.66], color="#fff3e0"),
                    dict(range=[0.66, 1], color="#e8f5e9"),
                ],
                threshold=dict(
                    line=dict(color="red", width=2), thickness=0.75, value=value
                ),
            ),
        )
    )

    fig.add_annotation(
        x=0.1, y=-0.1, text=low_label, showarrow=False, font=dict(size=10, color="gray")
    )

    fig.add_annotation(
        x=0.9,
        y=-0.1,
        text=high_label,
        showarrow=False,
        font=dict(size=10, color="gray"),
    )

    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=30))

    return fig


def render_dimensions_radar(dimensions: dict) -> go.Figure:
    """Radar chart showing all behavioral dimensions."""
    if not dimensions:
        return go.Figure()

    dimension_labels = {
        "chronotype_score": "Chronotype",
        "usage_intensity": "Usage Intensity",
        "attentional_granularity": "Attentional Gran.",
        "contextual_sensitivity": "Context Sensitivity",
        "social_orientation": "Social Orientation",
        "mobility_diversity": "Mobility Diversity",
        "routine_stability": "Routine Stability",
        "novelty_seeking": "Novelty Seeking",
    }

    categories = []
    values = []

    for dim_key, dim_label in dimension_labels.items():
        if dim_key in dimensions:
            categories.append(dim_label)
            values.append(dimensions[dim_key])

    # Close the radar chart
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            fillcolor="rgba(52, 152, 219, 0.3)",
            line=dict(color="#3498db", width=2),
            marker=dict(size=8, color="#3498db"),
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickvals=[0.25, 0.5, 0.75, 1.0])
        ),
        title="Behavioral Dimensions Profile",
        height=450,
        showlegend=False,
    )

    return fig


def render_session_burst_analysis(df: pd.DataFrame) -> go.Figure:
    """Identify and visualize session bursts."""
    if df.empty or "timestamp" not in df.columns or len(df) < 3:
        return go.Figure()

    df_sorted = df.sort_values("timestamp").copy()
    df_sorted["gap_minutes"] = df_sorted["timestamp"].diff().dt.total_seconds() / 60

    # Define burst as sessions within 5 minutes of each other
    burst_threshold = 5
    df_sorted["new_burst"] = df_sorted["gap_minutes"].isna() | (
        df_sorted["gap_minutes"] > burst_threshold
    )
    df_sorted["burst_id"] = df_sorted["new_burst"].cumsum()

    burst_sizes = df_sorted.groupby("burst_id").size()

    fig = go.Figure()

    fig.add_trace(go.Histogram(x=burst_sizes, marker_color="#9b59b6", opacity=0.75))

    fig.update_layout(
        title=f"Session Burst Sizes (sessions within {burst_threshold} min)",
        xaxis_title="Sessions per Burst",
        yaxis_title="Number of Bursts",
        height=300,
    )

    return fig


def render_weekday_pattern(df: pd.DataFrame, schedule: dict) -> go.Figure:
    """Show if it's a weekday vs weekend pattern."""
    if not schedule:
        return go.Figure()

    day_type = schedule.get("day_type", "unknown")

    fig = go.Figure(
        go.Indicator(
            mode="number+delta",
            value=len(df) if not df.empty else 0,
            title=dict(text=f"Sessions ({day_type.title()})"),
            delta=dict(reference=50, relative=True),
            domain=dict(x=[0, 1], y=[0, 1]),
        )
    )

    fig.update_layout(height=200)

    return fig


from typing import cast
import pandas as pd


def render_top_sessions_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a table of top 10 longest sessions."""
    if df.empty or "duration_seconds" not in df.columns:
        return pd.DataFrame()

    required_columns = [
        "timestamp",
        "app_category",
        "duration_seconds",
        "activity",
        "context",
        "location_label",
    ]
    available_columns = [col for col in required_columns if col in df.columns]

    top_sessions = cast(
        pd.DataFrame,
        df.nlargest(10, "duration_seconds")[available_columns].copy(),
    )

    if "timestamp" in top_sessions.columns:
        timestamp_series = pd.Series(
            pd.to_datetime(top_sessions["timestamp"], errors="coerce"),
            index=top_sessions.index,
        )
        top_sessions["timestamp"] = timestamp_series.dt.strftime("%H:%M:%S")

    duration_series = pd.Series(
        pd.to_numeric(top_sessions["duration_seconds"], errors="coerce"),
        index=top_sessions.index,
        dtype="float64",
    ).fillna(0.0)

    top_sessions["duration"] = duration_series.map(
        lambda x: f"{int(x // 60)}m {int(x % 60)}s"
    )

    for col in ["app_category", "activity", "context", "location_label"]:
        if col in top_sessions.columns:
            string_series = pd.Series(
                top_sessions[col],
                index=top_sessions.index,
                dtype="string",
            )
            top_sessions[col] = string_series.str.replace(
                "_", " ", regex=False
            ).str.title()

    output_columns = [
        col
        for col in [
            "timestamp",
            "app_category",
            "duration",
            "activity",
            "context",
            "location_label",
        ]
        if col in top_sessions.columns
    ]

    return cast(pd.DataFrame, top_sessions.loc[:, output_columns])


def render_summary_metrics(df: pd.DataFrame, schedule: dict) -> dict:
    """Calculate summary metrics for the dashboard."""
    if df.empty:
        return {}

    total_sessions = len(df)
    total_screen_time = df["duration_seconds"].sum() / 60  # minutes
    avg_session_duration = df["duration_seconds"].mean()
    glance_rate = (
        (df["is_glance"].sum() / total_sessions * 100)
        if "is_glance" in df.columns
        else 0
    )
    user_initiated_rate = (
        (df["is_user_initiated"].sum() / total_sessions * 100)
        if "is_user_initiated" in df.columns
        else 0
    )

    # Peak hour
    if "hour" in df.columns:
        peak_hour = df["hour"].mode().iloc[0] if len(df["hour"].mode()) > 0 else 12
    else:
        peak_hour = "N/A"

    # Most used app
    if "app_category" in df.columns:
        most_used_app = (
            df["app_category"].mode().iloc[0]
            if len(df["app_category"].mode()) > 0
            else "N/A"
        )
        most_used_app = most_used_app.replace("_", " ").title()
    else:
        most_used_app = "N/A"

    return {
        "total_sessions": total_sessions,
        "total_screen_time": total_screen_time,
        "avg_session_duration": avg_session_duration,
        "glance_rate": glance_rate,
        "user_initiated_rate": user_initiated_rate,
        "peak_hour": peak_hour,
        "most_used_app": most_used_app,
    }


def render_overview_tab(
    df: pd.DataFrame, schedule: dict, dimensions: dict, survey_summary: dict
):
    """Render the Overview tab with key metrics and summary."""
    st.markdown(
        '<p class="section-header">📊 Daily Summary</p>', unsafe_allow_html=True
    )

    metrics = render_summary_metrics(df, schedule)

    if metrics:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Sessions",
                f"{metrics['total_sessions']}",
                help="Total number of phone usage sessions",
            )

        with col2:
            st.metric(
                "Screen Time",
                f"{metrics['total_screen_time']:.1f} min",
                help="Total time spent on phone",
            )

        with col3:
            st.metric(
                "Avg Session",
                f"{metrics['avg_session_duration']:.1f}s",
                help="Average session duration in seconds",
            )

        with col4:
            st.metric(
                "Glance Rate",
                f"{metrics['glance_rate']:.1f}%",
                help="Percentage of quick glances vs engaged sessions",
            )

        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.metric(
                "User Initiated",
                f"{metrics['user_initiated_rate']:.1f}%",
                help="Percentage of sessions initiated by user",
            )

        with col6:
            st.metric(
                "Peak Hour",
                f"{metrics['peak_hour']}:00",
                help="Hour with most phone usage",
            )

        with col7:
            st.metric(
                "Top App",
                metrics["most_used_app"],
                help="Most frequently used app category",
            )

        with col8:
            day_type = schedule.get("day_type", "N/A") if schedule else "N/A"
            st.metric("Day Type", day_type.title(), help="Weekday or Weekend")

    st.divider()

    # Two column layout for overview charts
    col_left, col_right = st.columns(2)

    with col_left:
        fig = render_sessions_by_hour_line(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="overview_sessions_by_hour")

        fig = render_glance_vs_engaged_pie(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="overview_glance_pie")

    with col_right:
        fig = render_app_category_sessions_bar(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="overview_app_category_bar")

        fig = render_user_initiated_pie(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="overview_user_initiated_pie")

    # Persona info
    st.markdown(
        '<p class="section-header">👤 Persona Information</p>', unsafe_allow_html=True
    )

    if survey_summary:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**City:** {survey_summary.get('city', 'N/A')}")
        with col2:
            st.info(f"**Occupation:** {survey_summary.get('occupation', 'N/A')}")
        with col3:
            st.info(f"**Age Range:** {survey_summary.get('age_range', 'N/A')}")


def render_glances_tab(df: pd.DataFrame):
    """Render the Glances Analysis tab."""

    st.markdown(
        '<p class="section-header">👁️ Glance Behavior Analysis</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    **Glances** are quick phone checks (typically < 15 seconds) where users briefly look at their screen 
    without engaging deeply with content. Understanding glance patterns reveals habitual checking behavior.
    """
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = render_glance_vs_engaged_pie(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="glances_glance_pie")

        fig = render_glance_timeline(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="glances_timeline")

    with col2:
        fig = render_glance_duration_comparison(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="glances_duration_comparison")

        fig = render_glance_rate_by_app(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="glances_rate_by_app")

    # Glance statistics
    st.markdown(
        '<p class="section-header">📈 Glance Statistics</p>', unsafe_allow_html=True
    )

    if not df.empty and "is_glance" in df.columns:
        glances = df[df["is_glance"] == True]
        engaged = df[df["is_glance"] == False]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Glances", len(glances))
        with col2:
            st.metric("Total Engaged", len(engaged))
        with col3:
            avg_glance_dur = (
                glances["duration_seconds"].mean() if len(glances) > 0 else 0
            )
            st.metric("Avg Glance Duration", f"{avg_glance_dur:.1f}s")
        with col4:
            avg_engaged_dur = (
                engaged["duration_seconds"].mean() if len(engaged) > 0 else 0
            )
            st.metric("Avg Engaged Duration", f"{avg_engaged_dur:.1f}s")


def render_sessions_tab(df: pd.DataFrame):
    """Render the Session Duration tab."""

    st.markdown(
        '<p class="section-header">⏱️ Session Duration Analysis</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = render_session_duration_histogram(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="sessions_duration_histogram")

        fig = render_time_between_sessions(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="sessions_time_between")

    with col2:
        fig = render_duration_by_app_box(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="sessions_duration_by_app_box")

        fig = render_session_burst_analysis(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="sessions_burst_analysis")

    # Top sessions table
    st.markdown(
        '<p class="section-header">🏆 Longest Sessions</p>', unsafe_allow_html=True
    )

    top_sessions = render_top_sessions_table(df)
    if not top_sessions.empty:
        st.dataframe(
            top_sessions,
            width="stretch",
            hide_index=True,
            column_config={
                "timestamp": "Time",
                "app_category": "App Category",
                "duration": "Duration",
                "activity": "Activity",
                "context": "Context",
                "location_label": "Location",
            },
        )


def render_apps_tab(df: pd.DataFrame):
    """Render the App Usage tab."""

    st.markdown(
        '<p class="section-header">📱 App Category Analysis</p>', unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = render_app_category_sessions_bar(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="apps_app_category_bar")

        fig = render_user_initiated_by_app(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="apps_user_initiated_by_app")

    with col2:
        fig = render_app_total_time_bar(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="apps_total_time_bar")

        fig = render_glance_rate_by_app(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="apps_glance_rate_by_app")

    # Full-width heatmap
    st.markdown(
        '<p class="section-header">🗓️ Hourly App Usage Heatmap</p>',
        unsafe_allow_html=True,
    )

    fig = render_app_heatmap_by_hour(df)
    if fig:
        st.plotly_chart(fig, width="stretch")


def render_temporal_tab(df: pd.DataFrame, schedule: dict):
    """Render the Temporal Patterns tab."""

    st.markdown(
        '<p class="section-header">🕐 Temporal Usage Patterns</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    # with col1:
    #     fig = render_sessions_by_hour_line(df)
    #     if fig:
    #         st.plotly_chart(fig, width="stretch", key="temporal_sessions_by_hour")

    #     fig = render_glance_timeline(df)
    #     if fig:
    #         st.plotly_chart(fig, width="stretch", key="temporal_glance_timeline")

    with col1:
        fig = render_activity_time_distribution(df)
        if fig:
            st.plotly_chart(
                fig, width="stretch", key="temporal_activity_time_distribution"
            )

        fig = render_time_between_sessions(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="temporal_time_between_sessions")

    # Schedule segments visualization
    if schedule and "segments" in schedule:
        st.markdown(
            '<p class="section-header">📅 Daily Schedule Segments</p>',
            unsafe_allow_html=True,
        )

        segments = schedule.get("segments", [])
        if segments:
            segment_data = []
            for seg in segments:
                segment_data.append(
                    {
                        "Time": f"{seg.get('start_time', '')} - {seg.get('end_time', '')}",
                        "Activity": seg.get("activity", "").replace("_", " ").title(),
                        "Context": normalize_context_label(seg.get("context", ""))
                        .replace("_", " ")
                        .title(),
                        "Location": seg.get("location_label", "")
                        .replace("_", " ")
                        .title(),
                    }
                )

            st.dataframe(pd.DataFrame(segment_data), width="stretch", hide_index=True)


def render_context_tab(df: pd.DataFrame):
    """Render the Context & Activity tab."""

    st.markdown(
        '<p class="section-header">🎯 Context & Activity Analysis</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = render_context_distribution_pie(df)
        if fig:
            st.plotly_chart(
                fig,
                width="stretch",
                key="context_distribution_pie",
            )

        if "activity" in df.columns and "duration_seconds" in df.columns:
            activity_raw_values = df["activity"].tolist()
            activity_duration_raw_values = df["duration_seconds"].tolist()

            activity_duration_values: list[float] = []
            for value in activity_duration_raw_values:
                try:
                    activity_duration_values.append(float(value))
                except (TypeError, ValueError):
                    activity_duration_values.append(0.0)

            activity_base_df = pd.DataFrame(
                {
                    "activity": activity_raw_values,
                    "duration_seconds": activity_duration_values,
                }
            )

            activity_totals_df = (
                activity_base_df.groupby("activity", dropna=False)
                .agg(total_seconds=("duration_seconds", "sum"))
                .reset_index()
            )

            activity_total_seconds = activity_totals_df["total_seconds"].tolist()
            activity_total_minutes = [float(v) / 60.0 for v in activity_total_seconds]
            activity_totals_df["total_minutes"] = activity_total_minutes

            activity_totals_df = activity_totals_df.sort_values(
                by="total_seconds",
                ascending=True,
            )

            activity_labels = [
                str(value).replace("_", " ").title()
                for value in activity_totals_df["activity"].tolist()
            ]

            fig = go.Figure(
                go.Bar(
                    x=activity_totals_df["total_minutes"].tolist(),
                    y=activity_labels,
                    orientation="h",
                    marker_color=px.colors.qualitative.Pastel,
                    text=[
                        f"{float(v):.1f} min"
                        for v in activity_totals_df["total_minutes"].tolist()
                    ],
                    textposition="outside",
                )
            )

            fig.update_layout(
                title="Total Time by Activity",
                xaxis_title="Time (min)",
                height=350,
                margin=dict(l=150),
            )

            st.plotly_chart(
                fig,
                width="stretch",
                key="context_activity_time_bar",
            )

    with col2:
        fig = render_activity_breakdown_bar(df)
        if fig:
            st.plotly_chart(
                fig,
                width="stretch",
                key="context_activity_breakdown",
            )

        fig = render_user_initiated_pie(df)
        if fig:
            st.plotly_chart(
                fig,
                width="stretch",
                key="context_user_initiated_pie",
            )

    st.markdown(
        '<p class="section-header">📊 Usage by Context</p>',
        unsafe_allow_html=True,
    )

    if "context" in df.columns and "duration_seconds" in df.columns:
        context_df = df.copy()
        context_df["context_group"] = context_df["context"].apply(
            normalize_context_label
        )

        context_group_raw_values = context_df["context_group"].tolist()
        context_duration_raw_values = context_df["duration_seconds"].tolist()

        context_duration_values: list[float] = []
        for value in context_duration_raw_values:
            try:
                context_duration_values.append(float(value))
            except (TypeError, ValueError):
                context_duration_values.append(0.0)

        if "is_glance" in context_df.columns:
            context_glance_raw_values = context_df["is_glance"].tolist()
        else:
            context_glance_raw_values = [False] * len(context_group_raw_values)

        context_glance_values: list[float] = []
        for value in context_glance_raw_values:
            context_glance_values.append(1.0 if bool(value) else 0.0)

        context_base_df = pd.DataFrame(
            {
                "context_group": context_group_raw_values,
                "duration_seconds": context_duration_values,
                "is_glance": context_glance_values,
            }
        )

        context_stats_df = (
            context_base_df.groupby("context_group", dropna=False)
            .agg(
                Sessions=("duration_seconds", "count"),
                Avg_Duration_s=("duration_seconds", "mean"),
                Total_Time_s=("duration_seconds", "sum"),
                Glance_Rate=("is_glance", "mean"),
            )
            .reset_index()
        )

        context_stats_df["Total_Time_min"] = [
            float(v) / 60.0 for v in context_stats_df["Total_Time_s"].tolist()
        ]

        context_stats_df["Glance_Rate_Display"] = [
            f"{float(v) * 100:.1f}%" for v in context_stats_df["Glance_Rate"].tolist()
        ]

        context_stats_df = context_stats_df.sort_values(
            by="Total_Time_s",
            ascending=False,
        )

        context_labels = [
            str(value).replace("_", " ").title()
            for value in context_stats_df["context_group"].tolist()
        ]

        display_df = pd.DataFrame(
            {
                "Context": context_labels,
                "Sessions": context_stats_df["Sessions"].tolist(),
                "Avg Duration (s)": [
                    round(float(v), 2)
                    for v in context_stats_df["Avg_Duration_s"].tolist()
                ],
                "Total Time (min)": [
                    round(float(v), 1)
                    for v in context_stats_df["Total_Time_min"].tolist()
                ],
                "Glance Rate": context_stats_df["Glance_Rate_Display"].tolist(),
            }
        )

        st.dataframe(display_df, width="stretch")


def render_location_tab(df: pd.DataFrame) -> None:
    """Render the Location Analysis tab."""

    st.markdown(
        '<p class="section-header">📍 Location-Based Analysis</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = render_location_sessions_bar(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="location_sessions_bar")

        if "location_label" in df.columns and "duration_seconds" in df.columns:
            grouped = df.groupby("location_label")["duration_seconds"].sum()
            time_by_loc = pd.Series(grouped, dtype="float64") / 60.0
            time_by_loc = cast(pd.Series, time_by_loc.sort_values(ascending=True))

            location_labels = (
                pd.Index(time_by_loc.index)
                .astype("string")
                .str.replace("_", " ", regex=False)
                .str.title()
            )

            fig = go.Figure(
                go.Bar(
                    x=time_by_loc.to_numpy(),
                    y=location_labels.to_list(),
                    orientation="h",
                    marker_color="#2ecc71",
                    text=[f"{v:.1f} min" for v in time_by_loc.to_list()],
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="Phone Usage Time by Location",
                xaxis_title="Phone Usage Time (minutes)",
                height=350,
                margin=dict(l=120),
            )
            st.plotly_chart(fig, width="stretch", key="location_screen_time_bar")

    with col2:
        fig = render_location_map(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="location_map")


def render_dimensions_tab(dimensions: dict):
    """Render the Behavioral Dimensions tab."""

    st.markdown(
        '<p class="section-header">🧠 Behavioral Dimensions Profile</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1.5])

    with col1:
        fig = render_dimensions_radar(dimensions)
        if fig:
            st.plotly_chart(fig, width="stretch", key="dimensions_radar")

    with col2:
        st.markdown("### Dimension Details")

        # Grid of gauges
        g_col1, g_col2 = st.columns(2)

        dimension_info = [
            ("usage_intensity", "Usage Intensity", "Light", "Heavy"),
            ("chronotype_score", "Chronotype", "Morning", "Evening"),
            ("attentional_granularity", "Attentional Granularity", "Broad", "Focused"),
            ("contextual_sensitivity", "Contextual Sensitivity", "Low", "High"),
            ("social_orientation", "Social Orientation", "Individual", "Social"),
            ("mobility_diversity", "Mobility Diversity", "Static", "Mobile"),
            ("routine_stability", "Routine Stability", "Variable", "Stable"),
            ("novelty_seeking", "Novelty Seeking", "Habitual", "Exploratory"),
        ]

        for i, (key, label, low, high) in enumerate(dimension_info):
            if key in dimensions:
                with g_col1 if i % 2 == 0 else g_col2:
                    fig = render_dimension_gauge(label, dimensions[key], low, high)
                    st.plotly_chart(fig, width="stretch", key=f"dimensions_gauge_{key}")


def render_session_drilldown(df: pd.DataFrame) -> None:
    """Interactive drill-down to explore individual sessions."""
    if df.empty:
        st.warning("No session data available.")
        return

    def _to_float_scalar(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _format_time_scalar(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, pd.Timestamp):
            return value.strftime("%H:%M:%S")
        if isinstance(value, datetime):
            return value.strftime("%H:%M:%S")
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return "N/A"
            parsed = pd.to_datetime(text, errors="coerce")
            if isinstance(parsed, pd.Timestamp):
                return parsed.strftime("%H:%M:%S")
            return "N/A"
        return "N/A"

    def _format_duration_scalar(value: Any) -> tuple[float, str]:
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            seconds = 0.0
        return seconds, f"{int(seconds // 60)}m {int(seconds % 60)}s"

    def _is_true_scalar(value: Any) -> bool:
        return value is True

    st.markdown(
        '<p class="section-header">🔍 Session Explorer</p>',
        unsafe_allow_html=True,
    )
    st.markdown("Select filters to drill down into specific sessions.")

    col1, col2, col3 = st.columns(3)

    with col1:
        if "hour" in df.columns:
            hour_values: list[int] = []
            for raw_value in df["hour"].tolist():
                if raw_value is None or pd.isna(raw_value):
                    continue
                try:
                    hour_values.append(int(float(raw_value)))
                except (TypeError, ValueError):
                    continue

            hours_with_sessions = sorted(set(hour_values))
        else:
            hours_with_sessions = list(range(24))

        hour_options: dict[str, int | None] = {"All Hours": None}
        if "hour" in df.columns:
            hour_numeric = pd.to_numeric(df["hour"], errors="coerce")
            for h in hours_with_sessions:
                session_count = sum(
                    1
                    for raw_value in df["hour"].tolist()
                    if raw_value is not None
                    and not pd.isna(raw_value)
                    and int(float(raw_value)) == h
                )
                label = f"{h}:00 - {h}:59 ({session_count} sessions)"
                hour_options[label] = h
        else:
            for h in hours_with_sessions:
                hour_options[f"{h}:00 - {h}:59"] = h

        selected_hour_label = st.selectbox(
            "🕐 Filter by Hour",
            options=list(hour_options.keys()),
            index=0,
            key="drilldown_hour",
        )
        selected_hour = hour_options[selected_hour_label]

    with col2:
        selected_app: str | None = None
        if "app_category" in df.columns:
            app_series = df["app_category"].dropna().astype("string")
            app_categories = sorted(str(v) for v in app_series.unique().tolist())

            app_options = ["All Apps"] + [
                cat.replace("_", " ").title() for cat in app_categories
            ]

            selected_app_display = st.selectbox(
                "📱 Filter by App",
                app_options,
                index=0,
                key="drilldown_app",
            )
            if selected_app_display != "All Apps":
                selected_app = selected_app_display.lower().replace(" ", "_")

    with col3:
        selected_context: str | None = None
        if "context" in df.columns:
            context_series = df["context"].dropna().map(normalize_context_label)
            contexts = sorted(str(v) for v in context_series.unique().tolist())

            context_options = ["All Contexts"] + [
                ctx.replace("_", " ").title() for ctx in contexts
            ]

            selected_context_display = st.selectbox(
                "🎯 Filter by Context",
                context_options,
                index=0,
                key="drilldown_context",
            )
            if selected_context_display != "All Contexts":
                selected_context = selected_context_display.lower().replace(" ", "_")

    col4, col5, col6 = st.columns(3)

    with col4:
        session_type_filter = st.selectbox(
            "👁️ Session Type",
            ["All", "Glances Only", "Engaged Only"],
            index=0,
            key="drilldown_session_type",
        )

    with col5:
        initiated_filter = st.selectbox(
            "👤 Initiated By",
            ["All", "User Initiated", "System/Notification"],
            index=0,
            key="drilldown_initiated",
        )

    with col6:
        sort_by = st.selectbox(
            "📊 Sort By",
            [
                "Time (earliest first)",
                "Time (latest first)",
                "Duration (longest first)",
                "Duration (shortest first)",
            ],
            index=0,
            key="drilldown_sort",
        )

    filtered_df = df.copy()

    if selected_hour is not None and "hour" in filtered_df.columns:
        hour_numeric = pd.to_numeric(filtered_df["hour"], errors="coerce")
        filtered_df = filtered_df.loc[hour_numeric == selected_hour]

    if selected_app is not None and "app_category" in filtered_df.columns:
        app_series = filtered_df["app_category"].astype("string")
        filtered_df = filtered_df.loc[app_series == selected_app]

    if selected_context is not None and "context" in filtered_df.columns:
        context_series = filtered_df["context"].map(normalize_context_label)
        filtered_df = filtered_df.loc[context_series == selected_context]

    if session_type_filter == "Glances Only" and "is_glance" in filtered_df.columns:
        glance_series = filtered_df["is_glance"].map(lambda v: v is True)
        filtered_df = filtered_df.loc[glance_series]
    elif session_type_filter == "Engaged Only" and "is_glance" in filtered_df.columns:
        glance_series = filtered_df["is_glance"].map(lambda v: v is True)
        filtered_df = filtered_df.loc[~glance_series]

    if (
        initiated_filter == "User Initiated"
        and "is_user_initiated" in filtered_df.columns
    ):
        initiated_series = filtered_df["is_user_initiated"].map(lambda v: v is True)
        filtered_df = filtered_df.loc[initiated_series]
    elif (
        initiated_filter == "System/Notification"
        and "is_user_initiated" in filtered_df.columns
    ):
        initiated_series = filtered_df["is_user_initiated"].map(lambda v: v is True)
        filtered_df = filtered_df.loc[~initiated_series]

    if "timestamp" in filtered_df.columns:
        timestamp_series = pd.to_datetime(filtered_df["timestamp"], errors="coerce")
        if sort_by == "Time (earliest first)":
            filtered_df = filtered_df.assign(_sort_ts=timestamp_series).sort_values(
                "_sort_ts",
                ascending=True,
            )
        elif sort_by == "Time (latest first)":
            filtered_df = filtered_df.assign(_sort_ts=timestamp_series).sort_values(
                "_sort_ts",
                ascending=False,
            )

    if "duration_seconds" in filtered_df.columns:
        duration_series = pd.to_numeric(
            filtered_df["duration_seconds"],
            errors="coerce",
        )
        if sort_by == "Duration (longest first)":
            filtered_df = filtered_df.assign(
                _sort_duration=duration_series
            ).sort_values(
                "_sort_duration",
                ascending=False,
            )
        elif sort_by == "Duration (shortest first)":
            filtered_df = filtered_df.assign(
                _sort_duration=duration_series
            ).sort_values(
                "_sort_duration",
                ascending=True,
            )

    filtered_df = filtered_df.drop(
        columns=["_sort_ts", "_sort_duration"],
        errors="ignore",
    )

    st.divider()

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("Sessions Found", len(filtered_df), delta=f"of {len(df)} total")
    with col_s2:
        if "duration_seconds" in filtered_df.columns and len(filtered_df) > 0:
            total_seconds = 0.0
            for raw_value in filtered_df["duration_seconds"].tolist():
                if (
                    raw_value is None
                    or pd.isna(raw_value)
                    or isinstance(raw_value, bool)
                ):
                    continue
                try:
                    total_seconds += float(raw_value)
                except (TypeError, ValueError):
                    continue

            total_time = total_seconds / 60.0
            st.metric("Total Time", f"{total_time:.1f} min")
        else:
            st.metric("Total Time", "N/A")
    with col_s3:
        if "duration_seconds" in filtered_df.columns and len(filtered_df) > 0:
            duration_values: list[float] = []

            for raw_value in filtered_df["duration_seconds"].tolist():
                if (
                    raw_value is None
                    or isinstance(raw_value, bool)
                    or pd.isna(raw_value)
                ):
                    continue

                try:
                    duration_values.append(float(raw_value))
                except (TypeError, ValueError):
                    continue

            avg_dur_value = (
                sum(duration_values) / len(duration_values)
                if len(duration_values) > 0
                else 0.0
            )

            st.metric("Avg Duration", f"{avg_dur_value:.1f}s")
        else:
            st.metric("Avg Duration", "N/A")
    with col_s4:
        if "is_glance" in filtered_df.columns and len(filtered_df) > 0:
            glance_pct = (
                filtered_df["is_glance"].map(lambda v: v is True).mean() * 100.0
            )
            st.metric("Glance Rate", f"{glance_pct:.1f}%")
        else:
            st.metric("Glance Rate", "N/A")

    st.divider()

    if len(filtered_df) == 0:
        st.info("No sessions match the selected filters. Try adjusting your criteria.")
        return

    display_df = filtered_df.copy()

    if "timestamp" in display_df.columns:
        parsed_timestamps = pd.to_datetime(display_df["timestamp"], errors="coerce")
        display_df["Time"] = parsed_timestamps.dt.strftime("%H:%M:%S").fillna("N/A")

    if "app_category" in display_df.columns:
        display_df["App"] = (
            display_df["app_category"]
            .astype("string")
            .str.replace("_", " ", regex=False)
            .str.title()
        )

    if "duration_seconds" in display_df.columns:
        duration_numeric = pd.to_numeric(
            display_df["duration_seconds"], errors="coerce"
        )

        def _format_duration_display(x: Any) -> str:
            if x is None or isinstance(x, bool) or pd.isna(x):
                return "N/A"
            try:
                seconds = float(x)
            except (TypeError, ValueError):
                return "N/A"
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"

        display_df["Duration"] = [
            _format_duration_display(x) for x in display_df["duration_seconds"].tolist()
        ]

    if "activity" in display_df.columns:
        display_df["Activity"] = (
            display_df["activity"]
            .astype("string")
            .str.replace("_", " ", regex=False)
            .str.title()
        )

    if "context" in display_df.columns:
        display_df["Context"] = display_df["context"].map(
            lambda x: (
                normalize_context_label(x).replace("_", " ").title()
                if pd.notna(x)
                else "N/A"
            )
        )

    if "location_label" in display_df.columns:
        display_df["Location"] = (
            display_df["location_label"]
            .astype("string")
            .str.replace("_", " ", regex=False)
            .str.title()
        )

    if "is_glance" in display_df.columns:
        display_df["Type"] = display_df["is_glance"].map(
            lambda x: "👁️ Glance" if x is True else "📱 Engaged"
        )

    if "is_user_initiated" in display_df.columns:
        display_df["Initiated"] = display_df["is_user_initiated"].map(
            lambda x: "👤 User" if x is True else "🔔 System"
        )

    display_columns = [
        "Time",
        "App",
        "Duration",
        "Activity",
        "Context",
        "Location",
        "Type",
        "Initiated",
    ]
    display_columns = [col for col in display_columns if col in display_df.columns]

    st.markdown(f"**📋 Session List** ({len(filtered_df)} sessions)")

    st.dataframe(
        display_df.loc[:, display_columns],
        use_container_width=True,
        hide_index=True,
        height=min(400, len(filtered_df) * 35 + 38),
    )

    st.divider()
    st.markdown("**📝 Session Details** (click to expand)")

    max_display = 30
    sessions_to_show = cast(pd.DataFrame, filtered_df.head(max_display))

    row_records = cast(list[dict[str, Any]], sessions_to_show.to_dict(orient="records"))

    for row_dict in row_records:

        time_str = _format_time_scalar(row_dict.get("timestamp"))
        app_str = str(row_dict.get("app_category", "unknown")).replace("_", " ").title()

        dur_sec, dur_str = _format_duration_scalar(row_dict.get("duration_seconds"))
        is_glance = _is_true_scalar(row_dict.get("is_glance"))
        is_user_initiated = _is_true_scalar(row_dict.get("is_user_initiated"))

        glance_icon = "👁️" if is_glance else "📱"
        expander_title = f"{glance_icon} **{time_str}** — {app_str} ({dur_str})"

        with st.expander(expander_title):
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.markdown("**📍 Context**")
                st.write(
                    f"Activity: {str(row_dict.get('activity', 'N/A')).replace('_', ' ').title()}"
                )
                st.write(
                    f"Context: {normalize_context_label(row_dict.get('context', 'N/A')).replace('_', ' ').title()}"
                )
                st.write(
                    f"Location: {str(row_dict.get('location_label', 'N/A')).replace('_', ' ').title()}"
                )

            with col_b:
                st.markdown("**📱 Session Info**")
                st.write(f"Type: {'Quick Glance' if is_glance else 'Engaged Session'}")
                st.write(
                    f"Initiated: {'User opened app' if is_user_initiated else 'Notification/System'}"
                )
                st.write(f"Duration: {dur_str} ({dur_sec:.0f}s)")

            with col_c:
                st.markdown("**🗺️ Location Data**")
                lat_raw = row_dict.get("latitude")
                lon_raw = row_dict.get("longitude")

                lat_num = _to_float_scalar(lat_raw)
                lon_num = _to_float_scalar(lon_raw)

                if lat_num is not None and lon_num is not None:
                    st.write(f"Lat: {lat_num:.4f}")
                    st.write(f"Lon: {lon_num:.4f}")
                else:
                    st.write("Coordinates: N/A")

    if len(filtered_df) > max_display:
        st.caption(
            f"⚠️ Showing first {max_display} sessions. Use filters above to narrow down."
        )


def render_results():
    """Render the results dashboard."""
    result = st.session_state.generation_result
    if not result:
        st.warning("No persona data available. Please generate one first.")
        return

    persona_id = result.get("persona_id", "Unknown")
    st.markdown(
        f'<p class="main-header">📊 Persona: {persona_id[:8]}...</p>',
        unsafe_allow_html=True,
    )

    # Extract data
    schedule = result.get("schedule", {})
    dimensions = result.get("dimensions", {})
    survey_summary = result.get("survey_summary", {})

    df = get_sessions_df(schedule)

    if df.empty:
        st.error("No session data found in the generated persona.")
        return

    # Tabs for different analyses
    tabs = st.tabs(
        [
            "📊 Overview",
            # "🧠 Dimensions",
            "🔍 Explorer",
            "👁️ Glances",
            "⏱️ Sessions",
            "📱 Apps",
            "🕐 Temporal",
            "🎯 Context",
            # "📍 Location",
        ]
    )

    with tabs[0]:
        render_overview_tab(df, schedule, dimensions, survey_summary)

    # with tabs[1]:
    #     render_dimensions_tab(dimensions)

    with tabs[1]:
        render_session_drilldown(df)

    with tabs[2]:
        render_glances_tab(df)

    with tabs[3]:
        render_sessions_tab(df)

    with tabs[4]:
        render_apps_tab(df)

    with tabs[5]:
        render_temporal_tab(df, schedule)

    with tabs[6]:
        render_context_tab(df)

    # with tabs[8]:
    #     render_location_tab(df)


def render_survey_form():
    """Render the survey form for generating a new persona."""

    st.markdown(
        '<p class="main-header">📝 Persona Generation Survey</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Fill out the details below to generate a realistic synthetic smartphone usage persona."
    )

    with st.form("survey_form"):
        # Section 1: Demographics
        st.markdown(
            '<p class="section-header">👤 Demographics</p>', unsafe_allow_html=True
        )
        col1, col2 = st.columns(2)
        with col1:
            city = st.text_input("City", value="San Francisco")
            state_or_region = st.text_input("State / Region", value="California")
            country = st.text_input("Country", value="USA")
            postal_code = st.text_input("ZIP / Postal Code (optional)", value="")
            occupation = st.text_input("Occupation", value="Software Engineer")
        with col2:
            age_range = st.selectbox("Age Range", options=AGE_OPTIONS, index=1)
            area_type = st.selectbox(
                "Area Type", options=list(AREA_TYPE_OPTIONS.keys())
            )

        # Section 2: Sleep & Chronotype
        st.markdown(
            '<p class="section-header">😴 Sleep & Chronotype</p>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            wake_time = st.selectbox(
                "Typical Wake Time", options=list(WAKE_TIME_OPTIONS.keys()), index=1
            )
            sleep_time = st.selectbox(
                "Typical Sleep Time", options=list(SLEEP_TIME_OPTIONS.keys()), index=1
            )
        with col2:
            chronotype = st.selectbox(
                "Chronotype", options=list(CHRONOTYPE_OPTIONS.keys()), index=2
            )
            peak_usage = st.selectbox(
                "Peak Usage Time", options=list(PEAK_USAGE_OPTIONS.keys()), index=2
            )

        # Section 3: Daily Routine
        st.markdown(
            '<p class="section-header">📅 Daily Routine</p>', unsafe_allow_html=True
        )
        col1, col2 = st.columns(2)
        with col1:
            routine_level = st.selectbox(
                "Routine Structure", options=list(ROUTINE_OPTIONS.keys()), index=1
            )
            places_visited = st.slider("Typical places visited daily", 1, 6, 3)
            physical_activity = st.selectbox(
                "Physical activity (days/week)",
                options=list(PHYSICAL_ACTIVITY_OPTIONS.keys()),
                index=1,
            )
        with col2:
            commute_days = st.selectbox(
                "Commute Frequency (days/week)",
                options=list(COMMUTE_DAYS_OPTIONS.keys()),
                index=2,
            )
            commute_mode = st.selectbox(
                "Primary Commute Mode",
                options=list(COMMUTE_MODE_OPTIONS.keys()),
                index=0,
            )
            commute_time = st.selectbox(
                "Typical Commute Time (one way)",
                options=list(COMMUTE_TIME_OPTIONS.keys()),
                index=2,
            )

        # Section 4: Phone Usage Habits
        st.markdown(
            '<p class="section-header">📱 Phone Usage Habits</p>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            screen_time = st.selectbox(
                "Estimated Daily Screen Time",
                options=list(SCREEN_TIME_OPTIONS.keys()),
                index=2,
            )
            checking_freq = st.selectbox(
                "Phone Checking Frequency",
                options=list(CHECKING_FREQUENCY_OPTIONS.keys()),
                index=2,
            )
            glance_freq = st.selectbox(
                "Glance Frequency",
                options=list(GLANCE_FREQUENCY_OPTIONS.keys()),
                index=1,
            )
        with col2:
            session_type = st.selectbox(
                "Typical Session Type",
                options=list(SESSION_TYPE_OPTIONS.keys()),
                index=4,
            )
            work_restriction = st.selectbox(
                "Phone Use Restriction at Work",
                options=list(WORK_RESTRICTION_OPTIONS.keys()),
                index=1,
            )
            evening_change = st.selectbox(
                "Evening Usage Change",
                options=list(EVENING_CHANGE_OPTIONS.keys()),
                index=2,
            )

        # Section 5: App Preferences
        st.markdown(
            '<p class="section-header">🔍 App Preferences</p>', unsafe_allow_html=True
        )
        top_apps = st.multiselect(
            "Top App Categories (select 3-5)",
            options=list(APP_CATEGORY_OPTIONS.keys()),
            default=list(APP_CATEGORY_OPTIONS.keys())[:3],
        )
        usage_reasons = st.multiselect(
            "Primary Reasons for Using Phone",
            options=list(USAGE_REASON_OPTIONS.keys()),
            default=list(USAGE_REASON_OPTIONS.keys())[:2],
        )
        exploration = st.selectbox(
            "App Exploration Style", options=list(EXPLORATION_OPTIONS.keys()), index=1
        )

        st.divider()

        # Generation options
        col1, col2 = st.columns(2)
        with col1:
            day_type = st.radio(
                "Generate for which day type?",
                options=["weekday", "weekend"],
                horizontal=True,
            )

        submit = st.form_submit_button("🚀 Generate Synthetic Persona", width="stretch")

        if submit:
            if not city or not occupation:
                st.error("Please fill in city and occupation.")
            elif len(top_apps) < 1:
                st.error("Please select at least one app category.")
            else:
                # Prepare payload
                payload = {
                    "survey": {
                        "age_range": age_range,
                        "city": city,
                        "occupation": occupation,
                        "area_type": AREA_TYPE_OPTIONS[area_type],
                        "wake_time": WAKE_TIME_OPTIONS[wake_time],
                        "sleep_time": SLEEP_TIME_OPTIONS[sleep_time],
                        "chronotype_self_report": CHRONOTYPE_OPTIONS[chronotype],
                        "peak_usage_time": PEAK_USAGE_OPTIONS[peak_usage],
                        "routine_structure": ROUTINE_OPTIONS[routine_level],
                        "places_visited_daily": min(
                            places_visited, 6
                        ),  # backend allows 1..6
                        "commute_days": COMMUTE_DAYS_OPTIONS[commute_days],
                        "commute_mode": COMMUTE_MODE_OPTIONS[commute_mode],
                        "physical_activity_days": PHYSICAL_ACTIVITY_OPTIONS[
                            physical_activity
                        ],
                        "commute_time": COMMUTE_TIME_OPTIONS[commute_time],
                        "daily_screen_time": SCREEN_TIME_OPTIONS[screen_time],
                        "checking_frequency": CHECKING_FREQUENCY_OPTIONS[checking_freq],
                        "session_type": SESSION_TYPE_OPTIONS[session_type],
                        "glance_frequency": GLANCE_FREQUENCY_OPTIONS[glance_freq],
                        "work_phone_restriction": WORK_RESTRICTION_OPTIONS[
                            work_restriction
                        ],
                        "evening_session_change": EVENING_CHANGE_OPTIONS[
                            evening_change
                        ],
                        "evening_activities_increase": [
                            APP_CATEGORY_OPTIONS[app] for app in top_apps
                        ],
                        "usage_reasons": [
                            USAGE_REASON_OPTIONS[r] for r in usage_reasons
                        ],
                        "commute_activities": [],
                        "most_used_categories": [
                            APP_CATEGORY_OPTIONS[app] for app in top_apps
                        ],
                        "exploration_style": EXPLORATION_OPTIONS[exploration],
                        "top_apps": ", ".join(top_apps),
                        "important_habit": None,
                    },
                    "generate_schedule": True,
                    "day_type": day_type,
                }

                with st.spinner("Generating persona... This may take a moment."):
                    result = generate_persona_api(payload)
                    if result:
                        st.session_state.generation_result = normalize_persona_data(
                            result
                        )
                        st.session_state.view_mode = "results"
                        st.query_params["id"] = result.get("persona_id", "")
                        st.success("✅ Persona generated successfully!")
                        st.rerun()


def main():
    render_sidebar()

    if st.session_state.view_mode == "survey":
        render_survey_form()
    else:
        render_results()


if __name__ == "__main__":
    main()
