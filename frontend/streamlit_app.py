# frontend/streamlit_app.py

import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime, date
from typing import List, Optional

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


def fetch_recent_personas(limit: int = 10) -> List[dict]:
    """Fetch list of recent personas."""
    try:
        response = requests.get(
            f"{API_URL}/personas", params={"limit": limit}, timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("personas", [])
        return []
    except:
        return []


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
    The /persona/{id} endpoint returns data nested under 'survey', 'dimensions', etc.
    The /generate endpoint returns it directly.
    """
    # If data has nested structure from /persona/{id}
    if "survey" in data and "dimensions" in data and "parameters" in data:
        # Already in the right format
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
    # Already normalized (from /generate)
    return data


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

    # Check URL query params first
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
        load_clicked = st.button("Load", use_container_width=True, key="load_btn")
    with col2:
        clear_clicked = st.button("Clear", use_container_width=True, key="clear_btn")

    # Handle load
    if load_clicked and persona_id_input:
        with st.sidebar.status("Loading...", expanded=True):
            data = fetch_persona_by_id(persona_id_input.strip())
            if data:
                st.session_state.generation_result = normalize_persona_data(data)
                st.session_state.view_mode = "results"
                # Update URL
                st.query_params["id"] = persona_id_input.strip()
                st.rerun()
            else:
                st.sidebar.error(f"Persona not found: {persona_id_input[:20]}...")

    # Handle clear
    if clear_clicked:
        st.session_state.generation_result = None
        st.session_state.view_mode = "survey"
        st.query_params.clear()
        st.rerun()

    # Auto-load from URL on first visit
    if url_persona_id and st.session_state.generation_result is None:
        data = fetch_persona_by_id(url_persona_id.strip())
        if data:
            st.session_state.generation_result = normalize_persona_data(data)
            st.session_state.view_mode = "results"

    st.sidebar.divider()

    # ========== RECENT PERSONAS ==========
    st.sidebar.subheader("📋 Recent Personas")

    recent = fetch_recent_personas(limit=8)

    if recent:
        for p in recent:
            pid = p.get("persona_id", "")
            city = p.get("city", "Unknown")
            occupation = p.get("occupation", "")
            label = f"{city}"
            if occupation:
                label += f" - {occupation[:15]}"

            if st.sidebar.button(
                f"📄 {label}",
                key=f"recent_{pid}",
                use_container_width=True,
                help=f"ID: {pid}",
            ):
                st.query_params["id"] = pid
                st.rerun()
    else:
        st.sidebar.caption("No personas generated yet")

    st.sidebar.divider()

    # ========== NAVIGATION ==========
    st.sidebar.subheader("🧭 Navigation")

    if st.sidebar.button(
        "📝 New Survey", use_container_width=True, type="secondary", key="nav_survey"
    ):
        st.session_state.view_mode = "survey"
        st.session_state.generation_result = None
        st.query_params.clear()
        st.rerun()

    if st.session_state.generation_result:
        if st.sidebar.button(
            "📊 View Results",
            use_container_width=True,
            type="primary",
            key="nav_results",
        ):
            st.session_state.view_mode = "results"
            st.rerun()


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================


def render_dimensions_radar(dimensions: dict) -> go.Figure:
    """Radar chart of behavioral dimensions."""

    labels = [
        "Chronotype<br>(evening →)",
        "Usage<br>Intensity",
        "Attentional<br>Granularity",
        "Contextual<br>Sensitivity",
        "Social<br>Orientation",
        "Mobility<br>Diversity",
        "Routine<br>Stability",
        "Novelty<br>Seeking",
    ]

    keys = [
        "chronotype_score",
        "usage_intensity",
        "attentional_granularity",
        "contextual_sensitivity",
        "social_orientation",
        "mobility_diversity",
        "routine_stability",
        "novelty_seeking",
    ]

    values = [dimensions.get(k, 0.5) for k in keys]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.3)",
            line=dict(color="rgb(31, 119, 180)", width=2),
            name="Profile",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                ticktext=["0.25", "0.5", "0.75", "1.0"],
            ),
        ),
        showlegend=False,
        height=450,
        margin=dict(t=30, b=30, l=80, r=80),
    )

    return fig


def render_app_weights_pie(parameters: dict) -> go.Figure:
    """Pie chart of app category weights."""

    categories = [
        "Social",
        "Messaging",
        "Video",
        "Music",
        "Navigation",
        "Productivity",
        "News",
        "Games",
        "Shopping",
    ]
    keys = [
        "weight_social",
        "weight_messaging",
        "weight_video",
        "weight_music",
        "weight_navigation",
        "weight_productivity",
        "weight_news",
        "weight_games",
        "weight_shopping",
    ]

    values = [parameters.get(k, 0) for k in keys]

    # Filter out zeros
    filtered = [(c, v) for c, v in zip(categories, values) if v > 0.01]
    if not filtered:
        filtered = [("No data", 1)]

    cats, vals = zip(*filtered)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=cats,
                values=vals,
                hole=0.4,
                marker=dict(colors=px.colors.qualitative.Set2),
            )
        ]
    )

    fig.update_layout(
        title="App Category Distribution",
        height=350,
        margin=dict(t=50, b=20, l=20, r=20),
    )

    return fig


def render_hourly_usage_pattern(parameters: dict) -> go.Figure:
    """Bar chart of predicted hourly usage."""

    waking_start = parameters.get("waking_hour_start", 7)
    waking_end = parameters.get("waking_hour_end", 23)
    peak_hour = parameters.get("temporal_peak_hour", 20)
    late_night_prob = parameters.get("late_night_probability", 0.1)

    hours = list(range(24))
    activity = []

    for h in hours:
        # Base: gaussian around peak hour
        distance = min(abs(h - peak_hour), 24 - abs(h - peak_hour))
        base = max(0, 1 - (distance / 8) ** 2)

        # Suppress during sleep
        if waking_end >= waking_start:
            # Normal schedule (e.g., 7am - 11pm)
            if h < waking_start or h > waking_end:
                base *= 0.05
        else:
            # Wraps around midnight (e.g., 10am - 2am)
            if h < waking_start and h > waking_end:
                base *= 0.05

        # Late night boost
        if h >= 23 or h <= 2:
            base = max(base, late_night_prob * 0.7)

        activity.append(base)

    # Normalize to 0-1
    max_val = max(activity) if max(activity) > 0 else 1
    activity = [a / max_val for a in activity]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=hours,
            y=activity,
            marker_color=[
                (
                    "#1f77b4"
                    if waking_start <= h <= waking_end
                    or (
                        waking_end < waking_start
                        and (h >= waking_start or h <= waking_end)
                    )
                    else "#95a5a6"
                )
                for h in hours
            ],
            hovertemplate="Hour %{x}:00<br>Activity: %{y:.2f}<extra></extra>",
        )
    )

    # Mark peak hour
    fig.add_vline(
        x=peak_hour, line_dash="dash", line_color="red", annotation_text="Peak"
    )

    fig.update_layout(
        title="Predicted Hourly Phone Usage Pattern",
        xaxis_title="Hour of Day",
        yaxis_title="Relative Activity",
        xaxis=dict(tickmode="linear", tick0=0, dtick=2),
        yaxis=dict(range=[0, 1.1]),
        height=300,
        margin=dict(t=50, b=50, l=50, r=30),
    )

    return fig


def render_schedule_timeline(schedule: dict) -> Optional[go.Figure]:
    """Gantt-style timeline of daily schedule."""

    if not schedule or "segments" not in schedule:
        return None

    segments = schedule["segments"]

    # Activity colors
    colors = {
        "sleeping": "#2C3E50",
        "waking_up": "#F39C12",
        "morning_routine": "#E74C3C",
        "commuting": "#9B59B6",
        "working": "#3498DB",
        "lunch_break": "#2ECC71",
        "exercising": "#1ABC9C",
        "errands": "#E67E22",
        "home_evening": "#F1C40F",
        "leisure": "#FF6B6B",
        "winding_down": "#95A5A6",
    }

    fig = go.Figure()

    for i, seg in enumerate(segments):
        activity = seg.get("activity", "unknown")
        start_str = seg.get("start_time", "00:00")
        end_str = seg.get("end_time", "00:00")
        context = seg.get("context", "unknown")
        sessions = seg.get("expected_sessions", 0)

        # Parse time strings
        start_parts = start_str.split(":")
        end_parts = end_str.split(":")
        start_h = int(start_parts[0]) + int(start_parts[1]) / 60
        end_h = int(end_parts[0]) + int(end_parts[1]) / 60

        # Handle overnight
        if end_h < start_h:
            end_h += 24

        color = colors.get(activity, "#BDC3C7")

        fig.add_trace(
            go.Bar(
                x=[end_h - start_h],
                y=[0],
                base=[start_h],
                orientation="h",
                marker=dict(color=color, line=dict(color="white", width=1)),
                name=activity,
                text=f"{activity}<br>{sessions} sessions",
                textposition="inside",
                hovertemplate=f"<b>{activity}</b><br>Time: {start_str} - {end_str}<br>Context: {context}<br>Expected sessions: {sessions}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Daily Schedule Timeline",
        xaxis=dict(
            title="Hour of Day",
            range=[0, 24],
            tickmode="linear",
            tick0=0,
            dtick=2,
            ticktext=[f"{h}:00" for h in range(0, 25, 2)],
            tickvals=list(range(0, 25, 2)),
        ),
        yaxis=dict(visible=False),
        height=150,
        margin=dict(t=50, b=50, l=30, r=30),
        barmode="overlay",
    )

    return fig


def render_schedule_detail_table(schedule: dict) -> pd.DataFrame:
    """Table view of schedule segments."""

    if not schedule or "segments" not in schedule:
        return pd.DataFrame()

    rows = []
    for seg in schedule["segments"]:
        rows.append(
            {
                "Time": f"{seg.get('start_time', '')} - {seg.get('end_time', '')}",
                "Activity": seg.get("activity", "").replace("_", " ").title(),
                "Context": seg.get("context", "").replace("_", " ").title(),
                "Location": seg.get("location_label", ""),
                "Phone Access": "✅" if seg.get("phone_accessible", True) else "❌",
                "Expected Sessions": seg.get("expected_sessions", 0),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SURVEY PAGE
# ============================================================


def render_survey_page():
    """Render the survey input form."""

    st.markdown("### Complete the survey below to generate your persona")

    # Progress indicator
    st.progress(0, text="Fill out all sections below")

    with st.form("persona_survey", clear_on_submit=False):

        # ========== SECTION 1: Demographics ==========
        st.markdown(
            '<p class="section-header">📋 Section 1: Demographics</p>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Q1. How old are you?**")
            age_range = st.selectbox(
                "Age", AGE_OPTIONS, key="age", label_visibility="collapsed"
            )

            st.markdown("**Q2. What city do you currently live in?**")
            city = st.text_input(
                "City",
                placeholder="e.g., San Francisco",
                key="city",
                label_visibility="collapsed",
            )

        with col2:
            st.markdown("**Q3. What is your current occupation or primary role?**")
            occupation = st.text_input(
                "Occupation",
                placeholder="e.g., Software Engineer",
                key="occupation",
                label_visibility="collapsed",
            )

            st.markdown("**Q4. Which best describes the area you live in?**")
            area_type = st.selectbox(
                "Area",
                list(AREA_TYPE_OPTIONS.keys()),
                key="area",
                label_visibility="collapsed",
            )

        # ========== SECTION 2: Sleep & Chronotype ==========
        st.markdown(
            '<p class="section-header">🌙 Section 2: Sleep & Chronotype</p>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Q5. What time do you usually wake up on weekdays?**")
            wake_time = st.selectbox(
                "Wake",
                list(WAKE_TIME_OPTIONS.keys()),
                key="wake",
                label_visibility="collapsed",
            )

            st.markdown("**Q6. What time do you usually go to sleep on weekdays?**")
            sleep_time = st.selectbox(
                "Sleep",
                list(SLEEP_TIME_OPTIONS.keys()),
                key="sleep",
                label_visibility="collapsed",
            )

        with col2:
            st.markdown("**Q7. Are you a morning person or evening person?**")
            chronotype = st.selectbox(
                "Chronotype",
                list(CHRONOTYPE_OPTIONS.keys()),
                key="chronotype",
                label_visibility="collapsed",
            )

            st.markdown("**Q8. When do you use your phone the most?**")
            peak_usage = st.selectbox(
                "Peak",
                list(PEAK_USAGE_OPTIONS.keys()),
                key="peak",
                label_visibility="collapsed",
            )

        # ========== SECTION 3: Daily Structure ==========
        st.markdown(
            '<p class="section-header">🏠 Section 3: Daily Structure</p>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Q9. How structured is your typical weekday?**")
            routine = st.selectbox(
                "Routine",
                list(ROUTINE_OPTIONS.keys()),
                key="routine",
                label_visibility="collapsed",
            )

            st.markdown(
                "**Q10. How many days per week do you commute to work/school?**"
            )
            commute_days = st.selectbox(
                "Commute Days",
                list(COMMUTE_DAYS_OPTIONS.keys()),
                key="commute_days",
                label_visibility="collapsed",
            )

            st.markdown("**Q11. What is your primary mode of commuting?**")
            commute_mode = st.selectbox(
                "Commute Mode",
                list(COMMUTE_MODE_OPTIONS.keys()),
                key="commute_mode",
                label_visibility="collapsed",
            )

        with col2:
            st.markdown("**Q12. How much time do you spend commuting daily (total)?**")
            commute_time = st.selectbox(
                "Commute Time",
                list(COMMUTE_TIME_OPTIONS.keys()),
                key="commute_time",
                label_visibility="collapsed",
            )

            st.markdown("**Q13. How many days per week do you do physical exercise?**")
            physical_activity = st.selectbox(
                "Physical Activity",
                list(PHYSICAL_ACTIVITY_OPTIONS.keys()),
                key="physical_activity",
                label_visibility="collapsed",
            )

        # ========== SECTION 4: Phone Usage Patterns ==========
        st.markdown(
            '<p class="section-header">📱 Section 4: Phone Usage Patterns</p>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Q14. How much time do you spend on your phone daily?**")
            screen_time = st.selectbox(
                "Screen Time",
                list(SCREEN_TIME_OPTIONS.keys()),
                key="screen_time",
                label_visibility="collapsed",
            )

            st.markdown("**Q15. How often do you check your phone?**")
            checking_frequency = st.selectbox(
                "Checking Frequency",
                list(CHECKING_FREQUENCY_OPTIONS.keys()),
                key="checking_freq",
                label_visibility="collapsed",
            )

            st.markdown("**Q16. What best describes your typical phone sessions?**")
            session_type = st.selectbox(
                "Session Type",
                list(SESSION_TYPE_OPTIONS.keys()),
                key="session_type",
                label_visibility="collapsed",
            )

        with col2:
            st.markdown(
                "**Q17. How often do you glance at your phone without unlocking?**"
            )
            glance_frequency = st.selectbox(
                "Glance Frequency",
                list(GLANCE_FREQUENCY_OPTIONS.keys()),
                key="glance_freq",
                label_visibility="collapsed",
            )

            st.markdown("**Q18. How restricted is phone use at your workplace?**")
            work_restriction = st.selectbox(
                "Work Restriction",
                list(WORK_RESTRICTION_OPTIONS.keys()),
                key="work_restriction",
                label_visibility="collapsed",
            )

            st.markdown(
                "**Q19. How do your evening phone sessions compare to daytime?**"
            )
            evening_change = st.selectbox(
                "Evening Change",
                list(EVENING_CHANGE_OPTIONS.keys()),
                key="evening_change",
                label_visibility="collapsed",
            )

        # ========== SECTION 5: App Preferences ==========
        st.markdown(
            '<p class="section-header">📲 Section 5: App Preferences</p>',
            unsafe_allow_html=True,
        )

        st.markdown(
            "**Q20. Which types of apps do you use most frequently?** (Select up to 5)"
        )
        top_apps = st.multiselect(
            "Top Apps",
            list(APP_CATEGORY_OPTIONS.keys()),
            max_selections=5,
            key="top_apps",
            label_visibility="collapsed",
        )

        st.markdown(
            "**Q21. What are your main reasons for using your phone?** (Select up to 5)"
        )
        usage_reasons = st.multiselect(
            "Usage Reasons",
            list(USAGE_REASON_OPTIONS.keys()),
            max_selections=5,
            key="usage_reasons",
            label_visibility="collapsed",
        )

        st.markdown(
            "**Q22. Do you prefer exploring new content or sticking to what you know?**"
        )
        exploration = st.selectbox(
            "Exploration",
            list(EXPLORATION_OPTIONS.keys()),
            key="exploration",
            label_visibility="collapsed",
        )

        # ========== SUBMIT ==========
        st.markdown("---")

        submitted = st.form_submit_button(
            "🚀 Generate Persona", use_container_width=True, type="primary"
        )

        if submitted:
            # Validate required fields
            if not city:
                st.error("Please enter your city.")
                return
            if not occupation:
                st.error("Please enter your occupation.")
                return
            if len(top_apps) == 0:
                st.error("Please select at least one app category.")
                return
            if len(usage_reasons) == 0:
                st.error("Please select at least one usage reason.")
                return

            # Build payload
            payload = {
                "age_range": age_range,
                "city": city,
                "occupation": occupation,
                "area_type": get_selection_value(AREA_TYPE_OPTIONS, area_type),
                "wake_time": get_selection_value(WAKE_TIME_OPTIONS, wake_time),
                "sleep_time": get_selection_value(SLEEP_TIME_OPTIONS, sleep_time),
                "chronotype": get_selection_value(CHRONOTYPE_OPTIONS, chronotype),
                "peak_usage_time": get_selection_value(PEAK_USAGE_OPTIONS, peak_usage),
                "routine_level": get_selection_value(ROUTINE_OPTIONS, routine),
                "commute_days": get_selection_value(COMMUTE_DAYS_OPTIONS, commute_days),
                "commute_mode": get_selection_value(COMMUTE_MODE_OPTIONS, commute_mode),
                "commute_time": get_selection_value(COMMUTE_TIME_OPTIONS, commute_time),
                "physical_activity_days": get_selection_value(
                    PHYSICAL_ACTIVITY_OPTIONS, physical_activity
                ),
                "screen_time": get_selection_value(SCREEN_TIME_OPTIONS, screen_time),
                "checking_frequency": get_selection_value(
                    CHECKING_FREQUENCY_OPTIONS, checking_frequency
                ),
                "session_type": get_selection_value(SESSION_TYPE_OPTIONS, session_type),
                "glance_frequency": get_selection_value(
                    GLANCE_FREQUENCY_OPTIONS, glance_frequency
                ),
                "work_phone_restriction": get_selection_value(
                    WORK_RESTRICTION_OPTIONS, work_restriction
                ),
                "evening_usage_change": get_selection_value(
                    EVENING_CHANGE_OPTIONS, evening_change
                ),
                "top_app_categories": multi_select_to_values(
                    APP_CATEGORY_OPTIONS, top_apps
                ),
                "usage_reasons": multi_select_to_values(
                    USAGE_REASON_OPTIONS, usage_reasons
                ),
                "exploration_preference": get_selection_value(
                    EXPLORATION_OPTIONS, exploration
                ),
            }

            # Call API
            with st.spinner("🔄 Generating persona... This may take 30-60 seconds."):
                result = generate_persona_api(payload)

            if result:
                st.session_state.generation_result = result
                st.session_state.view_mode = "results"
                # Update URL with persona ID
                if "persona_id" in result:
                    st.query_params["id"] = result["persona_id"]
                st.success("✅ Persona generated successfully!")
                st.rerun()


# ============================================================
# RESULTS PAGE
# ============================================================


def render_results_page():
    """Render the results visualization page."""

    result = st.session_state.generation_result

    if not result:
        st.warning("No persona data available. Please generate or load a persona.")
        if st.button("Go to Survey"):
            st.session_state.view_mode = "survey"
            st.rerun()
        return

    # Extract data
    persona_id = result.get("persona_id", "unknown")
    dimensions = result.get("dimensions", {})
    parameters = result.get("parameters", {})
    schedule = result.get("schedule")
    survey_summary = result.get("survey_summary", {})

    # ========== HEADER ==========
    st.markdown(f"## 📊 Persona Results")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.markdown(f"**ID:** `{persona_id}`")
    with col2:
        city = survey_summary.get("city", parameters.get("city", "N/A"))
        occupation = survey_summary.get(
            "occupation", parameters.get("occupation", "N/A")
        )
        st.markdown(f"**{city}** • {occupation}")
    with col3:
        if st.button("📋 Copy ID"):
            st.write(f"```{persona_id}```")

    st.divider()

    # ========== TABS ==========
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 Dimensions", "⚙️ Parameters", "📅 Schedule", "📄 Raw JSON"]
    )

    # ----- TAB 1: Dimensions -----
    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Behavioral Profile")
            fig = render_dimensions_radar(dimensions)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Dimension Values")
            dim_df = pd.DataFrame(
                [
                    {
                        "Dimension": k.replace("_", " ").title(),
                        "Value": f"{v:.2f}",
                        "Bar": "█" * int(v * 20) + "░" * (20 - int(v * 20)),
                    }
                    for k, v in dimensions.items()
                ]
            )
            st.dataframe(dim_df, hide_index=True, use_container_width=True)

    # ----- TAB 2: Parameters -----
    with tab2:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("App Category Weights")
            fig = render_app_weights_pie(parameters)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Timing Parameters")
            timing_data = {
                "Wake Hour": parameters.get("waking_hour_start", "N/A"),
                "Sleep Hour": parameters.get("waking_hour_end", "N/A"),
                "Peak Hour": parameters.get("temporal_peak_hour", "N/A"),
                "Sessions/Day": parameters.get("sessions_per_day", "N/A"),
                "Avg Duration (s)": parameters.get(
                    "avg_session_duration_seconds", "N/A"
                ),
                "Late Night Prob": f"{parameters.get('late_night_probability', 0):.2f}",
            }
            for k, v in timing_data.items():
                st.metric(k, v)

        st.subheader("Hourly Usage Pattern")
        fig = render_hourly_usage_pattern(parameters)
        st.plotly_chart(fig, use_container_width=True)

    # ----- TAB 3: Schedule -----
    with tab3:
        if schedule:
            st.subheader("Daily Timeline")
            fig = render_schedule_timeline(schedule)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Schedule Details")
            df = render_schedule_detail_table(schedule)
            if not df.empty:
                st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("No schedule data available for this persona.")

    # ----- TAB 4: Raw JSON -----
    with tab4:
        st.subheader("Full Response Data")
        st.json(result)

        # Download button
        json_str = json.dumps(result, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"persona_{persona_id}.json",
            mime="application/json",
        )


# ============================================================
# MAIN
# ============================================================


def main():
    """Main app entry point."""

    # Render sidebar (handles navigation and loading)
    render_sidebar()

    # Main content area
    st.markdown(
        '<h1 class="main-header">📱 Synthetic Persona Generator</h1>',
        unsafe_allow_html=True,
    )

    # Route based on view mode
    if st.session_state.view_mode == "results" and st.session_state.generation_result:
        render_results_page()
    else:
        render_survey_page()


if __name__ == "__main__":
    main()
