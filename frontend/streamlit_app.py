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
# HELPER FUNCTIONS
# ============================================================


def get_selection_value(display_to_value: dict, display_key: str) -> str:
    """Convert display text to API value."""
    return display_to_value.get(display_key, display_key)


def multi_select_to_values(display_to_value: dict, selected_displays: list) -> list:
    """Convert list of display texts to API values."""
    return [display_to_value.get(d, d) for d in selected_displays]


# ============================================================
# SURVEY SECTIONS
# ============================================================


def render_section_1_demographics():
    """Section 1: Demographics (Q1-Q4)"""

    st.markdown(
        '<p class="section-header">📋 Section 1: Demographics</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        # Q1: Age
        st.markdown(
            '<p class="question-text">Q1. How old are you?</p>', unsafe_allow_html=True
        )
        age = st.selectbox(
            "Age range",
            options=AGE_OPTIONS,
            key="q1_age",
            label_visibility="collapsed",
        )

        # Q2: City
        st.markdown(
            '<p class="question-text">Q2. What city do you currently live in?</p>',
            unsafe_allow_html=True,
        )
        city = st.text_input(
            "City",
            placeholder="e.g., San Francisco, London, Tokyo",
            key="q2_city",
            label_visibility="collapsed",
        )

    with col2:
        # Q3: Occupation
        st.markdown(
            '<p class="question-text">Q3. What is your current occupation or primary role?</p>',
            unsafe_allow_html=True,
        )
        occupation = st.text_input(
            "Occupation",
            placeholder="e.g., Software Engineer, Student, Nurse",
            key="q3_occupation",
            label_visibility="collapsed",
        )

        # Q4: Area type
        st.markdown(
            '<p class="question-text">Q4. How would you describe the area where you live?</p>',
            unsafe_allow_html=True,
        )
        area_type = st.selectbox(
            "Area type",
            options=list(AREA_TYPE_OPTIONS.keys()),
            key="q4_area",
            label_visibility="collapsed",
        )

    return {
        "age_range": age,
        "city": city,
        "occupation": occupation,
        "area_type": get_selection_value(AREA_TYPE_OPTIONS, area_type),
    }


def render_section_2_sleep_chronotype():
    """Section 2: Sleep & Chronotype (Q5-Q8)"""

    st.markdown(
        '<p class="section-header">🌙 Section 2: Sleep & Chronotype</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        # Q5: Wake time
        st.markdown(
            '<p class="question-text">Q5. On a typical weekday, what time do you usually wake up?</p>',
            unsafe_allow_html=True,
        )
        wake_time = st.selectbox(
            "Wake time",
            options=list(WAKE_TIME_OPTIONS.keys()),
            key="q5_wake",
            label_visibility="collapsed",
        )

        # Q6: Sleep time
        st.markdown(
            '<p class="question-text">Q6. On a typical weekday, what time do you usually go to sleep?</p>',
            unsafe_allow_html=True,
        )
        sleep_time = st.selectbox(
            "Sleep time",
            options=list(SLEEP_TIME_OPTIONS.keys()),
            key="q6_sleep",
            label_visibility="collapsed",
        )

    with col2:
        # Q7: Chronotype self-report
        st.markdown(
            '<p class="question-text">Q7. Would you describe yourself as a "morning person" or an "evening person"?</p>',
            unsafe_allow_html=True,
        )
        chronotype = st.selectbox(
            "Chronotype",
            options=list(CHRONOTYPE_OPTIONS.keys()),
            key="q7_chronotype",
            label_visibility="collapsed",
        )

        # Q8: Peak usage time
        st.markdown(
            '<p class="question-text">Q8. When do you typically use your phone the most?</p>',
            unsafe_allow_html=True,
        )
        peak_usage = st.selectbox(
            "Peak usage",
            options=list(PEAK_USAGE_OPTIONS.keys()),
            key="q8_peak",
            label_visibility="collapsed",
        )

    return {
        "wake_time": get_selection_value(WAKE_TIME_OPTIONS, wake_time),
        "sleep_time": get_selection_value(SLEEP_TIME_OPTIONS, sleep_time),
        "chronotype_self_report": get_selection_value(CHRONOTYPE_OPTIONS, chronotype),
        "peak_usage_time": get_selection_value(PEAK_USAGE_OPTIONS, peak_usage),
    }


def render_section_3_daily_structure():
    """Section 3: Daily Structure (Q9-Q14)"""

    st.markdown(
        '<p class="section-header">📅 Section 3: Daily Structure</p>',
        unsafe_allow_html=True,
    )

    # Q9: Routine structure
    st.markdown(
        '<p class="question-text">Q9. How would you describe your typical daily routine?</p>',
        unsafe_allow_html=True,
    )
    routine = st.selectbox(
        "Routine",
        options=list(ROUTINE_OPTIONS.keys()),
        key="q9_routine",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)

    with col1:
        # Q10: Places visited
        st.markdown(
            '<p class="question-text">Q10. On a typical day, how many different places do you visit?</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="help-text">Include home, work, stores, gym, etc.</p>',
            unsafe_allow_html=True,
        )
        places = st.slider(
            "Places",
            min_value=1,
            max_value=6,
            value=3,
            key="q10_places",
            label_visibility="collapsed",
            help="1 = stay home only, 6+ = many locations",
        )

        # Q11: Commute days
        st.markdown(
            '<p class="question-text">Q11. How many days per week do you commute to a workplace or school?</p>',
            unsafe_allow_html=True,
        )
        commute_days = st.selectbox(
            "Commute days",
            options=list(COMMUTE_DAYS_OPTIONS.keys()),
            key="q11_commute_days",
            label_visibility="collapsed",
        )

        # Q12: Commute mode
        st.markdown(
            '<p class="question-text">Q12. What is your primary mode of transportation for commuting?</p>',
            unsafe_allow_html=True,
        )
        commute_mode = st.selectbox(
            "Commute mode",
            options=list(COMMUTE_MODE_OPTIONS.keys()),
            key="q12_commute_mode",
            label_visibility="collapsed",
        )

    with col2:
        # Q13: Physical activity
        st.markdown(
            '<p class="question-text">Q13. How many days per week do you exercise or do physical activity?</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="help-text">Include gym, sports, running, yoga, etc.</p>',
            unsafe_allow_html=True,
        )
        physical_activity = st.selectbox(
            "Physical activity",
            options=list(PHYSICAL_ACTIVITY_OPTIONS.keys()),
            key="q13_activity",
            label_visibility="collapsed",
        )

        # Q14: Commute time
        st.markdown(
            '<p class="question-text">Q14. How much total time do you typically spend traveling/moving between places each day?</p>',
            unsafe_allow_html=True,
        )
        commute_time = st.selectbox(
            "Commute time",
            options=list(COMMUTE_TIME_OPTIONS.keys()),
            key="q14_commute_time",
            label_visibility="collapsed",
        )

    return {
        "routine_structure": get_selection_value(ROUTINE_OPTIONS, routine),
        "places_visited_daily": places,
        "commute_days": get_selection_value(COMMUTE_DAYS_OPTIONS, commute_days),
        "commute_mode": get_selection_value(COMMUTE_MODE_OPTIONS, commute_mode),
        "physical_activity_days": get_selection_value(
            PHYSICAL_ACTIVITY_OPTIONS, physical_activity
        ),
        "commute_time": get_selection_value(COMMUTE_TIME_OPTIONS, commute_time),
    }


def render_section_4_phone_usage():
    """Section 4: Phone Usage Patterns (Q15-Q20)"""

    st.markdown(
        '<p class="section-header">📱 Section 4: Phone Usage Patterns</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        # Q15: Screen time
        st.markdown(
            '<p class="question-text">Q15. On a typical day, how much total time do you spend on your smartphone?</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="help-text">Check your screen time settings if unsure.</p>',
            unsafe_allow_html=True,
        )
        screen_time = st.selectbox(
            "Screen time",
            options=list(SCREEN_TIME_OPTIONS.keys()),
            key="q15_screen_time",
            label_visibility="collapsed",
        )

        # Q16: Checking frequency
        st.markdown(
            '<p class="question-text">Q16. How often do you check your phone (unlock or glance at notifications)?</p>',
            unsafe_allow_html=True,
        )
        checking_freq = st.selectbox(
            "Checking frequency",
            options=list(CHECKING_FREQUENCY_OPTIONS.keys()),
            key="q16_checking",
            label_visibility="collapsed",
        )

        # Q17: Session type
        st.markdown(
            '<p class="question-text">Q17. Which best describes your typical phone sessions?</p>',
            unsafe_allow_html=True,
        )
        session_type = st.selectbox(
            "Session type",
            options=list(SESSION_TYPE_OPTIONS.keys()),
            key="q17_session",
            label_visibility="collapsed",
        )

    with col2:
        # Q18: Glance frequency
        st.markdown(
            '<p class="question-text">Q18. How often do you quickly check your phone and put it away without unlocking?</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="help-text">e.g., checking time, glancing at notification</p>',
            unsafe_allow_html=True,
        )
        glance_freq = st.selectbox(
            "Glance frequency",
            options=list(GLANCE_FREQUENCY_OPTIONS.keys()),
            key="q18_glance",
            label_visibility="collapsed",
        )

        # Q19: Work restriction
        st.markdown(
            '<p class="question-text">Q19. When you\'re at work or school, how do you typically use your phone?</p>',
            unsafe_allow_html=True,
        )
        work_restriction = st.selectbox(
            "Work restriction",
            options=list(WORK_RESTRICTION_OPTIONS.keys()),
            key="q19_work",
            label_visibility="collapsed",
        )

        # Q20: Evening change
        st.markdown(
            '<p class="question-text">Q20. In the evening at home, are your phone sessions typically...</p>',
            unsafe_allow_html=True,
        )
        evening_change = st.selectbox(
            "Evening change",
            options=list(EVENING_CHANGE_OPTIONS.keys()),
            key="q20_evening",
            label_visibility="collapsed",
        )

    return {
        "daily_screen_time": get_selection_value(SCREEN_TIME_OPTIONS, screen_time),
        "checking_frequency": get_selection_value(
            CHECKING_FREQUENCY_OPTIONS, checking_freq
        ),
        "session_type": get_selection_value(SESSION_TYPE_OPTIONS, session_type),
        "glance_frequency": get_selection_value(GLANCE_FREQUENCY_OPTIONS, glance_freq),
        "work_phone_restriction": get_selection_value(
            WORK_RESTRICTION_OPTIONS, work_restriction
        ),
        "evening_session_change": get_selection_value(
            EVENING_CHANGE_OPTIONS, evening_change
        ),
    }


def render_section_5_app_preferences():
    """Section 5: App & Content Preferences (Q21-Q25)"""

    st.markdown(
        '<p class="section-header">📲 Section 5: App & Content Preferences</p>',
        unsafe_allow_html=True,
    )

    # Q21: Evening activities increase
    st.markdown(
        '<p class="question-text">Q21. Which activities do you do MORE in the evening compared to daytime?</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="help-text">Select up to 3</p>', unsafe_allow_html=True)
    evening_activities = st.multiselect(
        "Evening activities",
        options=list(APP_CATEGORY_OPTIONS.keys()),
        max_selections=3,
        key="q21_evening_activities",
        label_visibility="collapsed",
    )

    # Q22: Usage reasons
    st.markdown(
        '<p class="question-text">Q22. What are your most common reasons for using your phone?</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="help-text">Select up to 3</p>', unsafe_allow_html=True)
    usage_reasons = st.multiselect(
        "Usage reasons",
        options=list(USAGE_REASON_OPTIONS.keys()),
        max_selections=3,
        key="q22_reasons",
        label_visibility="collapsed",
    )

    # Q23: Commute activities
    st.markdown(
        '<p class="question-text">Q23. What do you typically do on your phone during commute or travel?</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="help-text">Select up to 3. Skip if you don\'t commute.</p>',
        unsafe_allow_html=True,
    )
    commute_activities = st.multiselect(
        "Commute activities",
        options=list(APP_CATEGORY_OPTIONS.keys()),
        max_selections=3,
        key="q23_commute",
        label_visibility="collapsed",
    )

    # Q24: Most used categories
    st.markdown(
        '<p class="question-text">Q24. Which app categories do you use most frequently?</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="help-text">Select up to 5</p>', unsafe_allow_html=True)
    most_used = st.multiselect(
        "Most used",
        options=list(APP_CATEGORY_OPTIONS.keys()),
        max_selections=5,
        key="q24_most_used",
        label_visibility="collapsed",
    )

    # Q25: Exploration style
    st.markdown(
        '<p class="question-text">Q25. When it comes to apps and content, which describes you best?</p>',
        unsafe_allow_html=True,
    )
    exploration = st.selectbox(
        "Exploration",
        options=list(EXPLORATION_OPTIONS.keys()),
        key="q25_exploration",
        label_visibility="collapsed",
    )

    return {
        "evening_activities_increase": multi_select_to_values(
            APP_CATEGORY_OPTIONS, evening_activities
        ),
        "usage_reasons": multi_select_to_values(USAGE_REASON_OPTIONS, usage_reasons),
        "commute_activities": multi_select_to_values(
            APP_CATEGORY_OPTIONS, commute_activities
        ),
        "most_used_categories": multi_select_to_values(APP_CATEGORY_OPTIONS, most_used),
        "exploration_style": get_selection_value(EXPLORATION_OPTIONS, exploration),
    }


def render_section_6_optional():
    """Section 6: Optional Details (Q26-Q27)"""

    st.markdown(
        '<p class="section-header">✨ Section 6: Additional Details (Optional)</p>',
        unsafe_allow_html=True,
    )

    # Q26: Top apps
    st.markdown(
        '<p class="question-text">Q26. What are your top 3 most-used apps?</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="help-text">List them in order of usage (most used first)</p>',
        unsafe_allow_html=True,
    )
    top_apps = st.text_input(
        "Top apps",
        placeholder="e.g., Instagram, WhatsApp, YouTube",
        key="q26_top_apps",
        label_visibility="collapsed",
    )

    # Q27: Important habit
    st.markdown(
        '<p class="question-text">Q27. Is there any specific phone habit or pattern that you think is important to capture?</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="help-text">e.g., "I always check Twitter first thing in the morning" or "I avoid my phone during dinner"</p>',
        unsafe_allow_html=True,
    )
    habit = st.text_area(
        "Habit",
        placeholder="Describe any specific habits or patterns...",
        key="q27_habit",
        label_visibility="collapsed",
        height=100,
    )

    return {
        "top_apps": top_apps if top_apps else None,
        "important_habit": habit if habit else None,
    }


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
# MAIN APP
# ============================================================


def main():
    # Header
    st.markdown("# 📱 Synthetic Smartphone Persona Generator")
    # st.markdown(
    #     "Generate realistic, literature-grounded smartphone user personas from survey responses."
    # )
    st.markdown("---")

    # Check if we should show results or survey
    if st.session_state.generation_result is not None:
        render_results_page()
    else:
        render_survey_page()


def render_survey_page():
    """Render the survey input form."""

    st.markdown("### Complete the survey below to generate your persona")
    st.markdown("This survey mirrors our Prolific study questionnaire (27 questions).")

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

            st.markdown("**Q4. Which best describes the are you live in?**")
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
            '<p class="section-header">📅 Section 3: Daily Structure</p>',
            unsafe_allow_html=True,
        )

        st.markdown("**Q9. How would you describe your typical daily routine?**")
        routine = st.selectbox(
            "Routine",
            list(ROUTINE_OPTIONS.keys()),
            key="routine",
            label_visibility="collapsed",
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                "**Q10. How many different places do you usually spend meaningful time in?**"
            )
            st.caption("Include home, work, stores, gym, etc.")
            places = st.slider(
                "Places", 1, 6, 3, key="places", label_visibility="collapsed"
            )

            st.markdown("**Q11. Days per week you commute?**")
            commute_days = st.selectbox(
                "CommuteDays",
                list(COMMUTE_DAYS_OPTIONS.keys()),
                key="commute_days",
                label_visibility="collapsed",
            )

        with col2:
            st.markdown("**Q12. Primary commute mode?**")
            commute_mode = st.selectbox(
                "CommuteMode",
                list(COMMUTE_MODE_OPTIONS.keys()),
                key="commute_mode",
                label_visibility="collapsed",
            )

            st.markdown("**Q13. Days per week you exercise?**")
            physical_activity = st.selectbox(
                "Activity",
                list(PHYSICAL_ACTIVITY_OPTIONS.keys()),
                key="activity",
                label_visibility="collapsed",
            )

        with col3:
            st.markdown("**Q14. Daily travel time?**")
            commute_time = st.selectbox(
                "CommuteTime",
                list(COMMUTE_TIME_OPTIONS.keys()),
                key="commute_time",
                label_visibility="collapsed",
            )

        # ========== SECTION 4: Phone Usage Patterns ==========
        st.markdown(
            '<p class="section-header">📱 Section 4: Phone Usage Patterns</p>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Q15. Total daily phone screen time?**")
            st.caption("Check your phone's screen time settings if unsure")
            screen_time = st.selectbox(
                "ScreenTime",
                list(SCREEN_TIME_OPTIONS.keys()),
                key="screen_time",
                label_visibility="collapsed",
            )

            st.markdown("**Q16. How often do you check your phone?**")
            checking_freq = st.selectbox(
                "Checking",
                list(CHECKING_FREQUENCY_OPTIONS.keys()),
                key="checking",
                label_visibility="collapsed",
            )

            st.markdown("**Q17. Typical phone session length?**")
            session_type = st.selectbox(
                "Session",
                list(SESSION_TYPE_OPTIONS.keys()),
                key="session",
                label_visibility="collapsed",
            )

        with col2:
            st.markdown("**Q18. How often do you glance without unlocking?**")
            st.caption("e.g., checking time, glancing at notification")
            glance_freq = st.selectbox(
                "Glance",
                list(GLANCE_FREQUENCY_OPTIONS.keys()),
                key="glance",
                label_visibility="collapsed",
            )

            st.markdown("**Q19. Phone use at work/school?**")
            work_restriction = st.selectbox(
                "Work",
                list(WORK_RESTRICTION_OPTIONS.keys()),
                key="work",
                label_visibility="collapsed",
            )

            st.markdown("**Q20. Evening sessions compared to daytime?**")
            evening_change = st.selectbox(
                "Evening",
                list(EVENING_CHANGE_OPTIONS.keys()),
                key="evening",
                label_visibility="collapsed",
            )

        # ========== SECTION 5: App Preferences ==========
        st.markdown(
            '<p class="section-header">📲 Section 5: App & Content Preferences</p>',
            unsafe_allow_html=True,
        )

        st.markdown(
            "**Q21. Which activities do you do MORE in the evening?** (select up to 3)"
        )
        evening_activities = st.multiselect(
            "EveningAct",
            list(APP_CATEGORY_OPTIONS.keys()),
            max_selections=3,
            key="evening_act",
            label_visibility="collapsed",
        )

        st.markdown(
            "**Q22. Most common reasons for using your phone?** (select up to 3)"
        )
        usage_reasons = st.multiselect(
            "Reasons",
            list(USAGE_REASON_OPTIONS.keys()),
            max_selections=3,
            key="reasons",
            label_visibility="collapsed",
        )

        st.markdown(
            "**Q23. What do you do on your phone during commute?** (select up to 3)"
        )
        commute_activities = st.multiselect(
            "CommuteAct",
            list(APP_CATEGORY_OPTIONS.keys()),
            max_selections=3,
            key="commute_act",
            label_visibility="collapsed",
        )

        st.markdown("**Q24. Most frequently used app categories?** (select up to 5)")
        most_used = st.multiselect(
            "MostUsed",
            list(APP_CATEGORY_OPTIONS.keys()),
            max_selections=5,
            key="most_used",
            label_visibility="collapsed",
        )

        st.markdown("**Q25. Do you explore new apps or stick to familiar ones?**")
        exploration = st.selectbox(
            "Explore",
            list(EXPLORATION_OPTIONS.keys()),
            key="explore",
            label_visibility="collapsed",
        )

        # ========== SECTION 6: Optional ==========
        st.markdown(
            '<p class="section-header">✨ Section 6: Additional Details (Optional)</p>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Q26. What are your top 3 most-used apps?**")
            top_apps = st.text_input(
                "TopApps",
                placeholder="e.g., Instagram, WhatsApp, YouTube, Tiktok",
                key="top_apps",
                label_visibility="collapsed",
            )

        with col2:
            st.markdown("**Q27. Any specific phone habits to capture?**")
            important_habit = st.text_area(
                "Habit",
                placeholder="e.g., 'I always check Twitter first thing in the morning'",
                key="habit",
                height=80,
                label_visibility="collapsed",
            )

        # ========== GENERATION OPTIONS ==========
        st.markdown("---")
        st.markdown("### Generation Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            generate_schedule = st.checkbox(
                "Generate 24-hour schedule", value=True, key="gen_schedule"
            )

        with col2:
            day_type = st.radio(
                "Day type", ["weekday", "weekend"], key="day_type", horizontal=True
            )

        with col3:
            pass  # Placeholder for future options

        # ========== SUBMIT ==========
        st.markdown("---")
        submitted = st.form_submit_button(
            "Generate Persona", use_container_width=True, type="primary"
        )

        if submitted:
            # Validate required fields
            if not city or not occupation:
                st.error("Please fill in city and occupation.")
                return

            if len(most_used) == 0:
                st.error("Please select at least one app category in Q24.")
                return

            # Build payload
            payload = {
                "survey": {
                    # Section 1
                    "age_range": age_range,
                    "city": city,
                    "occupation": occupation,
                    "area_type": get_selection_value(AREA_TYPE_OPTIONS, area_type),
                    # Section 2
                    "wake_time": get_selection_value(WAKE_TIME_OPTIONS, wake_time),
                    "sleep_time": get_selection_value(SLEEP_TIME_OPTIONS, sleep_time),
                    "chronotype_self_report": get_selection_value(
                        CHRONOTYPE_OPTIONS, chronotype
                    ),
                    "peak_usage_time": get_selection_value(
                        PEAK_USAGE_OPTIONS, peak_usage
                    ),
                    # Section 3
                    "routine_structure": get_selection_value(ROUTINE_OPTIONS, routine),
                    "places_visited_daily": places,
                    "commute_days": get_selection_value(
                        COMMUTE_DAYS_OPTIONS, commute_days
                    ),
                    "commute_mode": get_selection_value(
                        COMMUTE_MODE_OPTIONS, commute_mode
                    ),
                    "physical_activity_days": get_selection_value(
                        PHYSICAL_ACTIVITY_OPTIONS, physical_activity
                    ),
                    "commute_time": get_selection_value(
                        COMMUTE_TIME_OPTIONS, commute_time
                    ),
                    # Section 4
                    "daily_screen_time": get_selection_value(
                        SCREEN_TIME_OPTIONS, screen_time
                    ),
                    "checking_frequency": get_selection_value(
                        CHECKING_FREQUENCY_OPTIONS, checking_freq
                    ),
                    "session_type": get_selection_value(
                        SESSION_TYPE_OPTIONS, session_type
                    ),
                    "glance_frequency": get_selection_value(
                        GLANCE_FREQUENCY_OPTIONS, glance_freq
                    ),
                    "work_phone_restriction": get_selection_value(
                        WORK_RESTRICTION_OPTIONS, work_restriction
                    ),
                    "evening_session_change": get_selection_value(
                        EVENING_CHANGE_OPTIONS, evening_change
                    ),
                    # Section 5
                    "evening_activities_increase": multi_select_to_values(
                        APP_CATEGORY_OPTIONS, evening_activities
                    ),
                    "usage_reasons": multi_select_to_values(
                        USAGE_REASON_OPTIONS, usage_reasons
                    ),
                    "commute_activities": multi_select_to_values(
                        APP_CATEGORY_OPTIONS, commute_activities
                    ),
                    "most_used_categories": multi_select_to_values(
                        APP_CATEGORY_OPTIONS, most_used
                    ),
                    "exploration_style": get_selection_value(
                        EXPLORATION_OPTIONS, exploration
                    ),
                    # Section 6 (Optional)
                    "top_apps": top_apps if top_apps else None,
                    "important_habit": important_habit if important_habit else None,
                },
                "generate_schedule": generate_schedule,
                "day_type": day_type,
            }

            # Call the API
            with st.spinner("🔮 Generating persona... This may take a moment."):
                try:
                    response = requests.post(
                        f"{API_URL}/generate",
                        json=payload,
                        timeout=120,
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.generation_result = result
                        st.rerun()
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ Could not connect to the API server. Make sure `server.py` is running on port 8000."
                    )
                except requests.exceptions.Timeout:
                    st.error("❌ Request timed out. The server may be overloaded.")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")


def render_results_page():
    """Render the generated persona results."""

    result = st.session_state.generation_result

    # Back button
    if st.button("← Back to Survey", type="secondary"):
        st.session_state.generation_result = None
        st.rerun()

    st.markdown("---")
    st.markdown("## 🎭 Generated Persona")

    # Extract data
    persona_id = result.get("persona_id", "unknown")
    dimensions = result.get("dimensions", {})
    parameters = result.get("parameters", {})
    schedule = result.get("schedule")

    # ========== HEADER INFO ==========
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Persona ID", persona_id[:8] + "..." if len(persona_id) > 8 else persona_id
        )

    with col2:
        st.metric(
            "Waking Hours",
            f"{parameters.get('waking_hour_start', 7)}:00 - {parameters.get('waking_hour_end', 23)}:00",
        )

    with col3:
        screen_min = parameters.get("daily_screen_time_minutes", 180)
        st.metric("Est. Screen Time", f"{screen_min // 60}h {screen_min % 60}m")

    st.markdown("---")

    # ========== TABS FOR DIFFERENT VIEWS ==========
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Dimensions", "⚙️ Parameters", "📅 Schedule", "📥 Export"]
    )

    # ---------- TAB 1: DIMENSIONS ----------
    with tab1:
        st.markdown("### Behavioral Dimensions")
        st.markdown(
            "These 8 dimensions characterize the user's behavioral profile (0-1 scale)."
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = render_dimensions_radar(dimensions)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Dimension Values:**")

            dimension_labels = {
                "chronotype_score": ("🌙 Chronotype", "0=morning, 1=evening"),
                "usage_intensity": ("📱 Usage Intensity", "phone dependency"),
                "attentional_granularity": (
                    "🔍 Attentional Granularity",
                    "short vs long sessions",
                ),
                "contextual_sensitivity": (
                    "🎯 Contextual Sensitivity",
                    "adapts to context",
                ),
                "social_orientation": ("👥 Social Orientation", "social vs solo use"),
                "mobility_diversity": ("🚗 Mobility Diversity", "travel patterns"),
                "routine_stability": ("📋 Routine Stability", "consistent schedule"),
                "novelty_seeking": ("✨ Novelty Seeking", "explores new content"),
            }

            for key, (label, desc) in dimension_labels.items():
                val = dimensions.get(key, 0.5)
                st.markdown(f"**{label}**: `{val:.2f}`")
                st.caption(desc)

    # ---------- TAB 2: PARAMETERS ----------
    with tab2:
        st.markdown("### Simulation Parameters")
        st.markdown("These parameters can be used to drive the smartphone simulation.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ⏰ Temporal Parameters")

            temporal_params = {
                "waking_hour_start": "Wake Hour",
                "waking_hour_end": "Sleep Hour",
                "temporal_peak_hour": "Peak Usage Hour",
                "late_night_probability": "Late Night Probability",
            }

            for key, label in temporal_params.items():
                val = parameters.get(key, "N/A")
                if isinstance(val, float):
                    st.markdown(f"- **{label}**: `{val:.2f}`")
                else:
                    st.markdown(f"- **{label}**: `{val}`")

            st.markdown("#### 📊 Usage Parameters")

            usage_params = {
                "daily_screen_time_minutes": "Daily Screen Time (min)",
                "sessions_per_day": "Sessions per Day",
                "avg_session_duration_seconds": "Avg Session (sec)",
                "glance_ratio": "Glance Ratio",
            }

            for key, label in usage_params.items():
                val = parameters.get(key, "N/A")
                if isinstance(val, float):
                    st.markdown(f"- **{label}**: `{val:.1f}`")
                else:
                    st.markdown(f"- **{label}**: `{val}`")

        with col2:
            st.markdown("#### 📱 App Weights")
            fig = render_app_weights_pie(parameters)
            st.plotly_chart(fig, use_container_width=True)

        # Hourly pattern
        st.markdown("#### 📈 Predicted Hourly Usage Pattern")
        fig = render_hourly_usage_pattern(parameters)
        st.plotly_chart(fig, use_container_width=True)

    # ---------- TAB 3: SCHEDULE ----------
    with tab3:
        st.markdown("### 24-Hour Schedule")

        if schedule and "segments" in schedule:
            st.markdown(f"**Day Type**: {schedule.get('day_type', 'weekday').title()}")

            # Timeline visualization
            fig = render_schedule_timeline(schedule)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Detail table
            st.markdown("#### Segment Details")

            df = render_schedule_detail_table(schedule)
            if not df.empty:
                st.dataframe(df, use_container_width=True, hide_index=True)

            # Summary stats
            total_sessions = sum(
                seg.get("expected_sessions", 0) for seg in schedule["segments"]
            )
            st.markdown(f"**Total Expected Sessions**: {total_sessions}")
        else:
            st.info(
                "No schedule was generated. Check the 'Generate 24-hour schedule' option and try again."
            )

    # ---------- TAB 4: EXPORT ----------
    with tab4:
        st.markdown("### Export Options")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📄 JSON Export")
            st.markdown("Download the complete persona data as JSON.")

            json_str = json.dumps(result, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"persona_{persona_id}.json",
                mime="application/json",
            )

        with col2:
            st.markdown("#### 📋 Copy to Clipboard")
            st.markdown("Copy the persona ID or full JSON.")

            st.code(persona_id, language=None)
            st.code(
                json_str[:500] + "..." if len(json_str) > 500 else json_str,
                language="json",
            )

        # Raw JSON viewer
        st.markdown("#### 🔍 Raw JSON Preview")
        with st.expander("View Full JSON"):
            st.json(result)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
