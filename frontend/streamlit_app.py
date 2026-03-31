# streamlit_app.py

import streamlit as st
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Synthetic Persona Generator",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Synthetic Smartphone Persona Generator")
st.markdown("Generate realistic, literature-grounded smartphone user personas.")

# --- Sidebar: Survey Input ---
st.sidebar.header("📋 Persona Configuration")

with st.sidebar.form("persona_form"):
    st.subheader("Demographics")
    age = st.slider("Age", 18, 80, 30)
    city = st.text_input("City", "San Francisco")
    job = st.text_input("Occupation", "Software Engineer")

    st.subheader("Sleep & Chronotype")
    chronotype = st.select_slider(
        "When do you naturally prefer to wake up and be active?",
        options=["morning", "neutral", "night"],
        value="neutral",
    )

    st.subheader("Phone Usage")
    usage_level = st.select_slider(
        "Overall phone usage level",
        options=["light", "typical", "heavy"],
        value="typical",
    )

    phone_style = st.select_slider(
        "Typical phone interaction style",
        options=["quick_checks", "mixed", "long_sessions"],
        value="mixed",
    )

    primary_use = st.select_slider(
        "Primary phone use",
        options=["social", "mixed", "video_news"],
        value="mixed",
    )

    st.subheader("Lifestyle")
    activity_level = st.select_slider(
        "Physical activity level",
        options=["low", "moderate", "high"],
        value="moderate",
    )

    commute_frequency = st.select_slider(
        "How often do you commute?",
        options=["rare", "sometimes", "frequent"],
        value="sometimes",
    )

    routine_regularity = st.select_slider(
        "How structured is your daily routine?",
        options=["low", "medium", "high"],
        value="medium",
    )

    st.subheader("Simulation Context")
    context = st.selectbox(
        "Current context",
        ["home", "work", "commuting", "waiting", "other"],
    )

    day_type = st.radio("Day type", ["weekday", "weekend"])
    hour_of_day = st.slider("Hour of day", 0, 23, 12)

    submitted = st.form_submit_button("🚀 Generate Persona", use_container_width=True)

# --- Main Content ---
if submitted:
    with st.spinner("Generating persona..."):
        payload = {
            "age": age,
            "city": city,
            "job": job,
            "chronotype_self_report": chronotype,
            "phone_style": phone_style,
            "primary_use": primary_use,
            "usage_level": usage_level,
            "activity_level": activity_level,
            "commute_frequency": commute_frequency,
            "routine_regularity": routine_regularity,
            "context": context,
            "day_type": day_type,
            "hour_of_day": hour_of_day,
            "run_on_device": False,
            "max_steps": 10,
        }

        try:
            response = requests.post(f"{API_URL}/generate-persona", json=payload)
            response.raise_for_status()
            result = response.json()

            st.success("✅ Persona generated successfully!")

            # Store in session state
            st.session_state["persona_result"] = result

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error: {e}")

# --- Display Results ---
if "persona_result" in st.session_state:
    result = st.session_state["persona_result"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Behavior Specification")
        behavior_spec = result.get("behavior_spec", {})

        # Radar chart of behavior dimensions
        if behavior_spec:
            categories = list(behavior_spec.keys())
            values = []
            for v in behavior_spec.values():
                if isinstance(v, (int, float)):
                    values.append(v)
                else:
                    values.append(0)

            if values:
                fig = go.Figure(
                    data=go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill="toself",
                        name="Behavior Profile",
                    )
                )
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                    title="Behavioral Dimensions",
                )
                st.plotly_chart(fig, use_container_width=True)

        st.json(behavior_spec)

    with col2:
        st.subheader("⚙️ Derived Parameters")
        parameters = result.get("parameters", {})

        if parameters:
            # Key metrics
            metrics_col1, metrics_col2 = st.columns(2)

            with metrics_col1:
                st.metric(
                    "Daily Screen Time",
                    f"{parameters.get('total_daily_usage_minutes', 0):.0f} min",
                )
                st.metric(
                    "Avg Session Duration",
                    f"{parameters.get('avg_session_duration_seconds', 0):.0f} sec",
                )

            with metrics_col2:
                st.metric(
                    "Pickups/Hour",
                    f"{parameters.get('baseline_pickups_per_hour', 0):.1f}",
                )
                st.metric(
                    "Glance Probability",
                    f"{parameters.get('glance_probability', 0):.0%}",
                )

            # App weights pie chart
            app_weights = {
                "Social": parameters.get("social_weight", 0),
                "Process": parameters.get("process_weight", 0),
                "Navigation": parameters.get("navigation_weight", 0),
                "Music": parameters.get("music_weight", 0),
            }

            fig = px.pie(
                values=list(app_weights.values()),
                names=list(app_weights.keys()),
                title="App Category Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("View All Parameters"):
            st.json(parameters)

    # --- Temporal Distribution Visualization ---
    st.subheader("🕐 Temporal Usage Pattern")

    parameters = result.get("parameters", {})
    if parameters:
        # Simulate hourly usage probability
        import numpy as np

        peak_shift = parameters.get("temporal_peak_shift", 0)
        late_night_prob = parameters.get("late_night_usage_probability", 0.1)
        waking_start = parameters.get("waking_start_hour", 7)
        waking_end = parameters.get("waking_end_hour", 23)

        hours = list(range(24))
        base_activity = []

        for h in hours:
            # Base activity curve (peaks in evening)
            base_peak = 20 + peak_shift  # Default evening peak
            distance_from_peak = min(abs(h - base_peak), 24 - abs(h - base_peak))
            activity = max(0, 1 - (distance_from_peak / 12))

            # Suppress during sleep
            if waking_end > waking_start:
                if h < waking_start or h > waking_end:
                    activity *= 0.1
            else:  # Wraps around midnight
                if h < waking_start and h > waking_end:
                    activity *= 0.1

            # Late night boost for night owls
            if h >= 22 or h <= 2:
                activity += late_night_prob * 0.5

            base_activity.append(activity)

        # Normalize
        max_act = max(base_activity)
        if max_act > 0:
            base_activity = [a / max_act for a in base_activity]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=hours,
                y=base_activity,
                name="Usage Probability",
                marker_color="steelblue",
            )
        )
        fig.update_layout(
            title="Predicted Hourly Phone Usage Pattern",
            xaxis_title="Hour of Day",
            yaxis_title="Relative Activity",
            xaxis=dict(tickmode="linear", tick0=0, dtick=2),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- File Paths ---
    st.subheader("📁 Generated Artifacts")
    st.code(
        f"""
Persona ID: {result.get('persona_id')}
Save Directory: {result.get('save_dir')}
Persona JSON: {result.get('persona_path')}
    """
    )

    # --- Device Execution (Optional) ---
    st.subheader("📱 Device Execution")
    st.warning("⚠️ Requires MCP server running and device connected")

    if st.button("▶️ Execute on Device"):
        with st.spinner("Executing persona on device..."):
            try:
                exec_response = requests.post(
                    f"{API_URL}/execute-persona",
                    params={
                        "persona_path": result.get("persona_path"),
                        "max_steps": 10,
                    },
                )
                exec_response.raise_for_status()
                exec_result = exec_response.json()

                st.success("✅ Execution complete!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Steps Taken", exec_result.get("steps_taken", 0))
                with col2:
                    st.metric("Entropy", f"{exec_result.get('entropy', 0):.2f}")
                with col3:
                    st.metric(
                        "Repetition Rate",
                        f"{exec_result.get('repetition_rate', 0):.2%}",
                    )

                st.json(exec_result.get("app_distribution", {}))

            except Exception as e:
                st.error(f"❌ Execution failed: {e}")

# --- Footer ---
st.markdown("---")
st.markdown(
    "Built for research purposes. "
    "Behavioral parameters grounded in smartphone usage literature."
)
