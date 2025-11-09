
# uiauto_agent/planner.py
import random
from typing import List, Dict, Any

from agents.browsing_screenshots_agent.uiauto_agent.llm_planner import ThoughtAction
from backend.agents.persona_generator_agent.langgraph_persona_generator import PrivacyAttributes   # pyright: ignore[reportMissingImports]


def build_action_plan(thoughtAction: ThoughtAction, persona: PrivacyAttributes) -> List[Dict[str, Any]]:
    """
    Build a realistic mobile behavior session plan for a persona.
    Returns an ordered list of {app, action, args} steps.
    """

    plan: List[Dict[str, Any]] = []
    
    

    # --- Extract key traits ---
    act = (persona.activity_description or "").lower()
    online = (persona.online_behavior or "").lower()
    job = (persona.job or "").lower()
    income_type = (persona.income_type or "").lower()
    city = (persona.city or "their city").title()
    age = int(persona.age) 

    # --- Helper: probabilistic append ---
    def maybe(p: float, app: str, action: str, **args):
        """Append with probability p (0–1)."""
        if random.random() < p:
            plan.append({"app": app, "action": action, "args": args})

    # --- Activity anchors ---
    # Morning jogger / commuter → Spotify
    if any(k in act for k in ["jog", "run", "walk", "commute", "drive", "gym"]):
        plan.append({"app": "spotify", "action": "play_for_persona", "args": {}})

    # Working or studying → Spotify + Weather
    if any(k in act for k in ["work", "study", "office"]):
        maybe(0.8, "spotify", "play_for_persona")
        maybe(0.5, "weather", "check_weather")

    # Leisure / relaxing → TikTok, YouTube
    if any(k in act for k in ["resting", "break", "evening", "relaxing", "bed"]):
        maybe(0.6, "tiktok", "watch_and_scroll")
        maybe(0.6, "youtube", "watch_recommended")

    # Outdoors or traveler → Maps + Weather
    if any(k in act for k in ["travel", "trip", "vacation", "outdoor", "hiking"]):
        maybe(0.9, "weather", "check_weather")
        maybe(0.5, "maps", "search_location", query=f"cafes near {city}")

    # --- Profession-based anchors ---
    if any(k in job for k in ["engineer", "developer", "designer", "manager"]):
        maybe(0.7, "facebook", "open_and_browse")
        maybe(0.5, "linkedin", "browse_feed")
        maybe(0.3, "tiktok", "watch_and_scroll")

    if any(k in job for k in ["creator", "artist", "photographer", "influencer"]):
        maybe(0.8, "camera", "take_selfie")
        maybe(0.7, "instagram", "browse_feed")
        maybe(0.6, "instagram", "view_stories")
        maybe(0.5, "tiktok", "watch_and_scroll")

    # --- Online behavior patterns ---
    if "facebook" in online or "social" in online or "post" in online:
        plan.append({"app": "facebook", "action": "open_and_browse", "args": {}})
        maybe(0.7, "facebook", "search_topic", topic=f"{city} events")
        maybe(0.5, "facebook", "maybe_post_status")

    if "instagram" in online:
        maybe(0.9, "instagram", "browse_feed")
        maybe(0.8, "instagram", "view_stories")
        maybe(0.5, "instagram", "search_interest")

    if "tiktok" in online or "video" in online:
        maybe(0.8, "tiktok", "watch_and_scroll")

    # --- Demographic-based additions ---
    if age < 25:
        maybe(0.6, "tiktok", "watch_and_scroll")
        maybe(0.5, "instagram", "browse_feed")
    elif age > 40:
        maybe(0.7, "facebook", "open_and_browse")
        maybe(0.4, "weather", "check_weather")

    if "high" in income_type:
        maybe(0.6, "linkedin", "browse_feed")
        maybe(0.4, "news", "check_headlines")

    # --- Fallback if nothing planned ---
    if not plan:
        plan.append({"app": "facebook", "action": "open_and_browse", "args": {}})

    # Shuffle for natural variation, but keep Spotify first if present
    spotify_first = [p for p in plan if p["app"] == "spotify"]
    others = [p for p in plan if p["app"] != "spotify"]
    random.shuffle(others)
    final_plan = spotify_first + others

    return final_plan
