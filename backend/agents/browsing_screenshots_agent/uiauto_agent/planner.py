# uiauto_agent/planner.py
from typing import List, Dict, Any


def build_action_plan(persona) -> List[Dict[str, Any]]:
    """
    Returns a list of high-level actions the agent should take
    in order, derived from persona traits.
    """
    plan: List[Dict[str, Any]] = []

    act = (persona.activity_description or "").lower()
    online = (persona.online_behavior or "").lower()
    job = (persona.job or "").lower()
    income_type = (persona.income_type or "").lower()

    # --- Core logic: behavioral anchors ---
    # 1️⃣ Fitness or outdoor personas → Spotify first
    if any(k in act for k in ["jog", "run", "walk", "commute", "drive"]):
        plan.append({"app": "spotify", "action": "play_for_persona", "args": {}})

    # 2️⃣ Tech-savvy or social personas → Facebook or TikTok browsing
    if any(k in online for k in ["social", "facebook", "post", "tiktok", "instagram"]):
        plan.append({"app": "facebook", "action": "open_and_browse", "args": {}})
        plan.append(
            {
                "app": "facebook",
                "action": "search_topic",
                "args": {"topic": f"{persona.city} events"},
            }
        )
        if "high" in income_type or "moderate" in income_type:
            plan.append({"app": "facebook", "action": "maybe_post_status", "args": {}})

    # 3️⃣ Creative or influencer personas → Camera use
    if any(
        k in job
        for k in ["creator", "artist", "designer", "photographer", "influencer"]
    ):
        plan.append({"app": "camera", "action": "take_selfie", "args": {}})

    # 4️⃣ Entertainment / leisure personas → TikTok
    if any(k in act for k in ["resting", "break", "evening", "relaxing"]):
        plan.append({"app": "tiktok", "action": "watch_and_scroll", "args": {}})

    # 5️⃣ Default fallback
    if not plan:
        plan.append({"app": "facebook", "action": "open_and_browse", "args": {}})
    # Could add: News app, Maps, Weather, YouTube, etc. based on persona.
    return plan
