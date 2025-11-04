# uiauto_agent/planner.py
from typing import List, Dict, Any

def build_action_plan(persona) -> List[Dict[str, Any]]:
    """
    Returns a list of high-level actions the agent should take
    in order, derived from persona traits.
    """
    plan: List[Dict[str, Any]] = []

    act = (persona.activity_description or "").lower()
    income_type = (persona.income_type or "").lower()
    online = (persona.online_behavior or "").lower()

    # Always: play something on Spotify tailored to activity
    plan.append({"app": "spotify", "action": "play_for_persona"})

    # Facebook usage pattern
    if "social" in online or "facebook" in online or "browse" in act:
        plan.append({"app": "facebook", "action": "open_and_browse"})
        plan.append({"app": "facebook", "action": "search_topic",
                     "args": {"topic": f"{persona.city} events"}})

        # higher income or “outgoing” → maybe post
        if "high" in income_type or "moderate" in income_type:
            plan.append({"app": "facebook", "action": "maybe_post_status"})

    # Could add: News app, Maps, Weather, YouTube, etc. based on persona.
    return plan