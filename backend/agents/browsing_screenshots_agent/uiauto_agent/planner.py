# uiauto_agent/planner.py
import random
from typing import List, Dict, Any

from models import ThoughtAction, PrivacyAttributes


def build_action_plan(thoughtAction: ThoughtAction, persona: PrivacyAttributes) -> List[Dict[str, Any]]:
    """
    Build a realistic multi-step mobile behavior plan based on
    the LLM's latest ThoughtAction and the persona's long-term traits.
    Returns a list of {app, action, args} dicts.
    """

    plan: List[Dict[str, Any]] = []

    # --- Extract short- and long-term cues ---
    thought = (thoughtAction.thought or "").lower()
    action = thoughtAction.action
    app = (action.app or "").lower()
    intent = (action.intent or "").lower()
    args = action.args or {}

    act = (persona.activity_description or "").lower()
    job = (persona.job or "").lower()
    online = (persona.online_behavior or "").lower()
    income_type = (persona.income_type or "").lower()
    city = (persona.city or "their city").title()
    age = int(persona.age or 30)

    # --- Helper: probabilistic append ---
    def maybe(p: float, app: str, action: str, **kwargs):
        if random.random() < p:
            plan.append({"app": app, "action": action, "args": kwargs})

    # 1️⃣ Start with the immediate LLM-selected action
    plan.append({"app": app, "action": intent, "args": args})

    # 2️⃣ Use the LLM thought for near-future continuity
    if "facebook" in thought or app == "facebook":
        maybe(0.7, "facebook", "search_topic", topic=f"{city} events")
        maybe(0.5, "facebook", "maybe_post_status")

    if "spotify" in thought or app == "spotify":
        maybe(0.7, "spotify", "play_for_persona")
        maybe(0.5, "facebook", "open_and_browse")

    if "tiktok" in thought or app == "tiktok":
        maybe(0.8, "tiktok", "watch_and_scroll")
        maybe(0.5, "camera", "take_selfie")

    if "camera" in thought or app == "camera":
        maybe(0.8, "camera", "take_selfie")
        maybe(0.6, "instagram", "browse_feed")

    # 3️⃣ Persona-based tendencies (job, lifestyle, etc.)
    if "developer" in job or "engineer" in job:
        maybe(0.6, "linkedin", "browse_feed")
        maybe(0.3, "facebook", "open_and_browse")

    if "designer" in job or "artist" in job:
        maybe(0.8, "instagram", "view_stories")
        maybe(0.6, "camera", "take_selfie")

    if "travel" in act or "commute" in act or "drive" in act:
        maybe(0.8, "maps", "search_location", query=f"cafes near {city}")
        maybe(0.7, "weather", "check_weather")

    if "gym" in act or "run" in act:
        maybe(0.8, "spotify", "play_for_persona")

    # 4️⃣ Online behavior cues — strong personalization
    if any(word in online for word in ["social", "facebook", "friends", "post", "community"]):
        maybe(0.8, "facebook", "open_and_browse")
        maybe(0.6, "instagram", "browse_feed")
        maybe(0.4, "tiktok", "watch_and_scroll")

    if any(word in online for word in ["instagram", "photos", "fashion", "style"]):
        maybe(0.8, "instagram", "browse_feed")
        maybe(0.7, "instagram", "view_stories")

    if any(word in online for word in ["tech", "developer", "forums", "github"]):
        maybe(0.8, "linkedin", "browse_feed")
        maybe(0.4, "news", "check_headlines")

    if any(word in online for word in ["shop", "shopping", "buy", "ecommerce"]):
        maybe(0.8, "amazon", "browse_recommendations")
        maybe(0.6, "temu", "open_homepage")

    if any(word in online for word in ["video", "stream", "watch", "youtube"]):
        maybe(0.7, "tiktok", "watch_and_scroll")
        maybe(0.5, "youtube", "watch_recommended")

    # 5️⃣ Age and income adjustments
    if age < 25:
        maybe(0.6, "tiktok", "watch_and_scroll")
        maybe(0.5, "instagram", "browse_feed")
    elif age > 40:
        maybe(0.6, "facebook", "open_and_browse")
        maybe(0.4, "weather", "check_weather")

    if "high" in income_type:
        maybe(0.5, "linkedin", "browse_feed")
        maybe(0.4, "news", "check_headlines")

    # 6️⃣ Fallback
    if not plan:
        plan.append({"app": "facebook", "action": "open_and_browse", "args": {}})

    # 7️⃣ Shuffle for realism, but keep LLM action first
    llm_first = plan[:1]
    rest = plan[1:]
    random.shuffle(rest)

    return llm_first + rest
