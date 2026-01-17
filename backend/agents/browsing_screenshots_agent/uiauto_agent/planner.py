# uiauto_agent/planner.py
import random
from typing import List, Dict, Any

from models import ThoughtAction, PrivacyAttributes


def build_action_plan(thoughtAction: ThoughtAction, persona: PrivacyAttributes, installed_apps: List[str]) -> List[Dict[str, Any]]:
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
            matched = False
            for i in installed_apps:
                if app in i:
                    plan.append({"app": i, "action": action, "args": kwargs})
                    matched = True
                    break
            if not matched:
                return

    # 1️⃣ Start with the immediate LLM-selected action
    # plan.append({"app": app, "action": intent, "args": args})
    maybe(1, app, intent)


    def should_consider_app(app_name: str) -> bool:
        """Check if app is mentioned in thought/app context and not recently used."""
        recent_apps = {step["app"] for step in plan[-3:]}
        # Check if app_name is contained in any recent app (e.g., "spotify" in "com.spotify.test")
        is_recently_used = any(app_name in recent_app for recent_app in recent_apps)
        return (
            app_name in thought
            or app_name in app
        ) and not is_recently_used

        
    if should_consider_app("facebook"):
        maybe(0.7, "facebook", "search_topic", topic=f"{city} events")
        maybe(0.5, "facebook", "maybe_post_status")

    if should_consider_app("spotify"):
        maybe(0.7, "spotify", "play_for_persona")
        maybe(0.5, "facebook", "open_and_browse")

    if should_consider_app("tiktok"):
        maybe(0.8, "tiktok", "watch_and_scroll")
        maybe(0.5, "camera", "take_selfie")

    if should_consider_app("camera"):
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
        maybe(0.7, "apps.weather", "check_weather")

    if "gym" in act or "run" in act or "jog" in act:
        maybe(0.8, "spotify", "play_for_persona")

    # 4️⃣ Online behavior cues — strong personalization
    online_behavior_bias = {
    "social": 1.0,        # Facebook, community, posting
    "visual": 1.0,        # Instagram, photos, aesthetics
    "video": 1.0,         # TikTok, YouTube, streaming
    "news": 1.0,          # News, reading, headlines
    "commerce": 1.0,      # Amazon, shopping, buying
    "professional": 1.0,  # LinkedIn, tech, career
    }
    
    def prob(base: float, bias: float) -> float:
        return min(0.95, base * bias)


    # --- Online behavior bias amplification ---
    if any(w in online for w in ["social", "community", "friends", "post", "facebook"]):
        online_behavior_bias["social"] += 0.6

    if any(w in online for w in ["photos", "instagram", "fashion", "style", "visual"]):
        online_behavior_bias["visual"] += 0.6

    if any(w in online for w in ["video", "watch", "youtube", "stream", "tiktok"]):
        online_behavior_bias["video"] += 0.6

    if any(w in online for w in ["news", "reading", "headlines", "articles"]):
        online_behavior_bias["news"] += 0.5

    if any(w in online for w in ["shop", "buy", "shopping", "ecommerce"]):
        online_behavior_bias["commerce"] += 0.5

    if any(w in online for w in ["tech", "developer", "engineering", "forums", "github", "career"]):
        online_behavior_bias["professional"] += 0.5

    
    # --- Social ---
    maybe(prob(0.4, online_behavior_bias["social"]), "facebook", "open_and_browse")
    maybe(prob(0.3, online_behavior_bias["social"]), "facebook", "maybe_post_status")

    # --- Visual ---
    maybe(prob(0.35, online_behavior_bias["visual"]), "instagram", "browse_feed")
    maybe(prob(0.25, online_behavior_bias["visual"]), "instagram", "view_stories")

    # --- Video ---
    maybe(prob(0.35, online_behavior_bias["video"]), "tiktok", "watch_and_scroll")
    maybe(prob(0.25, online_behavior_bias["video"]), "youtube", "watch_recommended")

    # --- News ---
    maybe(prob(0.3, online_behavior_bias["news"]), "news", "check_headlines")

    # --- Commerce ---
    maybe(prob(0.35, online_behavior_bias["commerce"]), "amazon", "browse_recommendations")
    maybe(prob(0.25, online_behavior_bias["commerce"]), "temu", "open_homepage")

    # --- Professional ---
    maybe(prob(0.4, online_behavior_bias["professional"]), "linkedin", "browse_feed")
    maybe(prob(0.2, online_behavior_bias["professional"]), "news", "check_headlines")

    # 5️⃣ Age and income adjustments
    if age < 25:
        maybe(0.6, "tiktok", "watch_and_scroll")
        maybe(0.5, "instagram", "browse_feed")
    elif age > 40:
        maybe(0.6, "facebook", "open_and_browse")
        maybe(0.4, "apps.weather", "check_weather")

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
