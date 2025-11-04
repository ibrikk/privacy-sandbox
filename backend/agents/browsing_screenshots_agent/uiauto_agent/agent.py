# uiauto_agent/agent.py
from typing import Dict, Any, List
from .device import connect, press, stop_app, get_installed_apps
from . import llm_planner
from . import actions_spotify as SP
from . import actions_facebook as FB
from . import actions_tiktok as TK
from . import actions_camera as CAM
from .selectors import PKG_SPOTIFY, PKG_FB, PKG_TIKTOK, PKG_CAMERA

DISPATCH = {
    # Spotify
    ("spotify", "play_for_persona"): SP.play_for_persona,
    # Facebook
    ("facebook", "open_and_browse"): FB.open_and_browse,
    ("facebook", "search_topic"):     FB.search_topic,
    ("facebook", "maybe_post_status"): FB.maybe_post_status,
    # TikTok
    ("tiktok", "watch_and_scroll"): TK.watch_and_scroll,
    # Camera
    ("camera", "take_selfie"): CAM.take_selfie,
}

def run_persona_session(persona, serial: str | None = None, max_steps: int = 10, stop_after=True):
    d = connect(serial)
    history: List[Dict[str, Any]] = []
    apps_used = set()

    try:
        installed_apps = get_installed_apps(d)
        available_actions = [f"{app}.{act}" for app, act in DISPATCH.keys()]

        for i in range(max_steps):
            print(f"\n--- Step {i+1}/{max_steps} ---")
            step = llm_planner.get_next_action(persona, installed_apps, history, available_actions)
            if not step:
                print("⚠️ LLM returned no action. Ending session.")
                break

            app, action_name = step.get("app"), step.get("action")
            key = (app, action_name)
            fn = DISPATCH.get(key)

            if not fn:
                print(f"⚠️ LLM requested unknown action '{key}'. Skipping.")
                history.append({"action": step, "status": "error", "error": "Unknown action"})
                continue

            args = step.get("args", {})
            apps_used.add(app)
            try:
                print(f"▶️  Executing: {key} | args={args}")
                # Pass persona only if the function expects it
                if "persona" in fn.__code__.co_varnames:
                    fn(d, persona=persona, **args)
                else:
                    fn(d, **args)
                history.append({"action": step, "status": "success"})
            except Exception as e:
                print(f"❌ Action {key} failed: {e}")
                history.append({"action": step, "status": "error", "error": str(e)})

            d.sleep(1.5) # Pause between actions

    finally:
        # Optional cleanup
        if stop_after:
            print("\n--- Cleaning up ---")
            for app_name in apps_used:
                pkg = globals().get(f"PKG_{app_name.upper()}")
                if pkg:
                    try:
                        stop_app(d, pkg)
                        print(f"🛑 Stopped {app_name}")
                    except Exception:
                        pass
            press(d, "home")
        print("✅ Session finished.")