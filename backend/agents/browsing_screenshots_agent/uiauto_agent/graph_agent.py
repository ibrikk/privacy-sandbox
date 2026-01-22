import json
import os
from typing import Callable, Tuple, TypedDict, List, Dict, Any, Literal, cast
from agents.browsing_screenshots_agent.uiauto_agent import llm_planner
from langgraph.graph import StateGraph, END


from models import (
    PrivacyAttributes,
)


# 1. Define the State for our graph
class AgentState(TypedDict):
    persona: PrivacyAttributes
    installed_apps: List[str]
    available_actions: List[str]
    history: List[Dict[str, Any]]             
    current_action: Dict[str, Any]            
    steps_taken: int
    max_steps: int
    apps_used: List[str] 


ALLOWED_ACTIONS = {
    "com.spotify.music": ["play_for_persona"],
    "com.facebook.katana": ["open_and_browse", "search_topic", "maybe_post_status"],
    "com.instagram.android": ["browse_feed", "view_reels"],
    "com.zhiliaoapp.musically": ["watch_and_scroll"],
    "com.android.cameraextensions": ["take_selfie"],
    # "com.linkedin.android": ["browse_feed"],
    "com.google.android.youtube.music": ["watch_recommended"]
}

AVAILABLE_ACTIONS: List[str] = [
    f"{app}.{action}"
    for app, actions in ALLOWED_ACTIONS.items()
    for action in actions
]

def _is_valid_action(action: Dict[str, list[str]]) -> bool:
    app = action.get("app")
    act = action.get("action")

    if not app or not act:
        return False

    if app not in ALLOWED_ACTIONS:
        return False

    if act not in ALLOWED_ACTIONS[app]:
        return False

    return True

def _normalize_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize action structure coming from the LLM.

    Expected output shape:
      {
        "app": "facebook",
        "action": "open_and_browse",
        "args": {...}   # optional
      }
    """
    if not isinstance(action, dict):
        return {}

    app = action.get("app")
    act = action.get("action") or action.get("intent")
    args = action.get("args") or {}

    if isinstance(app, str):
        app = app.strip().lower()
    else:
        app = None

    if isinstance(act, str):
        act = act.strip().lower()
    else:
        act = None

    if not isinstance(args, dict):
        args = {}

    normalized: Dict[str, Any] = {"app": app, "action": act, "args": args}
    return normalized


# ----------------------------
# 3) Graph nodes
# ----------------------------

def plan_step(state: AgentState) -> AgentState:
    """Calls the LLM planner to decide the next action."""
    print(f"\n--- Planning Step {state['steps_taken'] + 1}/{state['max_steps']} ---")

    raw = llm_planner.get_next_action(
        state["persona"],
        state["installed_apps"],
        state["history"],
        state["available_actions"],
    )

    action = _normalize_action(raw)
    
    # If LLM returns nothing / malformed, terminate.
    if not action or not action.get("app") or not action.get("action"):
        print("⚠️ LLM returned no/invalid action. Terminating.")
        action = {"app": "system", "action": "terminate", "args": {}}
        new_history = state["history"] + [{"action": action, "status": "terminate"}]
        return {
            **state,
            "current_action": action,
            "history": new_history,
        }

    # If action is outside allowed space, terminate (and record).
    if not _is_valid_action(action):
        print(f"⚠️ Invalid LLM action (out of allowed set): {action}")
        term = {"app": "system", "action": "terminate", "args": {}}
        new_history = state["history"] + [
            {"action": action, "status": "invalid"},
            {"action": term, "status": "terminate"},
        ]
        return {
            **state,
            "current_action": term,
            "history": new_history,
        }

    return {**state, "current_action": action}

def record_step(state: AgentState) -> AgentState:
    """
    Records the validated action and advances the step counter.
    Execution happens elsewhere (executor/orchestrator layer using MCP).
    """
    action = state["current_action"]
    app_name = action.get("app")

    new_history = state["history"] + [{"action": action, "status": "planned"}]

    new_apps_used = list(state["apps_used"])
    if isinstance(app_name, str) and app_name not in new_apps_used and app_name != "system":
        new_apps_used.append(app_name)

    return {
        **state,
        "history": new_history,
        "steps_taken": state["steps_taken"] + 1,
        "apps_used": new_apps_used,
    }

# 3. Define the Conditional Edge
def should_continue(state: AgentState):
    """Determines whether to continue the loop or end the session."""
    if state["steps_taken"] >= state["max_steps"]:
        print("🏁 Reached max steps. Ending session.")
        return END
    if state["current_action"].get("action") == "terminate":
        print("🛑 Terminate action received. Ending session.")
        return END
    return "record_step"


# 4. Assemble the graph
workflow = StateGraph(AgentState)
workflow.add_node("plan_step", plan_step)
workflow.add_node("record_step", record_step)

workflow.set_entry_point("plan_step")
workflow.add_conditional_edges(
    "plan_step",
    should_continue,
    {
        "record_step": "record_step",
        END: END,
    },
)
workflow.add_edge("record_step", "plan_step")

app = workflow.compile()


def load_persona_from_json(persona_json_path: str) -> PrivacyAttributes:
    """Load and parse a persona from a JSON file path."""
    if not os.path.exists(persona_json_path):
        raise FileNotFoundError(f"Persona JSON file not found: {persona_json_path}")

    with open(persona_json_path, "r", encoding="utf-8") as f:
        persona_data = json.load(f)

    try:
        return PrivacyAttributes(**persona_data)
    except Exception as e:
        raise ValueError(f"Failed to parse persona from JSON: {e}")


def initialize_agent_state(
    persona_json_path: str, installed_apps: list[str], max_steps: int = 10
) -> AgentState:
    """Initialize AgentState from a persona.json file path.
    Note:
    - installed_apps should come from the orchestrator (MCP call)
    - no device/session objects are stored here
    """
    persona = load_persona_from_json(persona_json_path)

    return {
        "persona": persona,
        "installed_apps": installed_apps,
        "available_actions": list(AVAILABLE_ACTIONS),
        "history": [],
        "current_action": {},
        "steps_taken": 0,
        "max_steps": max_steps,
        "apps_used": [],
    }