import json
import os
from typing import Callable, Tuple, TypedDict, List, Dict, Any, Literal, cast
from langgraph.graph import StateGraph, END


from models import (
    PrivacyAttributes,
)


# 1. Define the State for our graph
class AgentState(TypedDict):
    d: Any  # The uiautomator2 device object
    persona: PrivacyAttributes
    installed_apps: List[str]
    available_actions: List[str]
    history: List[Dict[str, Any]]
    current_action: Dict[str, Any]
    steps_taken: int
    max_steps: int
    apps_used: set


ALLOWED_ACTIONS = {
    "spotify": ["play_for_persona"],
    "facebook": ["open_and_browse", "search_topic", "maybe_post_status"],
    "instagram": ["view_stories", "view_reels"],
    "tiktok": ["watch_and_scroll"],
    "camera": ["take_selfie"],
}


# 2. Define the Nodes for our graph


def plan_step(state: AgentState) -> dict:
    """Calls the LLM planner to decide the next action."""
    print(f"\n--- Planning Step {state['steps_taken'] + 1}/{state['max_steps']} ---")

    action = llm_planner.get_next_action(
        state["persona"],
        state["installed_apps"],
        state["history"],
        state["available_actions"],
    )

    if not action or not action.get("app") or not action.get("action"):
        print("⚠️ LLM returned incomplete action. Will end session.")
        return {**state, "current_action": {"app": "system", "action": "terminate"}}

    return {**state, "current_action": action}


def execute_step(state: AgentState) -> dict:
    """Executes the action planned by the LLM."""
    step = state["current_action"]
    app, action_name = step.get("app"), step.get("action")
    key: Tuple[str, str] = cast(Tuple[str, str], (app, action_name))

    fn = None
    if app and action_name:
        for (dispatch_app, dispatch_action), handler_fn in DISPATCH.items():
            if dispatch_app in app and dispatch_action == action_name:  
                fn = handler_fn
                break
    if not fn:
        error_msg = f"LLM requested unknown or invalid action '{(app, action_name)}'."
        print(f"⚠️ {error_msg}")
        new_history = state["history"] + [
            {"action": step, "status": "error", "error": error_msg}
        ]
        # This counts as a step, even if it's an error
        return {
            **state,
            "history": new_history,
            "steps_taken": state["steps_taken"] + 1,
        }

    args = step.get("args", {})
    new_apps_used = state["apps_used"].copy()
    new_apps_used.add(app)
    new_history = state["history"].copy()
  
    try:
        print(f"▶️  Executing: {key} | args={args}")
        # Pass persona only if the function expects it
        # TODO: check what this does
        if "persona" in fn.__code__.co_varnames:
            fn(d=state["d"], persona=state["persona"], **args)
        else:
            fn(d=state["d"], **args)

        new_history.append({"action": step, "status": "success"})
        print(f"✅ Action {key} succeeded.")
    except Exception as e:
        print(f"❌ Action {key} failed: {e}")
        new_history.append({"action": step, "status": "error", "error": str(e)})

    # Always increment the step counter
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
    return "execute_step"


# 4. Assemble the graph
workflow = StateGraph(AgentState)
workflow.add_node("plan_step", plan_step)
workflow.add_node("execute_step", execute_step)

workflow.set_entry_point("plan_step")
workflow.add_conditional_edges(
    "plan_step",
    should_continue,
    {
        "execute_step": "execute_step",
        END: END,
    },
)
workflow.add_edge("execute_step", "plan_step")

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
    persona_json_path: str, serial: str | None = None, max_steps: int = 10
) -> AgentState:
    """Initialize AgentState from a persona.json file path."""
    persona = load_persona_from_json(persona_json_path)
    d = connect(serial)
    return AgentState(
        d=d,
        persona=persona,
        installed_apps=get_installed_apps(d),
        available_actions = [
        f"{app}.{action}"
        for app, actions in ALLOWED_ACTIONS.items()
        for action in actions
        ],
        history=[],
        current_action={},
        steps_taken=0,
        max_steps=max_steps,
        apps_used=set(),
    )


def run_persona_session(
    persona_json_path: str,
    serial: str | None = None,
    max_steps: int = 10,
    stop_after=True,
):
    """
    Runs a persona-driven session on an Android device using a LangGraph agent.

    This is the main entry point for the agent. It initializes the state,
    invokes the graph, and handles cleanup.

    Args:
        persona_json_path: Path to the persona.json file.
        serial: Optional device serial number for connection.
        max_steps: Maximum number of steps to execute.
        stop_after: Whether to stop apps after the session.
    """
    initial_state: AgentState = initialize_agent_state(
        persona_json_path, serial, max_steps
    )
    d = initial_state["d"]
    final_state = None

    try:
        # The invoke method will stream all intermediate states.
        # We are only interested in the final state here.
        for s in app.stream(initial_state):
            final_state = s
    finally:
        if stop_after:
            print("\n--- Cleaning up ---")
            apps_to_stop = (
                final_state["apps_used"]
                if final_state and "apps_used" in final_state
                else initial_state["apps_used"]
            )
            for app_name in apps_to_stop:
                pkg = globals().get(f"PKG_{app_name.upper()}")
                if pkg:
                    try:
                        stop_app(d, pkg)
                        print(f"🛑 Stopped {app_name}")
                    except Exception as e:
                        print(f"Could not stop {app_name}: {e}")
            press(d, "home")
        print("✅ Session finished.")
        if final_state:
            print("\n--- Final State ---")
            # Print a summary of the final state without the device object
            summary = {k: v for k, v in final_state.items() if k != "d"}
            print(json.dumps(summary, indent=2, default=str))

