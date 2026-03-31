# server.py

import asyncio
import uuid
from typing import cast
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import (
    UserProfileInput,
    SimulationRequest,
    SimulationResponse,
)
from agents.persona_generator_agent.langgraph_persona_generator import (
    app as langgraph_app,
    GraphState,
)

app = FastAPI(
    title="Synthetic Persona Generator",
    description="Generate literature-grounded synthetic smartphone personas",
    version="0.1.0",
)

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def simulation_request_to_user_profile(req: SimulationRequest) -> UserProfileInput:
    """Convert API request to internal UserProfileInput."""
    return UserProfileInput(
        age=req.age,
        city=req.city,
        job=req.job,
        chronotype_self_report=req.chronotype_self_report,
        phone_style=req.phone_style,
        primary_use=req.primary_use,
        usage_level=req.usage_level,
        activity_level=req.activity_level,
        commute_frequency=req.commute_frequency,
        routine_regularity=req.routine_regularity,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate-persona", response_model=SimulationResponse)
async def generate_persona(request: SimulationRequest):
    """
    Generate a synthetic persona from survey input.

    This endpoint:
    1. Converts survey input to UserProfileInput
    2. Runs LangGraph pipeline (behavior spec → parameters → traces)
    3. Returns persona artifacts and visualization data
    """
    try:
        # Convert request to internal format
        user_profile = simulation_request_to_user_profile(request)

        # Run LangGraph pipeline
        # Note: Your existing pipeline expects UserInput, we may need to adapt
        initial_state = cast(
            GraphState,
            {
                "prompt": user_profile,
            },
        )

        final_state = langgraph_app.invoke(initial_state)

        # Extract results
        save_dir = final_state.get("save_dir")
        if not save_dir:
            raise HTTPException(status_code=500, detail="Pipeline failed: no save_dir")

        persona_path = f"{save_dir}/persona.json"

        # Build response
        response = SimulationResponse(
            persona_id=str(uuid.uuid4()),
            save_dir=save_dir,
            persona_path=persona_path,
            behavior_spec=(
                final_state.get("behavior_spec", {}).dict()
                if hasattr(final_state.get("behavior_spec", {}), "dict")
                else final_state.get("behavior_spec", {})
            ),
            parameters=(
                final_state.get("parameters", {}).dict()
                if hasattr(final_state.get("parameters", {}), "dict")
                else final_state.get("parameters", {})
            ),
            schedule_summary=None,  # TODO: add schedule generation
            execution_metrics=None,
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute-persona")
async def execute_persona(persona_path: str, max_steps: int = 10):
    """
    Execute a generated persona on connected device.
    Separate endpoint so generation and execution are decoupled.
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    from agents.browsing_screenshots_agent.uiauto_agent.graph_agent import (
        AgentState,
        initialize_agent_state,
        app as graph_app,
    )
    from agents.browsing_screenshots_agent.uiauto_agent.executor import execute_action
    from agents.browsing_screenshots_agent.uiauto_agent.metrics import (
        app_distribution,
        entropy,
        repetition_rate,
    )

    MCP_URL = "http://localhost:8080/mcp"

    try:
        async with streamablehttp_client(MCP_URL) as (read, write, _):
            async with ClientSession(read, write) as mcp:
                await mcp.initialize()

                # Get installed apps
                raw_apps = await mcp.call_tool("get_installed_apps")
                installed_apps = _normalize_apps(raw_apps)

                # Initialize agent
                state: AgentState = initialize_agent_state(
                    persona_json_path=persona_path,
                    installed_apps=installed_apps,
                    max_steps=max_steps,
                )

                # Run
                async for updated_state in graph_app.astream(state):
                    state = updated_state
                    action = _extract_action(state)
                    if not action or action.get("action") == "terminate":
                        break
                    await execute_action(mcp, action)

                # Metrics
                dist = app_distribution(state["history"])
                H = entropy(dist)
                R = repetition_rate(state["history"])

                # Cleanup
                await mcp.call_tool("stop_all_apps")
                await mcp.call_tool("screen_off")

                return {
                    "status": "completed",
                    "app_distribution": dist,
                    "entropy": H,
                    "repetition_rate": R,
                    "steps_taken": state.get("steps_taken", 0),
                }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _normalize_apps(raw_apps) -> list[str]:
    if not raw_apps:
        return []
    if hasattr(raw_apps, "structuredContent"):
        data = raw_apps.structuredContent
        if isinstance(data, dict) and "apps" in data:
            return data["apps"]
    raise ValueError(f"Unexpected get_installed_apps result: {raw_apps}")


def _extract_action(state: dict):
    if state.get("plan_step") is not None:
        return state["plan_step"].get("current_action")
    elif state.get("record_step") is not None:
        return state["record_step"].get("current_action")
    return state.get("current_action")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
