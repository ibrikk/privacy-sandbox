import asyncio
from agents.browsing_screenshots_agent.uiauto_agent.metrics import app_distribution, entropy, repetition_rate
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from typing import Any, Dict, cast
from agents.persona_generator_agent.langgraph_persona_generator import app, GraphState
from agents.browsing_screenshots_agent.uiauto_agent.graph_agent import (AgentState, initialize_agent_state, app as graph_app)
from agents.browsing_screenshots_agent.uiauto_agent.executor import execute_action
from models import UserInput
        
async def main():
    # Generate a persona based on the prompt

    prompt: Dict[str, str] = {
        "age": "28",
        "city": "San Francisco",
        "job": "software engineer",
        "context": "home",
        "activity_state": "active",
        "day_type": "weekday",
        "hour_of_day": "23",
        "chronotype_self_report": "neutral",
        "phone_style": "mixed",
        "primary_use": "social",
        "usage_level": "typical",
    }

    if prompt is None:
        raise ValueError("prompt is required")
    
    persona_prompt = UserInput(**prompt)
    final_state = app.invoke(cast(GraphState, {"prompt": persona_prompt}))
    
    # Get save_dir from state and construct persona path
    save_dir = final_state.get("save_dir")
    if not save_dir:
        raise ValueError("save_dir not found in final state")
    
    persona_path = f"{save_dir}/persona.json"
    # persona_path = f"Miguel_Garcia_export/persona.json"
    print("🚀 Starting persona-driven UI automation session...")
    
    MCP_URL = "http://localhost:8080/mcp"

    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as mcp:
            
            await mcp.initialize()
            
            # 1. Get installed apps from device
            raw_apps = await mcp.call_tool("get_installed_apps")

            def normalize_apps(raw_apps) -> list[str]:
                """
                Normalize MCP get_installed_apps CallToolResult into List[str].
                """
                if not raw_apps:
                    return []

                if hasattr(raw_apps, "structuredContent"):
                    data = raw_apps.structuredContent
                    if isinstance(data, dict) and "apps" in data:
                        return data["apps"]

                raise ValueError(f"Unexpected get_installed_apps result: {raw_apps}")


            installed_apps = normalize_apps(raw_apps)
            
            # 2. Initialize planning state
            state: AgentState = initialize_agent_state(
                persona_json_path=persona_path,
                installed_apps=installed_apps,
                max_steps=10,
            )
            
            state = state.copy()
            

            # 3. Run LangGraph
            async for updated_state in graph_app.astream(state):
                state = updated_state
                action = None
                if state.get("plan_step") is not None:
                    action = state["plan_step"]["current_action"]
                else:
                    action = state["record_step"]["current_action"]
                if not action or action.get("action") == "terminate":
                    break
                # 4. Execute via MCP
                await execute_action(mcp, action)
            
            dist = app_distribution(state["history"])
            H = entropy(dist)
            R = repetition_rate(state["history"])

            print(f"🔍 App Distribution: {dist}")
            print(f"🔍 Entropy: {H}")
            print(f"🔍 Repetition Rate: {R}")


            # 5. Cleanup
            await mcp.call_tool("stop_all_apps")
            await mcp.call_tool("screen_off")
            print("✅ Session finished")

if __name__ == "__main__":
    asyncio.run(main())
