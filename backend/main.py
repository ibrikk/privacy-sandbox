# main.py - simplified, uses same logic as server

import asyncio
import argparse
from typing import cast

from models import UserProfileInput
from agents.persona_generator_agent.langgraph_persona_generator import app, GraphState


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic persona")
    parser.add_argument("--age", type=int, default=28)
    parser.add_argument("--city", type=str, default="San Francisco")
    parser.add_argument("--job", type=str, default="software engineer")
    parser.add_argument(
        "--chronotype", choices=["morning", "neutral", "night"], default="neutral"
    )
    parser.add_argument(
        "--phone-style",
        choices=["quick_checks", "mixed", "long_sessions"],
        default="mixed",
    )
    parser.add_argument(
        "--primary-use", choices=["social", "video_news", "mixed"], default="mixed"
    )
    parser.add_argument(
        "--usage-level", choices=["light", "typical", "heavy"], default="typical"
    )
    parser.add_argument(
        "--activity-level", choices=["low", "moderate", "high"], default="moderate"
    )
    parser.add_argument(
        "--commute-frequency",
        choices=["rare", "sometimes", "frequent"],
        default="sometimes",
    )
    parser.add_argument(
        "--routine-regularity", choices=["low", "medium", "high"], default="medium"
    )
    parser.add_argument(
        "--execute", action="store_true", help="Execute on device after generation"
    )
    parser.add_argument("--max-steps", type=int, default=10)
    return parser.parse_args()


async def main():
    args = parse_args()

    # Build UserProfileInput from CLI args
    profile = UserProfileInput(
        age=args.age,
        city=args.city,
        job=args.job,
        chronotype_self_report=args.chronotype,
        phone_style=args.phone_style,
        primary_use=args.primary_use,
        usage_level=args.usage_level,
        activity_level=args.activity_level,
        commute_frequency=args.commute_frequency,
        routine_regularity=args.routine_regularity,
    )

    print(
        f"📋 Generating persona for: {profile.job} in {profile.city}, age {profile.age}"
    )
    print(f"   Chronotype: {profile.chronotype_self_report}")
    print(f"   Usage: {profile.usage_level}, Style: {profile.phone_style}")

    # Run LangGraph pipeline
    final_state = app.invoke(cast(GraphState, {"prompt": profile}))

    save_dir = final_state.get("save_dir")
    if not save_dir:
        raise ValueError("Pipeline failed: save_dir not found")

    persona_path = f"{save_dir}/persona.json"
    print(f"✅ Persona saved to: {persona_path}")

    # Print key parameters
    params = final_state.get("parameters")
    if params:
        print(f"\n📊 Key Parameters:")
        print(f"   Daily usage: {params.total_daily_usage_minutes:.0f} min")
        print(f"   Pickups/hour: {params.baseline_pickups_per_hour:.1f}")
        print(f"   Avg session: {params.avg_session_duration_seconds:.0f} sec")
        print(f"   Glance prob: {params.glance_probability:.0%}")

    # Optional device execution
    if args.execute:
        print("\n🚀 Starting device execution...")
        await execute_on_device(persona_path, args.max_steps)


async def execute_on_device(persona_path: str, max_steps: int):
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

    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as mcp:
            await mcp.initialize()

            raw_apps = await mcp.call_tool("get_installed_apps")
            installed_apps = _normalize_apps(raw_apps)

            state: AgentState = initialize_agent_state(
                persona_json_path=persona_path,
                installed_apps=installed_apps,
                max_steps=max_steps,
            )

            async for updated_state in graph_app.astream(state):
                state = updated_state
                action = state.get("current_action")
                if not action or action.get("action") == "terminate":
                    break
                await execute_action(mcp, action)

            dist = app_distribution(state["history"])
            print(f"\n🔍 Results:")
            print(f"   App Distribution: {dist}")
            print(f"   Entropy: {entropy(dist):.2f}")
            print(f"   Repetition Rate: {repetition_rate(state['history']):.2%}")

            await mcp.call_tool("stop_all_apps")
            await mcp.call_tool("screen_off")
            print("✅ Session finished")


def _normalize_apps(raw_apps) -> list[str]:
    if not raw_apps:
        return []
    if hasattr(raw_apps, "structuredContent"):
        data = raw_apps.structuredContent
        if isinstance(data, dict) and "apps" in data:
            return data["apps"]
    return []


if __name__ == "__main__":
    asyncio.run(main())
