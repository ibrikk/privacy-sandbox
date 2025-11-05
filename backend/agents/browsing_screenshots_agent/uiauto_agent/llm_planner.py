from typing import List, Dict, Any
import json, random
from .planner import build_action_plan

# Optional: if you want real LLM reasoning (currently simulated)
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-5-mini-2025-08-07")


def _build_prompt(persona, installed_apps, history, available_actions) -> str:
    persona_details = json.dumps(persona.__dict__, indent=2)
    return f"""
You are an expert Android user emulating a specific persona to test mobile app privacy.
Behave naturally, choosing the next app interaction that fits the persona's lifestyle.

**Persona Details:**
```json
{persona_details}
Installed Third-party Apps: {", ".join(installed_apps)}

Recent Session History (last 5 actions):
{json.dumps(history[-5:], indent=2)}

Available Actions:
{", ".join(available_actions)}

Guidelines:

Avoid repeating the same app more than twice in a row.

Alternate between music, social, and camera interactions when possible.

Reflect realistic routines: post-workout → music, then social media, then camera or browsing.

If one app has dominated history, pick a new one.

Keep reasoning short and human-like.

Return valid JSON:
{{
"thought": "...",
"action": {{"app": "...", "action": "...", "args": {{}}}}
}}
""".strip()


def _call_llm(prompt: str) -> str:
    """
    Placeholder for LLM call — replace with real API if desired.
    """
    print("--- LLM PROMPT ---")
    print(prompt)
    print("------------------")

    # Uncomment below for real LLM reasoning (OpenAI example):
    # response = llm.invoke(prompt)
    # thought = response.content.strip()

    thought = "Simulated reasoning: switching apps for variety and realistic behavior."
    return json.dumps({"thought": thought, "action": {}})


def get_next_action(
    persona,
    installed_apps: List[str],
    history: List[Dict[str, Any]],
    available_actions: List[str],
) -> Dict[str, Any]:
    """
    Combines deterministic planning with LLM-like reasoning.
    Prevents repetitive app use and adds diversity to persona behavior.
    """
    prompt = _build_prompt(persona, installed_apps, history, available_actions)
    llm_response = json.loads(_call_llm(prompt))
    thought = llm_response.get("thought", "")

    # 1️⃣ Generate a base action plan
    base_plan = build_action_plan(persona)

    # 2️⃣ Prevent repetition (no more than twice in a row)
    recent_apps = [h["action"]["app"] for h in history[-3:] if "action" in h]
    overused = set(a for a in recent_apps if recent_apps.count(a) >= 2)
    candidate_actions = [a for a in base_plan if a["app"] not in overused]

    # 3️⃣ Fallback if all filtered
    if not candidate_actions:
        candidate_actions = base_plan

    # 4️⃣ Prefer apps that exist on the device
    valid_candidates = [
        a
        for a in candidate_actions
        if any(pkg_part in app for app in installed_apps for pkg_part in [a["app"]])
    ] or candidate_actions

    chosen_action = random.choice(valid_candidates)

    print(f"🤖 LLM Thought: {thought}")
    print(f"🎯 Selected Action: {chosen_action}")

    return chosen_action
