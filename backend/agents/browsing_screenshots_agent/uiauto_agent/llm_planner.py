import os
from typing import List, Dict, Any, cast
import json, random

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, SecretStr
from langchain_core.output_parsers import PydanticOutputParser

from models import ThoughtAction

from .planner import build_action_plan

# Optional: if you want real LLM reasoning (currently simulated)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-5-mini-2025-08-07")

def _normalize_installed_apps(installed_apps: List[Any]) -> List[str]:
    """
    Normalize installed apps into a list of package-name strings.
    Accepts strings, tuples, or dicts (from MCP).
    """
    normalized = []
    for app in installed_apps:
        if isinstance(app, str):
            normalized.append(app)
        elif isinstance(app, dict):
            # MCP-style: {"package": "...", "label": "..."}
            if "package" in app:
                normalized.append(app["package"])
        elif isinstance(app, (list, tuple)):
            # Tuple-style: ("com.pkg.name", "Label")
            normalized.append(app[0])
    return normalized


def _build_prompt(persona, installed_apps, history, available_actions) -> str:
    installed_apps = _normalize_installed_apps(installed_apps)
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

Reflect realistic routines: post-workout → music, then social media, then camera or browsing, news, weather or whatver you think is realistic.

If one app has dominated history, pick a new one.

Keep reasoning short and human-like.

IMPORTANT:
- The "app" field MUST be one of the following Android package names exactly.
- Do NOT use short names like "spotify" or "youtube".

Valid apps:
com.spotify.music
com.facebook.katana
com.instagram.android
com.google.android.youtube

Allowed Actions:
com.spotify.music -> play_for_persona
com.facebook.katana -> open_and_browse, search_topic, maybe_post_status
com.instagram.android -> view_stories, view_reels
com.zhiliaoapp.musically -> watch_and_scroll
com.google.android.youtube -> watch_recommended

Return valid JSON:
{{
  "thought": "...",
  "action": {{
    "app": "<android package name>",
    "action": "<one of the allowed actions>",
    "args": {{}}
  }}
}}
""".strip()


def _call_llm(prompt: str) -> ThoughtAction:
    """
    Placeholder for LLM call — replace with real API if desired.
    """
    print("--- LLM PROMPT ---")
    print(prompt)
    print("------------------")

    groq_api_key = os.getenv("GROQ_API_KEY") or ""

    llm_model = ChatOpenAI(model="gpt-5-mini-2025-08-07")

    # llm_model = ChatGroq(model="llama-3.3-70b-versatile", api_key=SecretStr(groq_api_key))
    # llm_model = ChatGroq(model="llama-3.1-8b-instant", api_key=SecretStr(groq_api_key))

    # LangChain output parser for your Pydantic model
    parser = PydanticOutputParser(pydantic_object=ThoughtAction)

    # Define the prompt template
    prompt_template = ChatPromptTemplate.from_template(
        "You are a reasoning model. "
        "Given the user input, produce a structured ThoughtAction object.\n\n"
        "{format_instructions}\n\n"
        "User input:\n{user_input}"
    )

    formatted_prompt = prompt_template.format(
    format_instructions=parser.get_format_instructions(),
    user_input=prompt
    )
    
    response = llm_model.invoke(formatted_prompt)

    try:
        result = parser.parse(cast(str, response.content))
        return result
    except Exception as e:
        print("⚠️ Failed to parse LLM output:", e)
        print("Raw output:", response.content)
        raise

    # return json.dumps({"thought": thought, "action": {}})


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
    installed_apps = _normalize_installed_apps(installed_apps)
    print("INSTALLED APPS (planner):", installed_apps[:3])
    print("TYPE:", type(installed_apps[0]))
    prompt = _build_prompt(persona, installed_apps, history, available_actions)
    print("prompt: ", prompt)
    llm_response: ThoughtAction = _call_llm(prompt)

    # 1️⃣ Generate a base action plan
    base_plan: List[Dict[str, Any]] = build_action_plan(llm_response, persona, installed_apps)
    
    # for i in base_plan:
    #     if "systemui" in i["app"] or "camera" in i["app"]:
    #         base_plan.remove(i)

    # Filter invalid apps
    recent_apps = [h["action"]["app"] for h in history[-3:] if "action" in h]
    overused = set(a for a in recent_apps if recent_apps.count(a) >= 2)
    candidate_actions = (
        [a for a in base_plan if a["app"] not in overused]
        if overused else base_plan
    )

    # Ensure app is installed
    valid_candidates = [
        a for a in candidate_actions
        if any(a["app"] in pkg for pkg in installed_apps)
    ] or candidate_actions

    chosen_action = random.choice(valid_candidates)

    print(f"🤖 LLM Thought: {llm_response.thought}")
    print(f"🎯 Selected Action: {chosen_action}")

    return chosen_action
