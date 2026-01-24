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
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-5-mini-2025-08-07")

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


def _build_prompt(
    persona,
    installed_apps,
    history,
    available_actions,
    banned_apps: set[str],
) -> str:
    persona_details = json.dumps(persona.__dict__, indent=2)

    prompt = f"""
    You are an expert Android user emulating a specific persona to test mobile app privacy.

    Persona Details:
    {persona_details}

    Installed Apps:
    {", ".join(installed_apps)}

    Recent History (last 5 actions):
    {json.dumps(history[-5:], indent=2)}

    Available Actions:
    {", ".join(available_actions)}

    Behavior Rules:
    - Avoid repeating the same app more than twice in a row
    - Alternate between music, social, and browsing
    - Behave naturally and human-like
    """

    # 🔥 HARD COOLDOWN RULES (this is what stops Spotify spam)
    if banned_apps:
        prompt += "\nFORBIDDEN APPS (HARD COOLDOWN):\n"
        for app in banned_apps:
            prompt += f"- {app}\n"
        prompt += (
            "\nYou MUST NOT choose any forbidden app. "
            "Selecting a forbidden app is an error.\n"
        )

    prompt += """
Valid apps:
- com.facebook.katana
- com.instagram.android
- com.spotify.music
- com.google.android.youtube
- com.linkedin.android

Allowed Actions:
com.facebook.katana -> open_and_browse, search_topic, maybe_post_status
com.instagram.android -> view_stories, view_reels
com.spotify.music -> play_for_persona
com.linkedin.android -> open_and_browse
com.google.android.youtube -> watch_recommended
com.google.android.apps.youtube.music -> watch_recommended

Return VALID JSON ONLY:
{
  "thought": "...",
  "action": {
    "app": "<android package name>",
    "action": "<allowed action>",
    "args": {}
  }
}
"""

    return prompt.strip()



def _call_llm(prompt: str) -> ThoughtAction:
    """
    Placeholder for LLM call — replace with real API if desired.
    """
    print("--- LLM PROMPT ---")
    print(prompt)
    print("------------------")

    groq_api_key = os.getenv("GROQ_API_KEY") or ""

    # llm_model = ChatOpenAI(model="gpt-5-mini-2025-08-07")

    llm_model = ChatGroq(model="llama-3.3-70b-versatile", api_key=SecretStr(groq_api_key))
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

def is_on_hard_cooldown(app: str, history: list, window: int = 2) -> bool:
    recent_apps = [
        h["action"]["app"]
        for h in history[-window:]
        if "action" in h
    ]
    return app in recent_apps


def get_next_action(
    persona,
    installed_apps: List[str],
    history: List[Dict[str, Any]],
    available_actions: List[str],
) -> Dict[str, Any]:

    installed_apps = _normalize_installed_apps(installed_apps)

    # ✅ ALWAYS define it first
    banned_apps: set[str] = set()

    # Hard cooldown logic
    if is_on_hard_cooldown("com.spotify.music", history, window=2):
        banned_apps.add("com.spotify.music")

    if banned_apps:
        print("⛔ Hard cooldown active for:", banned_apps)

    # Now it's safe to use
    prompt = _build_prompt(
        persona,
        installed_apps,
        history,
        available_actions,
        banned_apps,
    )

    llm_response: ThoughtAction = _call_llm(prompt)

    base_plan = build_action_plan(llm_response, persona, installed_apps)

    valid_actions = [
        a for a in base_plan if a["app"] not in banned_apps
    ] or base_plan

    chosen_action = random.choice(valid_actions)

    print(f"🤖 Thought: {llm_response.thought}")
    print(f"🎯 Action: {chosen_action}")

    return chosen_action

