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

Reflect realistic routines: post-workout → music, then social media, then camera or browsing or whatver you think is realistic.

If one app has dominated history, pick a new one.

Keep reasoning short and human-like.

Return valid JSON:
{{
"thought": "...",
"action": {{"app": "...", "action": "...", "args": {{}}}}
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

    # llm_model = ChatOpenAI(model="gpt-5-mini-2025-08-07")

    llm_model = ChatGroq(model="llama-3.3-70b-versatile", api_key=SecretStr(groq_api_key))

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
    prompt = _build_prompt(persona, installed_apps, history, available_actions)
    llm_response: ThoughtAction = _call_llm(prompt)

    # 1️⃣ Generate a base action plan
    # TODO: Improve planning -- low priority
    base_plan = build_action_plan(llm_response, persona)

    # 2️⃣ Prevent repetition (no more than twice in a row)
    recent_apps = [h["action"]["app"] for h in history[-3:] if "action" in h]
    overused = set(a for a in recent_apps if recent_apps.count(a) >= 2)
    candidate_actions = [a for a in base_plan if a["app"] not in overused]

    # 3️⃣ Fallback if all filtered
    if not candidate_actions:
        candidate_actions = base_plan

    # 4️⃣ Prefer apps that exist on the device
    # TODO: Make sure this works right
    valid_candidates = [
        a
        for a in candidate_actions
        if any(pkg_part in app for app in installed_apps for pkg_part in [a["app"]])
    ] or candidate_actions

    chosen_action = random.choice(valid_candidates)

    print(f"🤖 LLM Thought: {llm_response.thought}")
    print(f"🎯 Selected Action: {chosen_action}")

    return chosen_action
