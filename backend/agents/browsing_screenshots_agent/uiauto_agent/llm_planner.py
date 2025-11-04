# uiauto_agent/llm_planner.py
from typing import List, Dict, Any
import json

# This is a placeholder for a real LLM client (like OpenAI, Anthropic, etc.)
# You would replace this with your actual client initialization.
# from openai import OpenAI
# client = OpenAI(api_key="YOUR_API_KEY")

def _call_llm(prompt: str) -> str:
    """
    Placeholder function for a real LLM API call.
    It returns a JSON string representing a plausible next action.
    """
    print("--- LLM PROMPT ---")
    print(prompt)
    print("------------------")
    
    # In a real implementation, you would make the API call here:
    # response = client.chat.completions.create(
    #     model="gpt-4-turbo",
    #     messages=[{"role": "user", "content": prompt}],
    #     response_format={"type": "json_object"},
    # )
    # return response.choices[0].message.content
    
    # For now, returning a hardcoded example action
    return json.dumps({
        "thought": "The persona is a software engineer in SF who runs in the mornings. After a run, they might check social media. I'll have them browse Facebook.",
        "action": {
            "app": "facebook",
            "action": "open_and_browse",
            "args": {}
        }
    })

def _build_prompt(persona, installed_apps: List[str], history: List[Dict[str, Any]], available_actions: List[str]) -> str:
    persona_details = json.dumps(persona.__dict__, indent=2)
    
    prompt = f"""
You are an expert Android user emulating a specific persona to test application privacy.
Your goal is to behave exactly as the persona would, interacting with apps on the device in a realistic sequence.

**Persona Details:**
```json
{persona_details}
```

**Device State:**
- Installed third-party apps: {', '.join(installed_apps)}
- Recent actions taken in this session: {json.dumps(history, indent=2)}

**Available Actions:**
You can perform any of the following actions:
{', '.join(available_actions)}

**Your Task:**
Based on the persona and the session history, decide the single next action to take.
First, think step-by-step about what this persona would do right now.
Then, provide your final decision as a JSON object with two keys: "thought" and "action".
The "action" value must be another JSON object with "app", "action", and "args" keys.

Example response format:
{{
  "thought": "The persona is a student who likes music. They would probably listen to a study playlist on Spotify.",
  "action": {{
    "app": "spotify",
    "action": "play_for_persona",
    "args": {{}}
  }}
}}

Now, determine the next action for the given persona.
"""
    return prompt.strip()

def get_next_action(persona, installed_apps: List[str], history: List[Dict[str, Any]], available_actions: List[str]) -> Dict[str, Any]:
    """
    Builds a prompt and calls the LLM to get the next action plan.
    """
    prompt = _build_prompt(persona, installed_apps, history, available_actions)
    response_str = _call_llm(prompt)
    
    try:
        response_json = json.loads(response_str)
        print(f"🤖 LLM Thought: {response_json.get('thought')}")
        return response_json.get("action", {})
    except (json.JSONDecodeError, KeyError) as e:
        print(f"❌ Error parsing LLM response: {e}")
        print(f"Raw response: {response_str}")
        return {}
