from typing import Any, Dict, cast
from agents.persona_generator_agent.langgraph_persona_generator import app, GraphState
from agents.browsing_screenshots_agent.uiauto_agent.graph_agent import run_persona_session
        
def main():
    prompt: str = "Sarah, software engineer in San Francisco, jogging in Golden Gate Park."
    app.invoke(cast(GraphState, {"prompt": prompt}))
    
    persona_path = "Sarah_Lee_export/persona.json"  # adjust path
    print("🚀 Starting persona-driven UI automation session...")
    run_persona_session(persona_json_path=persona_path)

if __name__ == "__main__":
    main()
