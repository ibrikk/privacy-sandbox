from typing import Any, Dict, cast
from agents.persona_generator_agent.langgraph_persona_generator import app, GraphState
from agents.browsing_screenshots_agent.uiauto_agent.graph_agent import run_persona_session
        
def main():
    # Generate a persona based on the prompt
    # prompt: str = "Sarah, software engineer in San Francisco, jogging in Golden Gate Park."

# 🏃‍♀️ Physical / Movement-Oriented

    prompt = "Miguel, a 29-year-old product designer in Austin, jogging along Lady Bird Lake while listening to music."

    # “Daniel, a 41-year-old sales manager in Denver, hiking a trail outside the city on a Saturday afternoon.”

    # “Lena, a 26-year-old fitness instructor in Los Angeles, cooling down after a gym session.”

    # “Omar, a delivery driver in Chicago, driving through downtown traffic during rush hour.”

    # ☕ Stationary / Browsing / Work

    # “Emily, a freelance writer in Brooklyn, sitting in a coffee shop editing an article on her laptop.”

    # “Sofia, a UX researcher in San Jose, taking a short break between meetings at home.”

    # 🌆 Social / Leisure / Evening

    # “Maria, a public relations specialist in Miami, getting ready to meet friends for dinner.”

    # “Yuki, a graphic artist in Tokyo, unwinding at night while browsing social media.”

    # “Andre, a music producer in Atlanta, listening to playlists while relaxing on his couch.”

    # “Clara, a law student in Paris, watching short videos before going to sleep.”

    # 🧭 Travel / Navigation / Exploration

    # “Thomas, a consultant visiting New York City for work, searching for nearby lunch spots.”

    # “Fatima, a tourist in Barcelona, walking around the city center looking for attractions.”

    # 🛍️ Commerce / Errands

    # “Isabella, a fashion buyer in Milan, browsing clothing apps while commuting home.”

    # “Wei, a graduate student in Vancouver, comparing prices for electronics online.”
    

    # final_state = app.invoke(cast(GraphState, {"prompt": prompt}))
    
    # # Get save_dir from state and construct persona path
    # save_dir = final_state.get("save_dir")
    # if not save_dir:
    #     raise ValueError("save_dir not found in final state")
    
    # persona_path = f"{save_dir}/persona.json"
    persona_path = f"Miguel_Garcia_export/persona.json"
    print("🚀 Starting persona-driven UI automation session...")
    run_persona_session(persona_json_path=persona_path)

if __name__ == "__main__":
    main()
