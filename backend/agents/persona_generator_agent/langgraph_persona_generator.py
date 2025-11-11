import datetime
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, TypedDict, Union, cast
import uuid

from dotenv import load_dotenv
from geopy import Location
from geopy.geocoders import Nominatim
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, SecretStr
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from models import PersonaPackage, PrivacyAttributes

system_message = (
    "You are a privacy and behavior modeling expert generating lifelike human personas for mobile privacy simulations. "
    "Each persona should be realistic, contextually grounded, and internally consistent across demographics, occupation, "
    "lifestyle, and daily activity.\n\n"
    "### Your reasoning steps (to yourself before output)\n"
    "1. Read the user prompt carefully to understand who the person is and what they're doing.\n"
    "2. Infer missing realistic details (e.g., if a 28-year-old woman is 'running in Golden Gate Park', she probably works in tech, lives near San Francisco, and earns a mid-to-high salary).\n"
    "3. Make sure demographic, income, and lifestyle attributes align logically with each other. And age matches the birthday\n"
    "4. Ensure diversity, neutrality, and privacy-awareness — avoid bias or stereotypes.\n"
    "5. Finally, output a complete and clean PrivacyAttributes object, with all required fields filled.\n\n"
    "### Example 1 (for inspiration)\n"
    "Prompt: 'A 30-year-old man named David commuting to work in Seattle.'\n"
    "→ Persona: David Chen, age 30, male, Asian, lives in Seattle, WA. Bachelor's in Computer Engineering. "
    "Software developer at Amazon, income $125,000/year, single, renter in downtown. "
    "Online behavior: reads Reddit tech forums, moderate app usage, privacy-conscious. "
    "Activity: 'commuting on the light rail while browsing phone notifications.'\n\n"
    "### Example 2\n"
    "Prompt: 'A 42-year-old woman named Alicia having coffee before work in Chicago.'\n"
    "→ Persona: Alicia Torres, age 42, female, Hispanic, lives in Chicago, IL. MBA, marketing director at a healthcare firm, "
    "income $145,000/year, married with grade-school children, homeowner in Oak Park. "
    "Online behavior: uses LinkedIn and Facebook daily, privacy-indifferent. "
    "Activity: 'sitting at a cafe checking emails.'\n\n"
    "### Output rules\n"
    "- Always respond with a structured PrivacyAttributes object.\n"
    "- Do NOT include reasoning or intermediate text.\n"
    "- Do NOT mention sensors or devices — only human and contextual fields.\n"
    "- Use realistic, diverse, non-stereotypical details.\n"
    "- Output must be consistent and human-like, suitable for downstream sensor spoofing simulation."
)


from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") or ""
os.environ["LANGCHAIN_SANDBOX_V1"] = "true"
os.environ["LANGCHAIN_SANDBOX"] = os.getenv("LANGCHAIN_PROJECT") or ""

# llm_model = ChatOpenAI(model="gpt-5-mini-2025-08-07")

groq_api_key = os.getenv("GROQ_API_KEY") or ""
llm_model: Any = ChatGroq(model="llama-3.3-70b-versatile", api_key=SecretStr(groq_api_key))


# ----------------------
# Sensor Schema Definitions
# ----------------------


class EmptyMoment(BaseModel):
    pass


class BaseStationTrace(BaseModel):
    id: str = Field(description="The unique identifier for the base station trace")
    time: int = Field(description="The timestamp of the base station trace")
    moments: List[EmptyMoment] = Field(
        default_factory=list, description="The moments of the base station trace"
    )


class SensorMoment(BaseModel):
    elapsed: float = Field(description="The elapsed time of the sensor moment")
    data: Optional[Dict[str, List[float]]] = Field(
        description="The data of the sensor moment"
    )


class SensorTrace(BaseModel):
    id: str = Field(description="The unique identifier for the sensor trace")
    time: int = Field(description="The timestamp of the sensor trace")
    moments: List[SensorMoment] = Field(
        default_factory=list, description="The moments of the sensor trace"
    )
    sensorsInvolved: List[int] = Field(
        default_factory=list, description="The sensors involved in the sensor trace"
    )


class DrawPoint(BaseModel):
    latitude: float = Field(description="The latitude of the draw point")
    longitude: float = Field(description="The longitude of the draw point")


class DrawTrace(BaseModel):
    id: str = Field(description="The unique identifier for the draw trace")
    name: str = Field(description="The name of the draw trace")
    coordSys: str = Field(description="The coordinate system of the draw trace")
    points: List[DrawPoint] = Field(
        default_factory=list, description="The points of the draw trace"
    )

# -----------------------------------------
# Generators
# -----------------------------------------


class PersonaGenerator:
    def __init__(self):
        self.llm = llm_model

    def generate(self, messages: List[BaseMessage]) -> PrivacyAttributes:
        structured_llm = self.llm.with_structured_output(PrivacyAttributes)
        return structured_llm.invoke(messages)


class SensorTraceGenerator:
    def __init__(self, persona: PrivacyAttributes, duration_s=60, fps=30):
        self.persona = persona
        self.duration_s = duration_s
        self.fps = fps
        # From Android documentation
        self.sensor_id_map = {
            "accelerometer": 1,
            "magnetic_field": 2,
            "gyroscope": 4,
            "light": 5,
            "linear_acceleration": 10,
            "magnetic_field_uncalibrated": 14,
            "gyroscope_uncalibrated": 16,
            "step_detector": 18,
            "step_counter": 19,
            "accelerometer_uncalibrated": 35,
        }
        self.activity_sensor_map = {
            "running": {
                "accelerometer": {"mean": 5.0, "base_drift": 0.8, "bias_drift": 0.3},
                "gyroscope": {"mean": 2.2, "base_drift": 0.5, "bias_drift": 0.2},
                "step_counter": {"mean": 1.8, "base_drift": 0.2},
                "step_detector": {"mean": 1.0, "base_drift": 0.05},
                "linear_acceleration": {"mean": 4.0, "base_drift": 1.0},
                "accelerometer_uncalibrated": {
                    "mean": 5.0,
                    "base_drift": 1.2,
                    "bias_drift": 0.5,
                },
                "gyroscope_uncalibrated": {
                    "mean": 2.0,
                    "base_drift": 0.6,
                    "bias_drift": 0.25,
                },
            },
            "sitting": {
                "accelerometer": {"mean": 0.1, "base_drift": 0.02, "bias_drift": 0.01},
                "gyroscope": {"mean": 0.05, "base_drift": 0.01},
                "linear_acceleration": {"mean": 0.1, "base_drift": 0.03},
                "light": {"mean": 300.0, "base_drift": 80.0},
                "accelerometer_uncalibrated": {
                    "mean": 0.2,
                    "base_drift": 0.05,
                    "bias_drift": 0.03,
                },
                "magnetic_field_uncalibrated": {
                    "mean": 45.0,
                    "base_drift": 5.0,
                    "bias_drift": 1.5,
                },
            },
            "commuting": {
                "accelerometer": {"mean": 1.5, "base_drift": 0.4, "bias_drift": 0.2},
                "gyroscope": {"mean": 0.6, "base_drift": 0.2, "bias_drift": 0.1},
                "magnetic_field": {"mean": 40, "base_drift": 8},
                "magnetic_field_uncalibrated": {
                    "mean": 42,
                    "base_drift": 10,
                    "bias_drift": 2,
                },
                "light": {"mean": 250, "base_drift": 60},
                "linear_acceleration": {"mean": 1.2, "base_drift": 0.4},
            },
            "driving": {
                "accelerometer": {"mean": 1.8, "base_drift": 0.6, "bias_drift": 0.3},
                "linear_acceleration": {
                    "mean": 1.5,
                    "base_drift": 0.5,
                    "bias_drift": 0.25,
                },
                "gyroscope": {"mean": 0.8, "base_drift": 0.4, "bias_drift": 0.2},
                "gyroscope_uncalibrated": {
                    "mean": 0.9,
                    "base_drift": 0.45,
                    "bias_drift": 0.25,
                },
                "magnetic_field": {"mean": 55.0, "base_drift": 10.0},
                "magnetic_field_uncalibrated": {
                    "mean": 60.0,
                    "base_drift": 12.0,
                    "bias_drift": 3.0,
                },
                "light": {"mean": 300.0, "base_drift": 150.0},  # day/night variation
                "step_counter": {"mean": 0.0, "base_drift": 0.0},
                "step_detector": {"mean": 0.0, "base_drift": 0.0},
            },
        }
        self.activity_aliases = {
            "running": ["running", "jogging", "sprinting", "trail running"],
            "sitting": [
                "sitting",
                "resting",
                "reading",
                "working on laptop",
                "typing",
                "watching tv",
            ],
            "commuting": ["bus", "subway", "train", "riding", "on the metro"],
            "driving": [
                "driving",
                "in a car",
                "in vehicle",
                "road trip",
                "stuck in traffic",
            ],
            "walking": ["walking", "strolling", "shopping", "browsing"],
            "sleeping": ["sleeping", "lying down", "napping", "resting in bed"],
        }

    def _normalize_activity(self, text: str) -> str:
        text = text.lower()
        for canonical, variants in self.activity_aliases.items():
            if any(v in text for v in variants):
                return canonical
        return "sitting"

    def _slow_bias(self, elapsed: float, amp: float = 0.2):
        return math.sin(elapsed / 20.0) * amp + random.gauss(0, amp / 4)

    def _activity_modulation(self, activity: str, elapsed: float) -> float:
        if activity == "running":
            return 1.0 + 0.8 * abs(math.sin(elapsed * 2.5))
        elif activity == "commuting":
            return 1.0 + 0.3 * math.sin(elapsed / 2.5)
        elif activity == "driving":
            return 1.0 + 0.3 * math.sin(elapsed / 3.0) + random.gauss(0, 0.05)
        elif activity == "sitting":
            return 1.0 + random.gauss(0, 0.01)
        return 1.0

    def generate(self) -> SensorTrace:
        id_str = str(uuid.uuid4())
        start_time = int(time.time() * 1000)
        total_frames = int(self.duration_s * self.fps)
        activity = self._normalize_activity(self.persona.activity_description)
        traits = self.activity_sensor_map.get(activity, {})

        uncalibrated = {
            "accelerometer_uncalibrated",
            "magnetic_field_uncalibrated",
            "gyroscope_uncalibrated",
        }
        single_val = {"light", "step_counter", "step_detector"}

        moments, sensors_used = [], []

        for i in range(total_frames):
            elapsed = i / self.fps
            frame_data = {}

            for sensor, stats in traits.items():
                if sensor not in self.sensor_id_map:
                    continue
                sensors_used.append(self.sensor_id_map[sensor])
                mean = stats.get("mean", 0.0)
                base_drift = stats.get("base_drift", 0.05)
                bias_drift = stats.get("bias_drift", base_drift / 2)
                bias = self._slow_bias(elapsed, amp=bias_drift)
                noise = random.gauss(0, base_drift)
                mod = self._activity_modulation(activity, elapsed)
                value = (mean + bias + noise) * mod

                if sensor in uncalibrated:
                    data = [
                        round(value + random.gauss(0, 0.05), 3),
                        round(value + random.gauss(0, 0.05), 3),
                        round(value + random.gauss(0, 0.05), 3),
                        round(bias, 3),
                        round(bias / 2, 3),
                        round(bias / 3, 3),
                    ]
                elif sensor in single_val:
                    data = [round(value, 3)]
                else:
                    data = [
                        round(value + random.gauss(0, 0.05), 3),
                        round(value + random.gauss(0, 0.05), 3),
                        round(value + random.gauss(0, 0.05), 3),
                    ]

                frame_data[str(self.sensor_id_map[sensor])] = data

            moments.append(SensorMoment(elapsed=elapsed, data=frame_data))

        return SensorTrace(
            id=id_str,
            time=start_time,
            moments=moments,
            sensorsInvolved=list(set(sensors_used)),
        )


class BaseStationTraceGenerator:
    def generate(self) -> BaseStationTrace:
        return BaseStationTrace(
            id=str(uuid.uuid4()),
            time=int(time.time() * 1000),
            moments=[],  # SIM-less fallback
        )


class DrawTraceGenerator:
    def __init__(
        self, persona: PrivacyAttributes, duration_s: int = 600, freq_hz: float = 0.2
    ):
        """
        freq_hz = 0.2 → one GPS point every 5 seconds
        duration_s = total simulated time (e.g., 600 = 10 minutes)
        """
        self.persona = persona
        self.steps = int(duration_s * freq_hz)

    def generate(self) -> DrawTrace:
        points = []
        geolocator: Nominatim = Nominatim(user_agent="persona_generator")
        location: Location = cast(Location, geolocator.geocode(f"{self.persona.city}, {self.persona.state}"))

        if location:
            lat, lon = location.latitude, location.longitude
        else:
            lat, lon = 39.2904, -76.6122  # fallback → Baltimore downtown

        # Simulate directionally consistent motion
        lat_drift = random.uniform(0.00005, 0.00015)
        lon_drift = random.uniform(0.00005, 0.00015)

        for i in range(self.steps):
            # periodic oscillation + jitter + directional drift
            lat += (
                lat_drift
                + math.sin(i / 15) * 0.00005
                + random.uniform(-0.00003, 0.00003)
            )
            lon += (
                lon_drift
                + math.cos(i / 15) * 0.00005
                + random.uniform(-0.00003, 0.00003)
            )

            # every ~50 steps, small random direction change
            if i % 50 == 0:
                lat_drift, lon_drift = lon_drift, -lat_drift

            points.append(DrawPoint(latitude=round(lat, 6), longitude=round(lon, 6)))

        return DrawTrace(
            id=str(uuid.uuid4()),
            name=f"{self.persona.first_name}_{self.persona.last_name}_trace",
            coordSys="WGS84",
            points=points,
        )


# -----------------------------------------
# LangGraph Implementation
# -----------------------------------------


class GraphState(TypedDict):
    prompt: str
    privacy_attrs: Optional[PrivacyAttributes]
    traces: Optional[Dict[str, BaseModel]]
    persona_package: Optional[PersonaPackage]
    save_dir: Optional[str]


def _ensure_privacy_attrs(attrs: Optional[PrivacyAttributes]) -> PrivacyAttributes:
    if not attrs:
        raise ValueError("privacy_attrs not found. Run generate_privacy_attrs first.")
    return attrs


def _ensure_traces(traces: Optional[Dict[str, BaseModel]]) -> Dict[str, BaseModel]:
    if not traces:
        raise ValueError("traces not found. Run generate_traces next.")
    return traces


# 1) Generate PrivacyAttributes ONLY
def generate_privacy_attrs_node(state: GraphState) -> Dict[str, Any]:
    prompt = state["prompt"]
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=prompt),
    ]
    persona_generator = PersonaGenerator()
    attrs = persona_generator.generate(messages)

    return {"privacy_attrs": attrs}


def persona_grader_node(state: GraphState) -> Dict[str, Any]:
    """
    Grade persona consistency and decide next step.
    Returns either 'redo' (go back to regenerate) or 'ok' (proceed).
    """
    persona: PrivacyAttributes = cast(PrivacyAttributes, state["privacy_attrs"])
    report = []
    score = 100

    # --- AGE ↔ BIRTHDAY check ---
    try:
        birth_year = datetime.datetime.strptime(persona.birthday, "%Y-%m-%d").year
        current_year = datetime.datetime.now().year
        derived_age = current_year - birth_year
        if abs(derived_age - int(persona.age)) > 1:
            persona.age = f"{derived_age}"
        if abs(derived_age - int(persona.age)) > 1:
            report.append(
                f"Age mismatch: derived {derived_age} vs stated {persona.age}"
            )
            score -= 15
    except Exception as e:
        report.append(f"Invalid birthday format: {e}")
        score -= 20

    # --- ZIP ↔ format ---
    if persona.zip_code:
        if not persona.zip_code.isdigit() or len(persona.zip_code) not in (5, 9):
            report.append("ZIP code format invalid")
            score -= 10

    # --- Occupation ↔ Income sanity ---
    if persona.job and persona.income:
        occ = persona.job.lower()
        try:
            income_val = int(str(persona.income).replace(",", "").replace("$", ""))
        except ValueError:
            income_val = 0

        if ("intern" in occ or "assistant" in occ) and income_val > 80000:
            report.append("Income too high for entry-level occupation")
            score -= 10
        elif (
            "director" in occ or "vp" in occ or "founder" in occ
        ) and income_val < 70000:
            report.append("Income too low for senior occupation")
            score -= 10

    # --- ADDRESS completeness ---
    missing = [f for f in [persona.city, persona.state] if not f]
    if missing:
        report.append("Incomplete address info")
        score -= 5

    # --- Decision ---
    valid = score >= 75 and len(report) <= 2
    decision = "proceed" if valid else "redo"
    print(f"\n🧩 Persona Grader Results:\n  Score: {score}\n  Issues: {report}\n")
    return {"decision": decision}


# 2) Generate traces independently
def generate_traces_node(state: GraphState) -> Dict[str, Any]:
    attrs = _ensure_privacy_attrs(state.get("privacy_attrs"))

    sensor_trace = SensorTraceGenerator(attrs).generate()
    base_trace = BaseStationTraceGenerator().generate()
    draw_trace = DrawTraceGenerator(attrs).generate()

    return {
        "traces": {
            "sensor_trace": sensor_trace,
            "base_station_trace": base_trace,
            "draw_trace": draw_trace,
        }
    }


def grade_traces_node(state: GraphState) -> Dict[str, Any]:
    traces: Optional[Dict[str, BaseModel]] = state.get("traces", {})
    persona = state.get("privacy_attrs", {})

    grader_prompt = (
        "You are a motion-data realism evaluator. "
        "Your job is to rate the believability of synthetic mobile sensor traces on a 0–10 scale.\n\n"
        "Interpret the scale as follows:\n"
        "  0–2 : Completely unrealistic, random or constant values.\n"
        "  3–5 : Somewhat plausible, noisy but inconsistent with real movement.\n"
        "  6–8 : Generally realistic with coherent temporal variation.\n"
        "  9–10: Indistinguishable from genuine human sensor data.\n\n"
        "Example A (clearly fake): accelerometer values all near 0, no variation → score: 1.\n"
        "Example B (moderate realism): accelerometer fluctuates smoothly 0–2, gyroscope small periodic drift → score: 6.\n"
        "Example C (very realistic): complex correlated motion with noise, GPS path continuous → score: 9.\n\n"
        "Be objective but not cynical; many good synthetic traces should earn 6–8.\n"
        "Output only a single integer 0–10.\n\n"
        f"Persona summary: {getattr(persona, 'first_name', '')} {getattr(persona, 'last_name', '')}, "
        f"activity={getattr(persona, 'activity_description', '')}, job={getattr(persona, 'job', '')}, "
        f"city={getattr(persona, 'city', '')}\n\n"
        f"Sensor snippet: {getattr(getattr(traces, 'get', lambda k, d=None: None)('sensor_trace', None), 'moments', [])[:3]}\n"
        f"GPS snippet: {getattr(getattr(traces, 'get', lambda k, d=None: None)('draw_trace', None), 'points', [])[:3]}\n"
    )

    response = llm_model.invoke(grader_prompt)
    text = response.content.strip()
    try:
        score = int("".join([ch for ch in text if ch.isdigit()]))
    except ValueError:
        score = 0

    decision = "proceed" if score >= 7 else "redo"
    print(f"\n🤖 LLM Trace Score: {score}/10\nDecision: {decision}\n")
    return {"decision": decision, "score": score}


# 3) Package persona and traces together
def package_persona_node(state: GraphState) -> Dict[str, Any]:
    attrs = _ensure_privacy_attrs(state.get("privacy_attrs"))
    traces = _ensure_traces(state.get("traces"))

    package = PersonaPackage(persona=attrs, traces=traces)

    return {"persona_package": package}


# 4) Save everything from the package
def save_package_node(state: GraphState) -> Dict[str, Any]:
    package = state.get("persona_package")
    if not package:
        raise ValueError("persona_package not found. Run package_persona_node first.")

    attrs = package.persona
    traces = package.traces
    export_root = f"{attrs.first_name}_{attrs.last_name}_export"
    os.makedirs(export_root, exist_ok=True)

    # --- Save standalone persona.json (do NOT include in tar) ---
    persona_path = os.path.join(export_root, "persona.json")
    try:
        with open(persona_path, "w") as pf:
            json.dump(attrs.model_dump(), pf, indent=2)
        print(f"🧍 Persona JSON saved (not included in tar): {persona_path}")
    except Exception as e:
        print(f"⚠️ Failed to save persona.json: {e}")

    # 1️⃣ Motion file (sensor data)
    sensor_trace = traces.get("sensor_trace")
    if isinstance(sensor_trace, SensorTrace):
        motion_json = {
            "id": str(uuid.uuid4())[:20],
            "time": int(time.time() * 1000),
            "moments": [m.model_dump() for m in sensor_trace.moments],
            "sensorsInvolved": sensor_trace.sensorsInvolved,
        }
        with open(
            os.path.join(export_root, f"motion_{motion_json['id']}.json"), "w"
        ) as f:
            json.dump(motion_json, f, indent=2)

    # 2️⃣ Cell file (base station)
    base_trace = traces.get("base_station_trace")
    if isinstance(base_trace, BaseStationTrace):
        cell_json = {
            "id": str(uuid.uuid4())[:20],
            "time": int(time.time() * 1000),
            "moments": [m.model_dump() for m in base_trace.moments],
        }
        with open(os.path.join(export_root, f"cells_{cell_json['id']}.json"), "w") as f:
            json.dump(cell_json, f, indent=2)

    # 3️⃣ Record file (GPS path)
    draw_trace = traces.get("draw_trace")
    if draw_trace is not None and isinstance(draw_trace, DrawTrace):
        record_json = {
            "id": str(uuid.uuid4())[:20],
            "name": f"{attrs.first_name}_{attrs.last_name}_route",
            "coordSys": "WGS84",
            "points": [p.model_dump() for p in draw_trace.points],
        }
        with open(
            os.path.join(export_root, f"record_{record_json['id']}.json"), "w"
        ) as f:
            json.dump(record_json, f, indent=2)

    # 4️⃣ Tar everything EXCEPT persona.json
    tar_path = f"{export_root}.tar.gz"
    import tarfile

    with tarfile.open(tar_path, "w:gz") as tar:
        for filename in os.listdir(export_root):
            # skip persona.json (explicit)
            if filename == "persona.json":
                continue
            # optionally skip hidden files
            if filename.startswith("."):
                continue
            tar.add(os.path.join(export_root, filename), arcname=filename)

    print(f"✅ Motion Emulator bundle ready: {tar_path}")
    print("📂 Tar Contents:")
    os.system(f"tar -tzf {tar_path}")

    return {"save_dir": export_root, "tar_path": tar_path, "persona_path": persona_path}


# ----- Wire the graph -----
workflow = StateGraph(GraphState)

workflow.add_node("generate_privacy_attrs", generate_privacy_attrs_node)
workflow.add_node("grade_persona", persona_grader_node)
workflow.add_node("generate_traces", generate_traces_node)
workflow.add_node("grade_traces_node", grade_traces_node)
workflow.add_node("package_persona", package_persona_node)
workflow.add_node("save_package", save_package_node)

workflow.set_entry_point("generate_privacy_attrs")
workflow.add_edge("generate_privacy_attrs", "grade_persona")
workflow.add_conditional_edges(
    "grade_persona",
    lambda output: output["decision"],  # the node returns either "proceed" or "redo"
    {
        "proceed": "generate_traces",
        "redo": "generate_privacy_attrs",
    },
)
workflow.add_edge("generate_traces", "grade_traces_node")
workflow.add_conditional_edges(
    "grade_traces_node",
    lambda x: x["decision"],
    {
        "proceed": "package_persona",
        "redo": "generate_traces",
    },
)
workflow.add_edge("package_persona", "save_package")
workflow.add_edge("save_package", END)

app = workflow.compile()

# -----------------------------------------
# Example Usage
# -----------------------------------------
# if __name__ == "__main__":
#     prompt: str = "Sarah, software engineer in San Francisco, jogging in Golden Gate Park."
#     app.invoke(cast(GraphState, {"prompt": prompt}))

    # prompt_2 = (
    #     "John who is a writer. "
    #     "He is currently sitting in a cafe in New York City, working on his laptop."
    # )
    # app.invoke({"prompt": prompt_2})
