import datetime
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, TypedDict, Union
from uu import Error
import uuid
from xxlimited import Null

from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage

from persona_generation_prompt import system_message
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_SANDBOX_V1"] = "true"
os.environ["LANGCHAIN_SANDBOX"] = os.getenv("LANGCHAIN_PROJECT")

# llm_model = ChatOpenAI(model="gpt-5-mini-2025-08-07")

groq_api_key = os.getenv("GROQ_API_KEY")
llm_model = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)


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


# ------------------------------------------------------------------------------------------------

# Privacy Attributes


class PrivacyAttributes(BaseModel):
    first_name: str = Field(description="The first name of the persona")
    last_name: str = Field(description="The last name of the persona")
    age: str = Field(description="The age of the persona")
    gender: str = Field(description="The gender of the persona")
    race: str = Field(description="The race of the persona")
    street: str = Field(description="The street of the persona living address")
    city: str = Field(description="The city of the persona living address")
    state: str = Field(description="The city of the persona living address")
    zip_code: str = Field(description="The zipcode of the persona living address")
    spoken_language: str = Field(description="The spoken language of the persona")
    education_background: str = Field(
        description="The education background of the persona"
    )
    education_level: str = Field(
        description="The most suitable education level of the persona, four options: high school diploma, attending college, bachelor's degree, advanced degree"
    )
    birthday: str = Field(description="The birthday of the persona")
    job: str = Field(description="The job of the persona")
    income: str = Field(description="The annual income of the persona")
    income_level: str = Field(
        description="The most suitable income level of the persona, three options: high income, moderate high income, average or lower income"
    )
    marital_status: str = Field(
        description="The most suitable marital status of the persona, three options: single, married, in a relationship"
    )
    parental_status: str = Field(
        description="The most suitable parental status of the persona, six options: not parents, parents of infants, parents of toddlers, parents of preschoolers, parents of grade schoolers, parents of teenagers"
    )
    online_behavior: str = Field(description="The online behavior of the persona")
    industry: str = Field(
        description="The most suitable industry of the persona, eight options: construction, education, finance, healthcare, hospitality, manufacturing, real estate, technology"
    )
    employer_size: str = Field(
        description="The most suitable employer size of the persona, three options: small employer (1-249 employees), large employer (250-10,000 employees), very large employer (more than 10,000 employees)"
    )
    homeownership: str = Field(
        description="The most suitable homeownership status of the persona, either renter of homeowner"
    )
    short_profile: str = Field(description="A short verstion of the profile")
    age_type: str = Field(
        description="Rate this persona age as young, mid-aged, or old"
    )
    gender_type: str = Field(
        description="Rate this persona gender as male, female, or non-binary"
    )
    location_type: str = Field(
        description="Rate this persona location as urban, suburb, or countryside"
    )
    income_type: str = Field(
        description="Rate this persona income as low, medium, or high"
    )
    edu_type: str = Field(
        description="Rate this persona educational level as low, medium, high"
    )
    activity_description: str = Field(
        description=f"A description of the persona's current activity (e.g., 'running', 'sitting', 'commuting', 'sitting at a bar', 'eating', 'sleeping', 'working', 'studying', 'reading', 'watching TV', 'listening to music', 'browsing the internet', 'socializing', 'other')."
    )


class PersonaPackage(BaseModel):
    persona: PrivacyAttributes
    traces: Dict[str, BaseModel]


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

    def generate(self) -> SensorTrace:
        id_str = str(uuid.uuid4())
        start_time = int(time.time() * 1000)
        total_frames = self.duration_s * self.fps

        moments = []
        sensors_involved = []

        # Define which sensors output different number of values
        uncalibrated_sensors = [
            "accelerometer_uncalibrated",
            "magnetic_field_uncalibrated",
            "gyroscope_uncalibrated",
        ]

        single_value_sensors = ["light", "step_counter", "step_detector"]

        # --- helper: simulate low-frequency bias (e.g., slow drift or calibration offset)

        def _slow_bias(elapsed: float, bias_amp: float = 0.2):
            # slow sinusoidal drift + slight random walk
            return math.sin(elapsed / 20.0) * bias_amp + random.gauss(0, bias_amp / 4)

        def _normalize_activity(self, text: str) -> str:
            text_lower = text.lower()
            for canonical, variants in self.activity_aliases.items():
                if any(alias in text_lower for alias in variants):
                    return canonical
            # default fallback
            return "sitting"

        # --- helper: simulate activity-specific temporal pattern
        def _activity_modulation(activity: str, elapsed: float) -> float:
            if activity == "running":
                # periodic spikes for steps
                return 1.0 + 0.8 * abs(math.sin(elapsed * 2.5))
            elif activity == "commuting":
                # smooth oscillations for vehicle motion
                return (
                    1.0 + 0.3 * math.sin(elapsed / 2.5) + 0.15 * math.sin(elapsed / 0.7)
                )
            elif activity == "driving":
                # bursts of acceleration & turns
                return (
                    1.0
                    + 0.3 * math.sin(elapsed / 2.5)
                    + 0.15 * math.sin(elapsed / 0.7)
                    + random.gauss(0, 0.05)
                )
            elif activity == "sitting":
                # small body micro-movements
                return 1.0 + random.gauss(0, 0.01)
            else:
                return 1.0 + random.gauss(0, 0.05)

        activity = _normalize_activity(
            self, text=self.persona.activity_description.lower()
        )
        traits = self.activity_sensor_map.get(activity, {})

        for i in range(total_frames):
            elapsed = i / self.fps
            frame_data = {}

            for sensor, stats in traits.items():
                if sensor not in self.sensor_id_map:
                    continue
                sensors_involved.append(self.sensor_id_map[sensor])

                mean = stats.get("mean", 0.0)
                base_drift = stats.get("base_drift", stats.get("drift", 0.05))
                bias_drift = stats.get("bias_drift", base_drift / 2.0)

                # bias term changes slowly over time
            bias = _slow_bias(elapsed, bias_amp=bias_drift)

            # short-term variation
            micro_noise = random.gauss(0, base_drift)

            # motion modulation
            motion_factor = _activity_modulation(activity, elapsed)

            value = (mean + bias + micro_noise) * motion_factor

            # --- build output per sensor type
            if sensor in uncalibrated_sensors:
                data = [
                    round(value + random.gauss(0, 0.05), 5),
                    round(value + random.gauss(0, 0.05), 5),
                    round(value + random.gauss(0, 0.05), 5),
                    round(bias, 5),  # bias_x
                    round(bias / 2, 5),  # bias_y
                    round(bias / 3, 5),  # bias_z
                ]
            elif sensor in single_value_sensors:
                data = [round(value, 5)]
            else:
                data = [
                    round(value + random.gauss(0, 0.05), 5),
                    round(value + random.gauss(0, 0.05), 5),
                    round(value + random.gauss(0, 0.05), 5),
                ]

            frame_data[str(self.sensor_id_map[sensor])] = data

            moments.append(SensorMoment(elapsed=elapsed, data=frame_data))

        return SensorTrace(
            id=id_str,
            time=start_time,
            moments=moments,
            sensorsInvolved=list(set(sensors_involved)),
        )


class BaseStationTraceGenerator:
    def generate(self) -> BaseStationTrace:
        return BaseStationTrace(
            id=str(uuid.uuid4()),
            time=int(time.time() * 1000),
            moments=[],  # SIM-less fallback
        )


class DrawTraceGenerator:
    def __init__(self, persona: PrivacyAttributes, duration_s: int = 600):
        self.persona = persona
        self.steps = max(60, duration_s)  # 1Hz default sampling

    def generate(self) -> DrawTrace:
        points = []
        geolocator = Nominatim(user_agent="persona_generator")
        location = geolocator.geocode(f"{self.persona.city}, {self.persona.state}")

        if location:
            lat, lon = location.latitude, location.longitude
        else:
            lat, lon = 37.7749, -122.4194

        for _ in range(self.steps):
            # ~11m per jitter at latitude ≈37°, tunable for realism
            lat += random.uniform(-0.0001, 0.0001)
            lon += random.uniform(-0.0001, 0.0001)
            points.append(DrawPoint(latitude=lat, longitude=lon))

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


def persona_grader_node(state: GraphState) -> str:
    """
    Grade persona consistency and decide next step.
    Returns either 'redo' (go back to regenerate) or 'ok' (proceed).
    """
    persona: PrivacyAttributes = state["privacy_attrs"]
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
    traces = state.get("traces", {})
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
        # "Be objective but not cynical; many good synthetic traces should earn 6–8.\n"
        "Output only a single integer 0–10.\n\n"
        f"Persona summary: {getattr(persona, 'first_name', '')} {getattr(persona, 'last_name', '')}, "
        f"activity={getattr(persona, 'activity_description', '')}, job={getattr(persona, 'job', '')}, "
        f"city={getattr(persona, 'city', '')}\n\n"
        f"Sensor snippet: {getattr(traces.get('sensor_trace'), 'moments', [])[:3]}\n"
        f"GPS snippet: {getattr(traces.get('draw_trace'), 'points', [])[:3]}\n"
    )

    # llm_trace_grader = ChatOpenAI(model="gpt-5-mini-2025-08-07")
    # response = llm_trace_grader.invoke(grader_prompt)
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

    persona_dir = f"{attrs.first_name}_{attrs.last_name}_data"
    os.makedirs(persona_dir, exist_ok=True)

    def _dump(path: str, model: BaseModel):
        with open(path, "w") as f:
            json.dump(model.model_dump(), f, indent=2)

    # Save persona attributes
    _dump(os.path.join(persona_dir, "persona.json"), attrs)

    # Save traces
    for name, trace in traces.items():
        _dump(os.path.join(persona_dir, f"{name}.json"), trace)

    print(f"✅ Saved persona and traces to '{persona_dir}'")
    return {"save_dir": persona_dir}


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
if __name__ == "__main__":
    prompt = "Sarah, software engineer in San Francisco, jogging in Golden Gate Park."
    app.invoke({"prompt": prompt})

    prompt_2 = (
        "John who is a writer. "
        "He is currently sitting in a cafe in New York City, working on his laptop."
    )
    app.invoke({"prompt": prompt_2})
