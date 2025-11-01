import os
import uuid
import time
import json
import random
from typing import List, Dict, Union, Optional, TypedDict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from geopy.geocoders import Nominatim


# Load environment variables
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_SANDBOX_V1"] = "true"
os.environ["LANGCHAIN_SANDBOX"] = os.getenv("LANGCHAIN_PROJECT")


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
        description="A description of the persona's current activity (e.g., 'running', 'sitting', 'commuting')."
    )


class PersonaPackage(BaseModel):
    persona: PrivacyAttributes
    traces: Dict[str, BaseModel]


# -----------------------------------------
# Generators
# -----------------------------------------


class PersonaGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-5-mini-2025-08-07")

    def generate(self, prompt: str) -> PrivacyAttributes:
        structured_llm = self.llm.with_structured_output(PrivacyAttributes)
        return structured_llm.invoke(prompt)


class SensorTraceGenerator:
    def __init__(self, persona: PrivacyAttributes, duration_s=60, fps=30):
        self.persona = persona
        self.duration_s = duration_s
        self.fps = fps
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
                "accelerometer": {"mean": 5.0, "drift": 1.5},
                "gyroscope": {"mean": 2.0, "drift": 0.8},
                "step_counter": {"mean": 2.0, "drift": 0.5},
                "step_detector": {"mean": 1.0, "drift": 0.1},
                "linear_acceleration": {"mean": 4.0, "drift": 1.2},
            },
            "sitting": {
                "accelerometer": {"mean": 0.1, "drift": 0.05},
                "gyroscope": {"mean": 0.05, "drift": 0.02},
                "step_counter": {"mean": 0.0, "drift": 0.0},
                "step_detector": {"mean": 0.0, "drift": 0.0},
                "linear_acceleration": {"mean": 0.1, "drift": 0.05},
            },
            "commuting": {
                "accelerometer": {"mean": 1.0, "drift": 0.5},
                "gyroscope": {"mean": 0.5, "drift": 0.3},
                "light": {"mean": 200, "drift": 50},
                "magnetic_field": {"mean": 40, "drift": 10},
            },
        }

    def generate(self) -> SensorTrace:
        id_str = str(uuid.uuid4())
        start_time = int(time.time() * 1000)
        total_frames = self.duration_s * self.fps

        moments = []
        sensors_involved = []

        activity = self.persona.activity_description.lower()
        traits = self.activity_sensor_map.get(activity, {})

        # Define which sensors output different number of values
        uncalibrated_sensors = [
            "accelerometer_uncalibrated",
            "magnetic_field_uncalibrated",
            "gyroscope_uncalibrated",
        ]

        single_value_sensors = ["light", "step_counter", "step_detector"]

        for i in range(total_frames):
            elapsed = i / self.fps
            frame_data = {}

            for sensor, stats in traits.items():
                if sensor not in self.sensor_id_map:
                    continue
                sensors_involved.append(self.sensor_id_map[sensor])

                mean = stats.get("mean", 0)
                drift = stats.get("drift", 0.01)

                # Generate data based on sensor type
                if sensor in uncalibrated_sensors:
                    # 6 values for uncalibrated sensors
                    data = [
                        round(random.gauss(mean, drift), 5),
                        round(random.gauss(mean, drift), 5),
                        round(random.gauss(mean, drift), 5),
                        round(random.gauss(0, drift), 5),  # Bias values
                        round(random.gauss(0, drift), 5),
                        round(random.gauss(0, drift), 5),
                    ]
                elif sensor in single_value_sensors:
                    # 1 value for light, step_counter, step_detector
                    data = [round(random.gauss(mean, drift), 5)]
                else:
                    # 3 values for standard sensors
                    data = [
                        round(random.gauss(mean, drift), 5),
                        round(random.gauss(mean, drift), 5),
                        round(random.gauss(mean, drift), 5),
                    ]
                # Use numeric sensor ID as key to match mock data format
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


from geopy.geocoders import Nominatim


class DrawTraceGenerator:
    def __init__(self, persona: PrivacyAttributes, steps: int = 60):
        self.persona = persona
        self.steps = steps

    def generate(self) -> DrawTrace:
        points = []
        geolocator = Nominatim(user_agent="persona_generator")
        location = geolocator.geocode(f"{self.persona.city}, {self.persona.state}")

        if location:
            start_lat, start_lon = location.latitude, location.longitude
        else:
            # Use a default start location if not provided in persona
            start_lat = 37.7749
            start_lon = -122.4194

        # A more realistic GPS trace would require more context from the persona,
        # like a schedule with locations. For now, we'll just jitter around a point.

        lat, lon = start_lat, start_lon
        for _ in range(self.steps):
            lat_jitter = random.uniform(-0.0001, 0.0001)
            lon_jitter = random.uniform(-0.0001, 0.0001)
            lat += lat_jitter
            lon += lon_jitter
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
    persona_generator = PersonaGenerator()
    attrs = persona_generator.generate(prompt)

    # Example lightweight guardrails you had in your code
    if "John" in attrs.first_name:
        attrs.city = "New York City"
        attrs.activity_description = attrs.activity_description or "sitting"
    elif "Sarah" in attrs.first_name:
        attrs.activity_description = attrs.activity_description or "running"

    return {"privacy_attrs": attrs}


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
        raise ValueError(
            "persona_package not found. Run package_persona_node first."
        )

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
workflow.add_node("generate_traces", generate_traces_node)
workflow.add_node("package_persona", package_persona_node)
workflow.add_node("save_package", save_package_node)

workflow.set_entry_point("generate_privacy_attrs")
workflow.add_edge("generate_privacy_attrs", "generate_traces")
workflow.add_edge("generate_traces", "package_persona")
workflow.add_edge("package_persona", "save_package")
workflow.add_edge("save_package", END)

app = workflow.compile()

# -----------------------------------------
# Example Usage
# -----------------------------------------
if __name__ == "__main__":
    prompt = (
        "Create a persona for a 28-year-old woman named Sarah who is a software engineer "
        "living in San Francisco. She is currently out for a morning run in Golden Gate Park."
    )
    app.invoke({"prompt": prompt})

    prompt_2 = (
        "Create a persona for a 45-year-old man named John who is a writer. "
        "He is currently sitting in a cafe in New York City, working on his laptop."
    )
    app.invoke({"prompt": prompt_2})
