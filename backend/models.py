
from typing import Any, Dict, Literal, cast
from pydantic import BaseModel, Field

class UserInput(BaseModel):
    age: int
    city: str
    job: str

    # right-now state (situational)
    context: Literal["work", "home", "commuting", "waiting", "other"]
    activity_state: Literal["stationary", "active"]
    day_type: Literal["weekday", "weekend"]
    hour_of_day: int  # 0-23 (local)

    # simple preferences (user-understandable)
    chronotype_self_report: Literal["morning", "neutral", "night"]
    phone_style: Literal["quick_checks", "mixed", "long_sessions"]
    primary_use: Literal["social", "video_news", "mixed"]
    usage_level: Literal["light", "typical", "heavy"]

class BehaviorSpec(BaseModel):
    chronotype: float  # inferred from chronotype_self_report + hour_of_day patterns later
    attentional_granularity: float  # from phone_style
    baseline_intensity: float  # how heavy user is (screen time + pickups); start with prior
    engagement_social_weight: float  # from primary_use

    mobility_radius: float  # from context/activity + later calibrated by GPS
    context: Literal["work", "home_evening", "commuting", "waiting", "other"]
    activity_state: Literal["stationary", "active"]
    day_type: Literal["weekday", "weekend"]
    hour_of_day: int

class BehavioralParameters(BaseModel):

    # --- Global Daily Controls ---
    total_daily_usage_minutes: float
    baseline_pickups_per_hour: float

    # --- Temporal Distribution ---
    temporal_peak_shift: float  
    # negative = morning bias, positive = evening bias

    late_night_usage_probability: float
    earliest_use_hour: int
    latest_use_hour: int

    # --- Session Structure ---
    avg_session_duration_seconds: float
    session_duration_variance: float
    app_loop_probability: float
    glance_probability: float

    # --- Context Modifiers ---
    context_duration_multiplier: float
    context_frequency_multiplier: float
    interaction_gate_probability: float

    # --- App Selection Bias ---
    social_weight: float
    process_weight: float
    navigation_weight: float
    music_weight: float

    # --- Mobility Metrics ---
    radius_of_gyration_target: float
    location_entropy_target: float

    # --- Validation Anchors ---
    exploit_fraction_target: float
    fragmentation_index_target: float

    
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
    birthday: str = Field(description="The birthday of the persona (Day and Month only)")
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
        description="Rate this persona age as young, mid-aged, or old based on age"
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
        description=f"A description of the persona's current activity (e.g., 'running', 'sitting', 'commuting', 'sitting at a bar', 'eating', 'sleeping', 'working', 'studying', 'reading', 'watching TV', 'listening to music', 'browsing the internet', 'socializing', 'other' based on activity_state)."
    )
    
class PersonaPackage(BaseModel):
    profile: PrivacyAttributes
    behavior_spec: BehaviorSpec
    parameters: BehavioralParameters
    traces: Dict[str, BaseModel]
    
class Action(BaseModel):
    app: str
    action: str
    args: Dict[str, Any]

class ThoughtAction(BaseModel):
    thought: str
    action: Action
