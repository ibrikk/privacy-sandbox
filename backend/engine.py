"""
engine.py - Literature-Grounded Smartphone Behavior Simulation Engine

This module implements the transformation pipeline:
    Survey Input → Behavioral Dimensions → Behavioral Parameters → Daily Schedule

All parameter derivations are grounded in empirical research from digital phenotyping
and mobile sensing literature.

Key Literature Sources:
- Chronotype effects: Randjelovic et al. (2021) - MEQ validation, 40% screen time difference
- Session structure: Heitmayer & Lahlou (2021) - 5-min rhythm, glances vs sessions
- Context invariance: Heitmayer & Lahlou (2021) - frequency invariant, duration varies
- Temporal patterns: Roehrick et al. (2023) - time bins, night RR=1.61
- Mobility metrics: Hackett et al. (2024), Yi et al. (2024) - RoG, entropy
- Activity suppression: Katevas et al. - 70-80% reduction during movement
- Session composition: Toth et al. (2025) - 78% single-interaction, 22% loops

Version: 2.0.0
"""

from datetime import date, datetime, time, timedelta
import math
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
from collections import defaultdict
import uuid
import json
from agent import LLMContextEnhancer

# Import all models from models.py
from models import (
    # Enums
    AgeRange,
    AreaType,
    WakeTime,
    SleepTime,
    ChronotypeLabel,
    PeakUsageTime,
    RoutineStructure,
    CommuteDays,
    CommuteMode,
    PhysicalActivityDays,
    CommuteTime,
    ScreenTime,
    CheckingFrequency,
    SessionType,
    GlanceFrequency,
    WorkPhoneRestriction,
    EveningSessionChange,
    AppCategory,
    UsageReason,
    ExplorationStyle,
    ActivityType,
    ContextType,
    # Data models
    ComprehensiveSurveyInput,
    BehavioralDimensions,
    BehavioralParameters,
    LiteratureReference,
    PhoneSession,
    ScheduleSegment,
    DailySchedule,
    GenerationRequest,
    GenerationResponse,
)


# =============================================================================
# SECTION 1: LITERATURE-GROUNDED CONSTANTS
# =============================================================================


class LiteratureConstants:
    """
    Empirically-derived constants with source citations.
    All values are grounded in peer-reviewed research.
    """

    # -------------------------------------------------------------------------
    # Session Metrics (Toth et al. 2025; Heitmayer & Lahlou 2021)
    # -------------------------------------------------------------------------

    # Daily session counts
    # Source: "58-72 sessions/day (median); 115-177 unlocks/day for heavy users"
    SESSIONS_PER_DAY_LIGHT = 35
    SESSIONS_PER_DAY_MEDIAN = 65
    SESSIONS_PER_DAY_HEAVY = 120

    # Glance parameters
    # Source: "Glances: 5-25 sec average; lock-screen: 5.2 sec"
    # Source: "~17% are lock-screen checks; ~30-35% of all interactions"
    GLANCE_DURATION_MEAN = 12.0  # seconds
    GLANCE_DURATION_SD = 5.0
    GLANCE_DURATION_MIN = 3.0
    GLANCE_DURATION_MAX = 25.0
    GLANCE_PROPORTION_BASE = 0.25  # 25% of interactions are glances

    # Session duration (unlocked interactions)
    # Source: "78% single-interaction (46 sec), 22% are loops (104 sec)"
    SINGLE_SESSION_MEAN = 46.0  # seconds
    SINGLE_SESSION_SD = 30.0
    LOOP_SESSION_MEAN = 104.0  # seconds
    LOOP_SESSION_SD = 60.0
    LOOP_PROBABILITY = 0.22  # 22% become multi-app loops

    # -------------------------------------------------------------------------
    # Checking Rhythm (Heitmayer & Lahlou 2021)
    # -------------------------------------------------------------------------

    # The "5-minute rhythm" - frequency is relatively INVARIANT across contexts
    # Source: "~290.5 seconds between pickups on average"
    INTER_CHECK_INTERVAL_MEAN_SEC = 290.5
    INTER_CHECK_INTERVAL_SD_SEC = 180.0

    # User initiation ratio
    # Source: "89% user-initiated (pull) vs 11% notification-triggered"
    USER_INITIATED_RATIO = 0.89

    # -------------------------------------------------------------------------
    # Screen Time by Chronotype (Randjelovic et al. 2021)
    # -------------------------------------------------------------------------

    # Source: "Evening types 5.81 hrs vs Morning types 4.16 hrs"
    MORNING_TYPE_SCREEN_TIME_MIN = 250  # 4.16 hrs
    EVENING_TYPE_SCREEN_TIME_MIN = 349  # 5.81 hrs
    MEDIAN_SCREEN_TIME_MIN = 245  # German adults study

    # -------------------------------------------------------------------------
    # Context Effects on Duration (Heitmayer & Lahlou 2021)
    # -------------------------------------------------------------------------

    # CRITICAL: These primarily affect DURATION, not frequency
    # Source: "Work: β=-36.02; Home: β=+42.53; Work-from-home: β=-59.05"
    # Baseline ~290 sec, so β values represent ~12-20% changes

    # DURATION varies by context (Heitmayer & Lahlou 2021)
    # Work: β=-36.02; Home: β=+42.53
    CONTEXT_DURATION_MULTIPLIERS = {
        ContextType.WORK_RESTRICTED: 0.60,
        ContextType.WORK_FREE: 0.85,
        ContextType.HOME_MORNING: 1.00,
        ContextType.HOME_EVENING: 1.25,
        ContextType.HOME_NIGHT: 1.40,
        # Treat all commute contexts as one light/generic commute behavior
        ContextType.COMMUTE_TRANSIT: 0.55,
        ContextType.COMMUTE_DRIVING: 0.55,
        ContextType.COMMUTE_WALKING: 0.55,
        ContextType.PUBLIC_PLACE: 0.95,
        ContextType.EXERCISING: 0.05,
        ContextType.SOCIAL_SETTING: 0.70,
    }

    CONTEXT_FREQUENCY_MULTIPLIERS = {
        ContextType.WORK_RESTRICTED: 0.85,
        ContextType.WORK_FREE: 0.95,
        ContextType.HOME_MORNING: 1.00,
        ContextType.HOME_EVENING: 1.05,
        ContextType.HOME_NIGHT: 0.80,
        # Much lower checking during commute so it stops dominating charts
        ContextType.COMMUTE_TRANSIT: 0.18,
        ContextType.COMMUTE_DRIVING: 0.18,
        ContextType.COMMUTE_WALKING: 0.18,
        ContextType.PUBLIC_PLACE: 1.00,
        ContextType.EXERCISING: 0.02,
        ContextType.SOCIAL_SETTING: 0.85,
    }

    # -------------------------------------------------------------------------
    # Mobility Metrics (Hackett et al. 2024; Yi et al. 2024)
    # -------------------------------------------------------------------------

    # Radius of gyration by area type
    # Source: "Older adults: ~9.3 km; derived from 26.6 km median travel"
    RADIUS_OF_GYRATION_KM = {
        AreaType.URBAN: 6.0,
        AreaType.SUBURBAN: 12.0,
        AreaType.RURAL: 20.0,
    }

    # Home time
    # Source: "14.6 hours/day; 67% of interactions at home"
    HOME_TIME_HOURS_MEAN = 14.6
    HOME_INTERACTION_PROPORTION = 0.67

    # -------------------------------------------------------------------------
    # Activity Suppression (Katevas et al.; Chen et al. 2022)
    # -------------------------------------------------------------------------

    # Source: "Only 6% of use while walking, 2% while exercising"
    ACTIVITY_SUPPRESSION = {
        ActivityType.SLEEPING: 0.01,
        ActivityType.WAKING_UP: 0.60,
        ActivityType.MORNING_ROUTINE: 0.50,
        ActivityType.COMMUTING: 0.40,  # varies by mode
        ActivityType.WORKING: 0.70,  # varies by restriction
        ActivityType.LUNCH_BREAK: 1.00,
        ActivityType.EXERCISING: 0.02,  # 2% while exercising
        ActivityType.ERRANDS: 0.60,
        ActivityType.HOME_EVENING: 1.20,  # peak usage
        ActivityType.LEISURE: 1.30,
        ActivityType.WINDING_DOWN: 0.90,
    }


# =============================================================================
# SECTION 2: SURVEY MAPPINGS
# =============================================================================


class SurveyMappings:
    """
    Mappings from survey enum values to numerical values for dimension extraction.
    Each mapping transforms categorical responses into continuous [0, 1] scales.
    """

    # --- Temporal mappings ---

    WAKE_TIME_TO_HOUR = {
        WakeTime.BEFORE_6: 5.0,
        WakeTime.SIX_TO_8: 7.0,
        WakeTime.EIGHT_TO_10: 9.0,
        WakeTime.TEN_TO_12: 11.0,
        WakeTime.AFTER_12: 13.0,
    }

    # Higher score = later wake = more evening type
    WAKE_TIME_TO_SCORE = {
        WakeTime.BEFORE_6: 0.0,
        WakeTime.SIX_TO_8: 0.25,
        WakeTime.EIGHT_TO_10: 0.50,
        WakeTime.TEN_TO_12: 0.75,
        WakeTime.AFTER_12: 1.0,
    }

    SLEEP_TIME_TO_HOUR = {
        SleepTime.BEFORE_10PM: 21.5,
        SleepTime.TEN_TO_12: 23.0,
        SleepTime.TWELVE_TO_2: 25.0,  # 1 AM = 25 for continuity
        SleepTime.TWO_TO_4: 27.0,
        SleepTime.AFTER_4AM: 28.5,
    }

    # Higher score = later sleep = more evening type
    SLEEP_TIME_TO_SCORE = {
        SleepTime.BEFORE_10PM: 0.0,
        SleepTime.TEN_TO_12: 0.25,
        SleepTime.TWELVE_TO_2: 0.50,
        SleepTime.TWO_TO_4: 0.75,
        SleepTime.AFTER_4AM: 1.0,
    }

    CHRONOTYPE_TO_SCORE = {
        ChronotypeLabel.DEFINITELY_MORNING: 0.0,
        ChronotypeLabel.MORE_MORNING: 0.25,
        ChronotypeLabel.NEITHER: 0.50,
        ChronotypeLabel.MORE_EVENING: 0.75,
        ChronotypeLabel.DEFINITELY_EVENING: 1.0,
    }

    PEAK_USAGE_TO_HOUR = {
        PeakUsageTime.MORNING: 9,
        PeakUsageTime.AFTERNOON: 14,
        PeakUsageTime.EVENING: 20,
        PeakUsageTime.LATE_NIGHT: 23,
        PeakUsageTime.EVENLY_DISTRIBUTED: 15,  # midpoint
    }

    # --- Usage intensity mappings ---

    SCREEN_TIME_TO_MINUTES = {
        ScreenTime.LESS_THAN_1_HOUR: 45,
        ScreenTime.ONE_TO_2_HOURS: 90,
        ScreenTime.TWO_TO_4_HOURS: 180,
        ScreenTime.FOUR_TO_6_HOURS: 300,
        ScreenTime.MORE_THAN_6_HOURS: 420,
    }

    # Normalized to [0, 1] based on literature (median ~4 hrs, heavy >6 hrs)
    SCREEN_TIME_TO_INTENSITY = {
        ScreenTime.LESS_THAN_1_HOUR: 0.10,
        ScreenTime.ONE_TO_2_HOURS: 0.25,
        ScreenTime.TWO_TO_4_HOURS: 0.50,
        ScreenTime.FOUR_TO_6_HOURS: 0.75,
        ScreenTime.MORE_THAN_6_HOURS: 0.95,
    }

    CHECKING_FREQ_TO_PER_HOUR = {
        CheckingFrequency.LESS_THAN_ONCE_PER_HOUR: 0.5,
        CheckingFrequency.ONE_TO_2_PER_HOUR: 1.5,
        CheckingFrequency.THREE_TO_4_PER_HOUR: 3.5,
        CheckingFrequency.FIVE_TO_6_PER_HOUR: 5.5,
        CheckingFrequency.MORE_THAN_6_PER_HOUR: 8.0,
    }

    # Normalized based on literature (median ~4/hr, heavy ~8+/hr)
    CHECKING_FREQ_TO_INTENSITY = {
        CheckingFrequency.LESS_THAN_ONCE_PER_HOUR: 0.05,
        CheckingFrequency.ONE_TO_2_PER_HOUR: 0.20,
        CheckingFrequency.THREE_TO_4_PER_HOUR: 0.45,
        CheckingFrequency.FIVE_TO_6_PER_HOUR: 0.70,
        CheckingFrequency.MORE_THAN_6_PER_HOUR: 0.95,
    }

    # --- Session structure mappings ---

    SESSION_TYPE_TO_GRANULARITY = {
        SessionType.LONG_SESSIONS: 0.10,
        SessionType.MEDIUM_SESSIONS: 0.35,
        SessionType.MIXED: 0.50,
        SessionType.SHORT_CHECKS: 0.70,
        SessionType.VERY_SHORT_CHECKS: 0.90,
    }

    SESSION_TYPE_TO_DURATION = {
        SessionType.VERY_SHORT_CHECKS: 15,  # seconds
        SessionType.SHORT_CHECKS: 45,
        SessionType.MEDIUM_SESSIONS: 150,
        SessionType.LONG_SESSIONS: 400,
        SessionType.MIXED: 90,
    }

    GLANCE_FREQ_TO_PROBABILITY = {
        GlanceFrequency.RARELY: 0.10,
        GlanceFrequency.SOMETIMES: 0.25,
        GlanceFrequency.OFTEN: 0.45,
        GlanceFrequency.VERY_OFTEN: 0.65,
    }

    # --- Context sensitivity mappings ---

    WORK_RESTRICTION_TO_MULTIPLIER = {
        WorkPhoneRestriction.USE_FREELY: 1.0,
        WorkPhoneRestriction.OCCASIONALLY: 0.70,
        WorkPhoneRestriction.BRIEFLY_WHEN_NECESSARY: 0.35,
        WorkPhoneRestriction.NOT_APPLICABLE: 1.0,  # no work = no restriction
    }

    WORK_RESTRICTION_TO_SENSITIVITY = {
        WorkPhoneRestriction.USE_FREELY: 0.10,
        WorkPhoneRestriction.OCCASIONALLY: 0.40,
        WorkPhoneRestriction.BRIEFLY_WHEN_NECESSARY: 0.80,
        WorkPhoneRestriction.NOT_APPLICABLE: 0.0,
    }

    EVENING_CHANGE_TO_MULTIPLIER = {
        EveningSessionChange.MUCH_SHORTER: 0.50,
        EveningSessionChange.SOMEWHAT_SHORTER: 0.75,
        EveningSessionChange.ABOUT_SAME: 1.00,
        EveningSessionChange.SOMEWHAT_LONGER: 1.35,
        EveningSessionChange.MUCH_LONGER: 1.80,
    }

    EVENING_CHANGE_TO_SENSITIVITY = {
        EveningSessionChange.MUCH_SHORTER: 0.90,
        EveningSessionChange.SOMEWHAT_SHORTER: 0.70,
        EveningSessionChange.ABOUT_SAME: 0.30,
        EveningSessionChange.SOMEWHAT_LONGER: 0.60,
        EveningSessionChange.MUCH_LONGER: 0.85,
    }

    # --- Mobility mappings ---

    COMMUTE_DAYS_TO_FREQUENCY = {
        CommuteDays.ZERO: 0.0,
        CommuteDays.ONE_TWO: 0.25,
        CommuteDays.THREE_FOUR: 0.60,
        CommuteDays.FIVE_PLUS: 0.90,
    }

    COMMUTE_TIME_TO_MINUTES = {
        CommuteTime.ALMOST_NONE: 5,
        CommuteTime.LESS_THAN_30: 20,
        CommuteTime.THIRTY_TO_60: 45,
        CommuteTime.ONE_TO_TWO_HOURS: 90,
        CommuteTime.MORE_THAN_2_HOURS: 150,
    }

    PLACES_TO_MOBILITY = {
        1: 0.10,
        2: 0.25,
        3: 0.45,
        4: 0.60,
        5: 0.75,
        6: 0.90,  # 6+ places
    }

    PHYSICAL_ACTIVITY_TO_DAYS = {
        PhysicalActivityDays.ZERO: 0,
        PhysicalActivityDays.ONE_TWO: 1.5,
        PhysicalActivityDays.THREE_FOUR: 3.5,
        PhysicalActivityDays.FIVE_SIX: 5.5,
        PhysicalActivityDays.EVERY_DAY: 7,
    }

    # --- Routine mappings ---

    ROUTINE_STRUCTURE_TO_STABILITY = {
        RoutineStructure.VERY_UNSTRUCTURED: 0.10,
        RoutineStructure.SOMEWHAT_UNSTRUCTURED: 0.30,
        RoutineStructure.MIXED: 0.50,
        RoutineStructure.SOMEWHAT_STRUCTURED: 0.70,
        RoutineStructure.VERY_STRUCTURED: 0.90,
    }

    # --- Exploration mappings ---

    EXPLORATION_TO_NOVELTY = {
        ExplorationStyle.MOSTLY_FAMILIAR: 0.20,
        ExplorationStyle.SOMETIMES_EXPLORE: 0.50,
        ExplorationStyle.BOTH_EQUALLY: 0.65,
    }

    # --- App category mappings ---

    # Social vs consumption orientation
    SOCIAL_CATEGORIES = {AppCategory.MESSAGING, AppCategory.SOCIAL_MEDIA}
    CONSUMPTION_CATEGORIES = {
        AppCategory.VIDEO_STREAMING,
        AppCategory.GAMES,
        AppCategory.NEWS_READING,
        AppCategory.MUSIC_AUDIO,
    }

    SOCIAL_REASONS = {UsageReason.MESSAGING_TALKING, UsageReason.SOCIAL_MEDIA}
    CONSUMPTION_REASONS = {
        UsageReason.WATCHING_VIDEOS,
        UsageReason.GAMING,
        UsageReason.READING_NEWS,
        UsageReason.LISTENING_MUSIC_PODCASTS,
    }


# =============================================================================
# SECTION 3: DIMENSION EXTRACTION
# =============================================================================


class DimensionExtractor:
    """
    Extracts BehavioralDimensions from ComprehensiveSurveyInput.

    Each dimension is computed from multiple survey questions using
    weighted combinations based on reliability and validity from literature.
    """

    def __init__(self):
        self.mappings = SurveyMappings()

    def extract(self, survey: ComprehensiveSurveyInput) -> BehavioralDimensions:
        """
        Extract all 8 behavioral dimensions from survey responses.

        Returns:
            BehavioralDimensions with all values in [0, 1] range
        """
        return BehavioralDimensions(
            chronotype_score=self._extract_chronotype(survey),
            usage_intensity=self._extract_usage_intensity(survey),
            attentional_granularity=self._extract_attentional_granularity(survey),
            contextual_sensitivity=self._extract_contextual_sensitivity(survey),
            social_orientation=self._extract_social_orientation(survey),
            mobility_diversity=self._extract_mobility_diversity(survey),
            routine_stability=self._extract_routine_stability(survey),
            novelty_seeking=self._extract_novelty_seeking(survey),
        )

    def _extract_chronotype(self, survey: ComprehensiveSurveyInput) -> float:
        """
        Chronotype score: 0 = extreme morning, 1 = extreme evening.

        Derived from: wake_time (Q5), sleep_time (Q6), chronotype_self_report (Q7)

        Grounding: Self-report chronotype is the strongest predictor of actual
        circadian behavior (Randjelovic et al., 2021), weighted highest.
        """
        wake_score = SurveyMappings.WAKE_TIME_TO_SCORE[survey.wake_time]
        sleep_score = SurveyMappings.SLEEP_TIME_TO_SCORE[survey.sleep_time]
        self_report = SurveyMappings.CHRONOTYPE_TO_SCORE[survey.chronotype_self_report]

        # Weights: self-report (50%), wake time (25%), sleep time (25%)
        # Literature supports self-report as most predictive
        chronotype = 0.50 * self_report + 0.25 * wake_score + 0.25 * sleep_score

        return np.clip(chronotype, 0.0, 1.0)

    def _extract_usage_intensity(self, survey: ComprehensiveSurveyInput) -> float:
        """
        Usage intensity: 0 = very light user, 1 = very heavy user.

        Derived from: daily_screen_time (Q15), checking_frequency (Q16)

        Grounding: Screen time and checking frequency are the primary indicators
        of usage intensity. ICC for frequency = 0.26 (Roehrick et al., 2023).
        """
        screen_intensity = SurveyMappings.SCREEN_TIME_TO_INTENSITY[
            survey.daily_screen_time
        ]
        check_intensity = SurveyMappings.CHECKING_FREQ_TO_INTENSITY[
            survey.checking_frequency
        ]

        # Weights: screen time (60%), checking frequency (40%)
        # Screen time is more stable trait, frequency more variable
        intensity = 0.60 * screen_intensity + 0.40 * check_intensity

        return np.clip(intensity, 0.0, 1.0)

    def _extract_attentional_granularity(
        self, survey: ComprehensiveSurveyInput
    ) -> float:
        """
        Attentional granularity: 0 = long focused sessions, 1 = fragmented glances.

        Derived from: session_type (Q17), glance_frequency (Q18)

        Grounding: Toth et al. (2025) show 78% of sessions are single-interaction,
        with glances representing 17-35% of all interactions.
        """
        session_granularity = SurveyMappings.SESSION_TYPE_TO_GRANULARITY[
            survey.session_type
        ]
        glance_tendency = SurveyMappings.GLANCE_FREQ_TO_PROBABILITY[
            survey.glance_frequency
        ]

        # Weights: session type (60%), glance frequency (40%)
        granularity = 0.60 * session_granularity + 0.40 * glance_tendency

        return np.clip(granularity, 0.0, 1.0)

    def _extract_contextual_sensitivity(
        self, survey: ComprehensiveSurveyInput
    ) -> float:
        """
        Contextual sensitivity: 0 = context-invariant, 1 = highly context-dependent.

        Derived from: work_phone_restriction (Q19), evening_session_change (Q20)

        Grounding: Heitmayer & Lahlou (2021) show duration varies by context
        (work β=-36, home β=+42) while frequency remains stable.
        """
        work_sensitivity = SurveyMappings.WORK_RESTRICTION_TO_SENSITIVITY[
            survey.work_phone_restriction
        ]
        evening_sensitivity = SurveyMappings.EVENING_CHANGE_TO_SENSITIVITY[
            survey.evening_session_change
        ]

        # Weights: equal contribution from work and evening contexts
        sensitivity = 0.50 * work_sensitivity + 0.50 * evening_sensitivity

        return np.clip(sensitivity, 0.0, 1.0)

    def _extract_social_orientation(self, survey: ComprehensiveSurveyInput) -> float:
        """
        Social orientation: 0 = consumption-focused, 1 = communication-focused.

        Derived from: most_used_categories (Q24), usage_reasons (Q22)

        Grounding: Stachl et al. (2020) show extraversion predicts call/messaging
        frequency, with social apps correlating with communication orientation.
        """
        # Count social vs consumption categories in top apps
        # Count social vs consumption categories
        social_count = sum(
            1
            for cat in survey.most_used_categories
            if cat in SurveyMappings.SOCIAL_CATEGORIES
        )
        consumption_count = sum(
            1
            for cat in survey.most_used_categories
            if cat in SurveyMappings.CONSUMPTION_CATEGORIES
        )

        # Count social vs consumption reasons
        social_reasons = sum(
            1
            for reason in survey.usage_reasons
            if reason in SurveyMappings.SOCIAL_REASONS
        )
        consumption_reasons = sum(
            1
            for reason in survey.usage_reasons
            if reason in SurveyMappings.CONSUMPTION_REASONS
        )

        # Compute ratios with explicit zero handling
        total_apps = social_count + consumption_count
        total_reasons = social_reasons + consumption_reasons

        # Default to 0.5 (neutral) if no relevant categories/reasons
        app_ratio = social_count / total_apps if total_apps > 0 else 0.5
        reason_ratio = social_reasons / total_reasons if total_reasons > 0 else 0.5

        orientation = 0.50 * app_ratio + 0.50 * reason_ratio
        return np.clip(orientation, 0.0, 1.0)

    def _extract_mobility_diversity(self, survey: ComprehensiveSurveyInput) -> float:
        """
        Mobility diversity: 0 = mostly stationary, 1 = highly mobile.

        Derived from: places_visited_daily (Q10), commute_days (Q11),
                      commute_time (Q14), physical_activity_days (Q13)

        Grounding: Hackett et al. (2024) show radius of gyration and location
        entropy correlate with cognitive function and wellbeing.
        """
        places_score = SurveyMappings.PLACES_TO_MOBILITY.get(
            min(survey.places_visited_daily, 6), 0.5
        )
        commute_freq = SurveyMappings.COMMUTE_DAYS_TO_FREQUENCY[survey.commute_days]

        commute_minutes = SurveyMappings.COMMUTE_TIME_TO_MINUTES[survey.commute_time]
        commute_time_score = min(1.0, commute_minutes / 120)  # normalized to 2 hrs

        activity_days = SurveyMappings.PHYSICAL_ACTIVITY_TO_DAYS[
            survey.physical_activity_days
        ]
        activity_score = min(1.0, activity_days / 7)
        # Weights: places (35%), commute frequency (25%), commute time (25%), activity (15%)
        mobility = (
            0.35 * places_score
            + 0.25 * commute_freq
            + 0.25 * commute_time_score
            + 0.15 * activity_score
        )

        return np.clip(mobility, 0.0, 1.0)

    def _extract_routine_stability(self, survey: ComprehensiveSurveyInput) -> float:
        """
        Routine stability: 0 = chaotic/variable, 1 = highly predictable.

        Derived from: routine_structure (Q9)

        Grounding: Routine stability predicts temporal regularity in phone use
        patterns. Structured individuals show more consistent usage timing.
        """
        stability = SurveyMappings.ROUTINE_STRUCTURE_TO_STABILITY[
            survey.routine_structure
        ]

        return np.clip(stability, 0.0, 1.0)

    def _extract_novelty_seeking(self, survey: ComprehensiveSurveyInput) -> float:
        """
        Novelty seeking: 0 = always familiar content, 1 = always exploring.

        Derived from: exploration_style (Q25)

        Grounding: Exploration vs exploitation tradeoff in content consumption.
        Related to openness personality trait (Stachl et al., 2020).
        """
        novelty = SurveyMappings.EXPLORATION_TO_NOVELTY[survey.exploration_style]

        return np.clip(novelty, 0.0, 1.0)


class ParameterDeriver:
    """
        Derives concrete BehavioralParameters from BehavioralDimensions.
        Each parameter derivation is grounded in literature with explicit formulas
    and citations. Parameters are designed to be directly usable in simulation.
    """

    def __init__(self):
        self.constants = LiteratureConstants()
        self.citations: List[LiteratureReference] = []

    def derive(
        self, dimensions: BehavioralDimensions, survey: ComprehensiveSurveyInput
    ) -> BehavioralParameters:
        """
        Derive all behavioral parameters from dimensions and survey data.

        Args:
            dimensions: Extracted behavioral dimensions (0-1 scales)
            survey: Original survey input (for direct mappings)

        Returns:
            BehavioralParameters ready for schedule generation
        """
        self.citations = []  # Reset citations

        # --- Temporal Distribution ---
        waking_start, waking_end = self._derive_waking_hours(survey)
        peak_hour = self._derive_peak_hour(survey, dimensions)
        late_night_prob = self._derive_late_night_probability(dimensions)

        # --- Usage Volume ---
        daily_minutes = self._derive_daily_screen_time(dimensions, survey)
        pickups = self._derive_pickups_per_day(dimensions)
        sessions = self._derive_sessions_per_day(dimensions)

        # --- Session Structure ---
        mean_duration, duration_std = self._derive_session_duration(dimensions, survey)
        glance_prob = self._derive_glance_probability(dimensions)

        # --- Checking Behavior ---
        inter_session = self._derive_inter_session_interval(sessions)
        burst_prob = self._derive_burst_probability(dimensions)

        # --- Context Multipliers ---
        work_dur_mult, work_freq_mult = self._derive_work_multipliers(survey)
        home_evening_mult = self._derive_home_evening_multiplier(survey)
        commute_freq_mult = self._derive_commute_multiplier(survey)
        active_suppression = self._derive_active_suppression(survey)

        # --- App Category Weights ---
        app_weights = self._derive_app_weights(survey)

        # --- Mobility ---
        rog = self._derive_radius_of_gyration(survey, dimensions)
        entropy = self._derive_location_entropy(dimensions)

        # --- Validation Anchors ---
        exploit_ratio = self._derive_exploit_ratio(dimensions)
        fragmentation = self._derive_fragmentation_index(dimensions)

        return BehavioralParameters(
            # Temporal
            waking_hour_start=waking_start,
            sleep_hour=waking_end,
            temporal_peak_hour=peak_hour,
            late_night_probability=late_night_prob,
            # Volume
            total_daily_minutes=daily_minutes,
            pickups_per_day=pickups,
            sessions_per_day=sessions,
            # Session structure
            mean_session_duration_seconds=mean_duration,
            session_duration_std=duration_std,
            glance_probability=glance_prob,
            # Checking
            mean_inter_session_interval_minutes=inter_session,
            checking_burst_probability=burst_prob,
            # Context multipliers
            work_duration_multiplier=work_dur_mult,
            work_frequency_multiplier=work_freq_mult,
            home_evening_duration_multiplier=home_evening_mult,
            commute_frequency_multiplier=commute_freq_mult,
            active_suppression_factor=active_suppression,
            # App weights
            weight_social=app_weights.get(AppCategory.SOCIAL_MEDIA, 0.1),
            weight_messaging=app_weights.get(AppCategory.MESSAGING, 0.1),
            weight_video=app_weights.get(AppCategory.VIDEO_STREAMING, 0.1),
            weight_music=app_weights.get(AppCategory.MUSIC_AUDIO, 0.1),
            weight_navigation=app_weights.get(AppCategory.MAPS_NAVIGATION, 0.05),
            weight_productivity=app_weights.get(AppCategory.PRODUCTIVITY_WORK, 0.1),
            weight_news=app_weights.get(AppCategory.NEWS_READING, 0.1),
            weight_games=app_weights.get(AppCategory.GAMES, 0.1),
            weight_shopping=app_weights.get(AppCategory.SHOPPING, 0.05),
            # Mobility
            radius_of_gyration_km=rog,
            location_entropy=entropy,
            # Validation
            expected_exploit_ratio=exploit_ratio,
            expected_fragmentation_index=fragmentation,
        )

    def _derive_waking_hours(self, survey: ComprehensiveSurveyInput) -> Tuple[int, int]:
        """
        Derive waking hours from survey wake/sleep times.

        Source: Direct mapping from survey Q5, Q6
        """
        wake_hour = int(SurveyMappings.WAKE_TIME_TO_HOUR[survey.wake_time])

        sleep_hour_raw = SurveyMappings.SLEEP_TIME_TO_HOUR[survey.sleep_time]
        sleep_hour = int(sleep_hour_raw) % 24  # Convert to 0-23

        self.citations.append(
            LiteratureReference(
                parameter="waking_hours",
                formula="Direct mapping from survey Q5 (wake) and Q6 (sleep)",
                sources=["Survey direct response"],
                notes="Sleep times after midnight wrapped to 0-23 range",
            )
        )

        return wake_hour, sleep_hour

    def _derive_peak_hour(
        self, survey: ComprehensiveSurveyInput, dimensions: BehavioralDimensions
    ) -> int:
        """
        Derive peak usage hour with stronger chronotype alignment.

        We still respect the survey's reported peak usage time, but evening
        chronotypes are shifted later and morning chronotypes earlier.
        """
        base_peak = SurveyMappings.PEAK_USAGE_TO_HOUR[survey.peak_usage_time]

        # Convert chronotype score [0,1] into a signed shift roughly [-3, +3]
        signed_shift = round((dimensions.chronotype_score - 0.5) * 6)

        peak_hour = (base_peak + signed_shift) % 24

        self.citations.append(
            LiteratureReference(
                parameter="temporal_peak_hour",
                formula=f"(base_peak + signed_shift) % 24 = ({base_peak} + {signed_shift}) % 24",
                sources=["Randjelovic et al. (2021) - chronotype effects on timing"],
                notes="Chronotype has stronger influence on peak timing than before",
            )
        )

        return int(peak_hour)

    def _derive_late_night_probability(self, dimensions: BehavioralDimensions) -> float:
        """
        Derive probability of usage after midnight.

        Source: Roehrick et al. (2023) - RR=1.61 for night usage
        Formula: P(late_night) = 0.05 + 0.25 * chronotype_score
        """
        base_prob = 0.05
        chronotype_effect = 0.25 * dimensions.chronotype_score
        prob = base_prob + chronotype_effect

        self.citations.append(
            LiteratureReference(
                parameter="late_night_probability",
                formula="0.05 + 0.25 * chronotype_score",
                sources=["Roehrick et al. (2023) - RR=1.61 for night bin"],
                notes="Evening types have ~6x higher late night usage",
            )
        )

        return np.clip(prob, 0.0, 0.40)

    def _derive_daily_screen_time(
        self, dimensions: BehavioralDimensions, survey: ComprehensiveSurveyInput
    ) -> float:
        """
        Derive expected daily screen time in minutes.

        Source: Randjelovic et al. (2021) - 4.16 hrs morning vs 5.81 hrs evening
        Formula: Linear interpolation based on chronotype and reported usage
        """
        # Base from survey report
        reported_minutes = SurveyMappings.SCREEN_TIME_TO_MINUTES[
            survey.daily_screen_time
        ]

        # Chronotype adjustment
        morning_baseline = LiteratureConstants.MORNING_TYPE_SCREEN_TIME_MIN
        evening_baseline = LiteratureConstants.EVENING_TYPE_SCREEN_TIME_MIN
        chronotype_baseline = morning_baseline + dimensions.chronotype_score * (
            evening_baseline - morning_baseline
        )

        # Blend reported with chronotype expectation (trust survey more)
        daily_minutes = 0.70 * reported_minutes + 0.30 * chronotype_baseline

        self.citations.append(
            LiteratureReference(
                parameter="total_daily_minutes",
                formula="0.70 * reported + 0.30 * chronotype_baseline",
                sources=[
                    "Randjelovic et al. (2021) - 4.16 hrs morning, 5.81 hrs evening",
                    "Survey Q15 - self-reported screen time",
                ],
                notes="Blends self-report with literature expectations",
            )
        )

        return daily_minutes

    def _derive_pickups_per_day(self, dimensions: BehavioralDimensions) -> float:
        """
        Derive expected phone pickups per day.

        Source: Toth et al. (2025) - 115-177 unlocks/day for heavy users
        Formula: Interpolation from light (50) to heavy (150) based on intensity
        """
        light_pickups = 50
        heavy_pickups = 150

        pickups = light_pickups + dimensions.usage_intensity * (
            heavy_pickups - light_pickups
        )

        self.citations.append(
            LiteratureReference(
                parameter="pickups_per_day",
                formula=f"{light_pickups} + usage_intensity * {heavy_pickups - light_pickups}",
                sources=["Toth et al. (2025) - 115-177 unlocks/day median"],
                notes="Pickups include glances that don't become full sessions",
            )
        )

        return pickups

    def _derive_sessions_per_day(self, dimensions: BehavioralDimensions) -> float:
        """
        Derive expected unlock sessions per day.

        Source: Toth et al. (2025) - 58-72 sessions/day median
        Formula: Interpolation based on usage intensity
        """
        light_sessions = LiteratureConstants.SESSIONS_PER_DAY_LIGHT
        heavy_sessions = LiteratureConstants.SESSIONS_PER_DAY_HEAVY

        sessions = light_sessions + dimensions.usage_intensity * (
            heavy_sessions - light_sessions
        )

        self.citations.append(
            LiteratureReference(
                parameter="sessions_per_day",
                formula=f"{light_sessions} + usage_intensity * {heavy_sessions - light_sessions}",
                sources=["Toth et al. (2025) - 58-72 sessions/day median"],
                notes="Sessions = unlocked interactions, excludes lock-screen glances",
            )
        )

        return sessions

    def _derive_session_duration(
        self, dimensions: BehavioralDimensions, survey: ComprehensiveSurveyInput
    ) -> Tuple[float, float]:
        """
        Derive mean session duration and standard deviation.

        Source: Toth et al. (2025) - 46 sec single, 104 sec loops
        Formula: Weighted average based on granularity and session type
        """
        # Base from session type
        base_duration = SurveyMappings.SESSION_TYPE_TO_DURATION[survey.session_type]

        # Adjust by attentional granularity
        # High granularity = shorter sessions
        granularity_factor = 1.0 - (dimensions.attentional_granularity * 0.5)

        mean_duration = base_duration * granularity_factor

        # Standard deviation typically 60-80% of mean for log-normal distribution
        duration_std = mean_duration * 0.70

        self.citations.append(
            LiteratureReference(
                parameter="session_duration",
                formula=f"base_duration * (1 - granularity * 0.5) = {base_duration} * {granularity_factor:.2f}",
                sources=[
                    "Toth et al. (2025) - 46 sec single, 104 sec loops",
                    "Survey Q17 - session type preference",
                ],
                notes="High attentional granularity reduces duration by up to 50%",
            )
        )

        return mean_duration, duration_std

    def _derive_glance_probability(self, dimensions: BehavioralDimensions) -> float:
        """
        Derive probability that a session is a glance (<15 sec).

        Source: Toth et al. (2025) - 17% lock-screen, 30-35% short interactions
        Formula: Base rate adjusted by attentional granularity
        """
        base_glance = LiteratureConstants.GLANCE_PROPORTION_BASE

        # High granularity increases glance probability
        glance_prob = base_glance + dimensions.attentional_granularity * 0.30

        self.citations.append(
            LiteratureReference(
                parameter="glance_probability",
                formula=f"{base_glance} + attentional_granularity * 0.30",
                sources=["Toth et al. (2025) - 17% lock-screen checks"],
                notes="Granular users show up to 55% glance rate",
            )
        )

        return np.clip(glance_prob, 0.10, 0.60)

    def _derive_inter_session_interval(self, sessions_per_day: float) -> float:
        """
        Derive mean inter-session interval in minutes.

        Source: Heitmayer & Lahlou (2021) - 290.5 sec between pickups
        Formula: Waking hours / sessions, adjusted by intensity
        """
        # Assume ~16 waking hours
        waking_minutes = 16 * 60

        # Calculate from session count
        interval = waking_minutes / sessions_per_day

        self.citations.append(
            LiteratureReference(
                parameter="inter_session_interval",
                formula=f"waking_minutes / sessions = {waking_minutes} / {sessions_per_day:.0f}",
                sources=["Heitmayer & Lahlou (2021) - 290.5 sec rhythm"],
                notes="Actual intervals are gamma-distributed, not constant",
            )
        )

        return interval

    def _derive_burst_probability(self, dimensions: BehavioralDimensions) -> float:
        """
        Derive probability of checking bursts (multiple checks within 5 min).

        Source: Toth et al. (2025) - 22% sessions become multi-app loops
        Formula: Base rate adjusted by intensity and granularity
        """
        base_burst = LiteratureConstants.LOOP_PROBABILITY

        # Intense, granular users burst more
        intensity_effect = dimensions.usage_intensity * 0.15
        granularity_effect = dimensions.attentional_granularity * 0.10

        burst_prob = base_burst + intensity_effect + granularity_effect

        self.citations.append(
            LiteratureReference(
                parameter="burst_probability",
                formula=f"{base_burst} + intensity*0.15 + granularity*0.10",
                sources=["Toth et al. (2025) - 22% multi-app loops"],
                notes="High intensity users show clustering behavior",
            )
        )

        return np.clip(burst_prob, 0.10, 0.50)

    def _derive_work_multipliers(
        self, survey: ComprehensiveSurveyInput
    ) -> Tuple[float, float]:
        """
        Derive work context multipliers for duration and frequency.

        Source: Heitmayer & Lahlou (2021) - β=-36.02 at work
        Formula: Direct mapping from work restriction level
        """
        restriction_mult = SurveyMappings.WORK_RESTRICTION_TO_MULTIPLIER[
            survey.work_phone_restriction
        ]

        # Duration affected more than frequency (literature finding)
        duration_mult = restriction_mult
        frequency_mult = 0.60 + (0.40 * restriction_mult)  # Less suppression

        self.citations.append(
            LiteratureReference(
                parameter="work_multipliers",
                formula="duration from restriction; frequency = 0.60 + 0.40*restriction",
                sources=["Heitmayer & Lahlou (2021) - work β=-36.02"],
                notes="Duration varies more than frequency by context",
            )
        )

        return duration_mult, frequency_mult

    def _derive_home_evening_multiplier(
        self, survey: ComprehensiveSurveyInput
    ) -> float:
        """
        Derive home evening duration multiplier.

        Source: Heitmayer & Lahlou (2021) - β=+42.53 at home
        Formula: Direct mapping from evening session change
        """
        multiplier = SurveyMappings.EVENING_CHANGE_TO_MULTIPLIER[
            survey.evening_session_change
        ]

        self.citations.append(
            LiteratureReference(
                parameter="home_evening_multiplier",
                formula="Direct from Q20 evening session change",
                sources=["Heitmayer & Lahlou (2021) - home β=+42.53"],
                notes="Evening 'stickiness' effect",
            )
        )

        return multiplier

    def _derive_commute_multiplier(self, survey: ComprehensiveSurveyInput) -> float:
        """
        Derive commute frequency multiplier based on mode.

        Source: Katevas et al. - usage varies by transit mode
        Formula: Mode-specific multipliers
        """
        mode_multipliers = {
            CommuteMode.DRIVING: 0.10,  # Safety constraint
            CommuteMode.PUBLIC_TRANSIT: 1.30,  # "Dead time" increases use
            CommuteMode.WALKING_BIKING: 0.25,  # Movement suppression
            CommuteMode.MIXED: 0.70,
            CommuteMode.STAY_HOME: 1.00,
            CommuteMode.WORK_FROM_HOME: 1.00,
        }

        multiplier = mode_multipliers.get(survey.commute_mode, 1.0)

        self.citations.append(
            LiteratureReference(
                parameter="commute_multiplier",
                formula="Mode-specific: transit=1.3, walking=0.25, driving=0.10",
                sources=["Katevas et al. - activity-based suppression"],
                notes="Transit enables use; movement/driving suppresses",
            )
        )

        return multiplier

    def _derive_active_suppression(self, survey: ComprehensiveSurveyInput) -> float:
        """
        Derive suppression factor during physical activity.

        Source: Chen et al. (2022) - 2% usage while exercising
        Formula: Fixed high suppression (0.08 = 92% reduction)
        """
        suppression = 0.08  # Only 8% of normal usage during exercise

        self.citations.append(
            LiteratureReference(
                parameter="active_suppression",
                formula="0.08 (92% reduction)",
                sources=["Chen et al. (2022) - 2% usage during exercise"],
                notes="Physical activity strongly suppresses phone use",
            )
        )

        return suppression

    def _derive_app_weights(
        self, survey: ComprehensiveSurveyInput
    ) -> Dict[AppCategory, float]:
        """
        Derive app category weights by triangulating multiple survey signals.
        No ranking assumed - presence in a list = equal weight from that source.
        """
        USED_CATEGORIES = [
            AppCategory.SOCIAL_MEDIA,
            AppCategory.MESSAGING,
            AppCategory.VIDEO_STREAMING,
            AppCategory.MUSIC_AUDIO,
            AppCategory.MAPS_NAVIGATION,
            AppCategory.PRODUCTIVITY_WORK,
            AppCategory.NEWS_READING,
            AppCategory.GAMES,
            AppCategory.SHOPPING,
        ]

        # Base weights
        weights = {cat: 0.02 for cat in USED_CATEGORIES}

        # Unified string-to-category mapping
        str_to_category = {
            # Category names
            "social_media": AppCategory.SOCIAL_MEDIA,
            "messaging": AppCategory.MESSAGING,
            "video_streaming": AppCategory.VIDEO_STREAMING,
            "music_audio": AppCategory.MUSIC_AUDIO,
            "music_podcasts": AppCategory.MUSIC_AUDIO,
            "maps_navigation": AppCategory.MAPS_NAVIGATION,
            "productivity_work": AppCategory.PRODUCTIVITY_WORK,
            "productivity": AppCategory.PRODUCTIVITY_WORK,
            "news_reading": AppCategory.NEWS_READING,
            "reading_news": AppCategory.NEWS_READING,
            "games": AppCategory.GAMES,
            "gaming": AppCategory.GAMES,
            "shopping": AppCategory.SHOPPING,
            # Usage reason variants
            "watching_videos": AppCategory.VIDEO_STREAMING,
            "messaging_talking": AppCategory.MESSAGING,
        }

        def add_weight(source_list, weight_per_item):
            """Add weight for each category found in source list."""
            for item in source_list:
                # Handle both AppCategory enums and strings
                if item in weights:
                    weights[item] += weight_per_item
                elif isinstance(item, str) and item in str_to_category:
                    weights[str_to_category[item]] += weight_per_item

        # --- Signal 1: most_used_categories (STRONG, equal weight) ---
        add_weight(survey.most_used_categories, 0.15)

        # --- Signal 2: usage_reasons (MEDIUM) ---
        add_weight(survey.usage_reasons, 0.10)

        # --- Signal 3: commute_activities (LOW) ---
        add_weight(survey.commute_activities, 0.05)

        # --- Signal 4: evening_activities_increase (LOW) ---
        add_weight(survey.evening_activities_increase, 0.05)

        # Normalize to sum to 1.0
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _force_weights_sum_to_one(
        self, weights: Dict[AppCategory, float]
    ) -> Dict[AppCategory, float]:
        """
        Adjust weights to sum exactly to 1.0, handling floating point errors.
        """
        if not weights:
            return weights

        current_sum = sum(weights.values())

        if current_sum == 0:
            n = len(weights)
            return {k: 1.0 / n for k in weights}

        # Normalize
        weights = {k: v / current_sum for k, v in weights.items()}

        # Fix any remaining floating point error by adjusting the largest weight
        new_sum = sum(weights.values())
        if new_sum != 1.0:
            max_key = max(weights.keys(), key=lambda k: weights[k])  # Fixed!
            weights[max_key] += 1.0 - new_sum

        return weights

    def _derive_radius_of_gyration(
        self, survey: ComprehensiveSurveyInput, dimensions: BehavioralDimensions
    ) -> float:
        """
        Derive radius of gyration (mobility metric) in km.

        Source: Hackett et al. (2024) - RoG varies by area type
        Formula: Base by area type, adjusted by mobility diversity
        """
        base_rog = LiteratureConstants.RADIUS_OF_GYRATION_KM[survey.area_type]

        # Adjust by mobility diversity
        mobility_factor = 0.5 + dimensions.mobility_diversity
        rog = base_rog * mobility_factor

        self.citations.append(
            LiteratureReference(
                parameter="radius_of_gyration",
                formula=f"base_rog * (0.5 + mobility) = {base_rog} * {mobility_factor:.2f}",
                sources=["Hackett et al. (2024) - RoG by area type"],
                notes="Urban ~6km, suburban ~12km, rural ~20km baseline",
            )
        )

        return rog

    def _derive_location_entropy(self, dimensions: BehavioralDimensions) -> float:
        """
        Derive location entropy (diversity of visited places).

        Source: Yi et al. (2024) - entropy correlates with wellbeing
        Formula: Based on mobility diversity dimension
        """
        # Entropy typically 0.5-2.5 for daily movement
        min_entropy = 0.5
        max_entropy = 2.5

        entropy = min_entropy + dimensions.mobility_diversity * (
            max_entropy - min_entropy
        )

        self.citations.append(
            LiteratureReference(
                parameter="location_entropy",
                formula=f"{min_entropy} + mobility * {max_entropy - min_entropy}",
                sources=["Yi et al. (2024) - location entropy metrics"],
                notes="Higher entropy = more diverse location visits",
            )
        )

        return entropy

    def _derive_exploit_ratio(self, dimensions: BehavioralDimensions) -> float:
        """
        Derive expected exploit ratio (familiar vs new content).

        Source: Survey Q25 exploration style
        Formula: Inverse of novelty seeking
        """
        exploit_ratio = 1.0 - dimensions.novelty_seeking

        self.citations.append(
            LiteratureReference(
                parameter="exploit_ratio",
                formula="1.0 - novelty_seeking",
                sources=["Survey Q25 - exploration style"],
                notes="Used for validation: familiar app usage proportion",
            )
        )

        return exploit_ratio

    def _derive_fragmentation_index(self, dimensions: BehavioralDimensions) -> float:
        """
        Derive fragmentation index for validation.

        Formula: Based on attentional granularity
        """
        fragmentation = dimensions.attentional_granularity

        self.citations.append(
            LiteratureReference(
                parameter="fragmentation_index",
                formula="= attentional_granularity",
                sources=["Derived dimension"],
                notes="Higher = more fragmented, shorter sessions",
            )
        )

        return fragmentation

    def get_citations(self) -> List[LiteratureReference]:
        """Return all citations generated during derivation."""
        return self.citations


class ScheduleGenerator:
    """
    Generates daily schedules with time segments and contexts.
    Creates a sequence of ScheduleSegments covering 24 hours,
    with appropriate activities and contexts based on survey input.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        context_enhancer: Optional["LLMContextEnhancer"] = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.context_enhancer = context_enhancer
        self._city_coords_cache: Dict[str, Tuple[float, float]] = {}

    def _should_commute_today(
        self,
        survey: ComprehensiveSurveyInput,
        day_type: str,
        context_modifiers: Optional[dict] = None,
    ) -> bool:
        """
        Determine if this specific day includes commuting.

        Grounding: Survey asks about commute DAYS per week, not every day.
        Must translate frequency to daily probability.
        """
        if day_type == "weekend":
            return False

        if context_modifiers and "commutes_today" in context_modifiers:
            return context_modifiers["commutes_today"]

        commute_probability = {
            CommuteDays.ZERO: 0.0,
            CommuteDays.ONE_TWO: 0.20,
            CommuteDays.THREE_FOUR: 0.55,
            CommuteDays.FIVE_PLUS: 0.80,
        }.get(survey.commute_days, 0.55)

        return self.rng.random() < commute_probability

    def _validate_segments(
        self, segments: List[ScheduleSegment]
    ) -> List[ScheduleSegment]:
        """Drop zero/negative-duration segments and sort."""
        if not segments:
            return segments

        valid_segments: List[ScheduleSegment] = []

        for segment in segments:
            start_minutes = segment.start_time.hour * 60 + segment.start_time.minute
            end_minutes = segment.end_time.hour * 60 + segment.end_time.minute

            if end_minutes <= start_minutes:
                continue

            valid_segments.append(segment)

        valid_segments.sort(key=lambda s: (s.start_time.hour, s.start_time.minute))
        return valid_segments

    def generate(
        self,
        survey: ComprehensiveSurveyInput,
        parameters: BehavioralParameters,
        date: str,
        day_type: Literal["weekday", "weekend"] = "weekday",
        context_modifiers: Optional[dict] = None,
    ) -> DailySchedule:
        """
        Generate a complete daily schedule.

        Args:
            survey: Original survey input
            parameters: Derived behavioral parameters
            date: Date string (YYYY-MM-DD)
            day_type: "weekday" or "weekend"

        Returns:
            DailySchedule with all segments populated
        """
        if day_type == "weekday":
            segments = self._generate_weekday_schedule(
                survey, parameters, context_modifiers
            )
        else:
            segments = self._generate_weekend_schedule(survey, parameters)

        segments = self._validate_segments(segments)
        segments = self._assign_locations(segments, survey)

        return DailySchedule(
            date=date, day_type=day_type, segments=segments, is_valid=True
        )

    def _generate_weekday_schedule(
        self,
        survey: ComprehensiveSurveyInput,
        parameters: BehavioralParameters,
        context_modifiers: Optional[dict] = None,
    ) -> List[ScheduleSegment]:
        """Generate typical weekday schedule with proper commute handling."""
        segments: List[ScheduleSegment] = []

        wake_hour = int(parameters.waking_hour_start)
        sleep_hour = int(parameters.sleep_hour)

        is_commute_day = self._should_commute_today(
            survey, "weekday", context_modifiers
        )

        commute_minutes = SurveyMappings.COMMUTE_TIME_TO_MINUTES[survey.commute_time]
        commute_context = self._get_commute_context(survey.commute_mode)

        # Sleep from midnight until wake-up
        segments.append(
            ScheduleSegment(
                start_time=time(0, 0),
                end_time=time(wake_hour, 0),
                activity=ActivityType.SLEEPING,
                context=ContextType.HOME_NIGHT,
                location_label="home",
                phone_accessible=False,
                duration_multiplier=0.01,
                frequency_multiplier=0.01,
            )
        )

        # Morning routine
        morning_end = min(wake_hour + 1, 23)
        segments.append(
            ScheduleSegment(
                start_time=time(wake_hour, 0),
                end_time=time(morning_end, 0),
                activity=ActivityType.MORNING_ROUTINE,
                context=ContextType.HOME_MORNING,
                location_label="home",
                phone_accessible=True,
                duration_multiplier=1.0,
                frequency_multiplier=1.0,
            )
        )

        lunch_start = 12
        lunch_end = 13

        # Defaults for non-commute / work-from-home days
        work_location = "home"
        work_context = ContextType.WORK_FREE
        work_start = morning_end

        if is_commute_day and commute_minutes > 10:
            work_location = "work"
            work_context = (
                ContextType.WORK_RESTRICTED
                if survey.work_phone_restriction
                == WorkPhoneRestriction.BRIEFLY_WHEN_NECESSARY
                else ContextType.WORK_FREE
            )

            # Only create a morning commute if there is actual room before lunch
            available_pre_lunch_minutes = max(0, (lunch_start - morning_end) * 60)

            if available_pre_lunch_minutes >= 15:
                actual_commute_minutes = min(
                    commute_minutes, available_pre_lunch_minutes
                )
                commute_hours = max(1, math.ceil(actual_commute_minutes / 60))

                commute_start = morning_end
                commute_end = min(commute_start + commute_hours, lunch_start)

                if commute_end > commute_start:
                    segments.append(
                        ScheduleSegment(
                            start_time=time(commute_start, 0),
                            end_time=time(commute_end, 0),
                            activity=ActivityType.COMMUTING,
                            context=commute_context,
                            location_label="commute",
                            phone_accessible=True,
                            duration_multiplier=LiteratureConstants.CONTEXT_DURATION_MULTIPLIERS[
                                commute_context
                            ],
                            frequency_multiplier=LiteratureConstants.CONTEXT_FREQUENCY_MULTIPLIERS[
                                commute_context
                            ],
                        )
                    )
                    work_start = commute_end

        # Pre-lunch work if there is room
        if work_start < lunch_start:
            segments.append(
                ScheduleSegment(
                    start_time=time(work_start, 0),
                    end_time=time(lunch_start, 0),
                    activity=ActivityType.WORKING,
                    context=work_context,
                    location_label=work_location,
                    phone_accessible=True,
                    duration_multiplier=LiteratureConstants.CONTEXT_DURATION_MULTIPLIERS[
                        work_context
                    ],
                    frequency_multiplier=LiteratureConstants.CONTEXT_FREQUENCY_MULTIPLIERS[
                        work_context
                    ],
                )
            )

        # Lunch
        segments.append(
            ScheduleSegment(
                start_time=time(lunch_start, 0),
                end_time=time(lunch_end, 0),
                activity=ActivityType.LUNCH_BREAK,
                context=(
                    ContextType.PUBLIC_PLACE
                    if is_commute_day
                    else ContextType.HOME_MORNING
                ),
                location_label=work_location,
                phone_accessible=True,
                duration_multiplier=1.0,
                frequency_multiplier=1.0,
            )
        )

        # Afternoon work
        segments.append(
            ScheduleSegment(
                start_time=time(13, 0),
                end_time=time(17, 0),
                activity=ActivityType.WORKING,
                context=work_context,
                location_label=work_location,
                phone_accessible=True,
                duration_multiplier=LiteratureConstants.CONTEXT_DURATION_MULTIPLIERS[
                    work_context
                ],
                frequency_multiplier=LiteratureConstants.CONTEXT_FREQUENCY_MULTIPLIERS[
                    work_context
                ],
            )
        )

        # Evening commute
        evening_start = 17
        if is_commute_day and commute_minutes > 10:
            commute_home_end = min(17 + int(math.ceil(commute_minutes / 60.0)), 20)
            if commute_home_end > 17:
                segments.append(
                    ScheduleSegment(
                        start_time=time(17, 0),
                        end_time=time(commute_home_end, 0),
                        activity=ActivityType.COMMUTING,
                        context=commute_context,
                        location_label="commute",
                        phone_accessible=True,
                        duration_multiplier=LiteratureConstants.CONTEXT_DURATION_MULTIPLIERS[
                            commute_context
                        ],
                        frequency_multiplier=LiteratureConstants.CONTEXT_FREQUENCY_MULTIPLIERS[
                            commute_context
                        ],
                    )
                )
                evening_start = commute_home_end

        # Handle sleep after midnight correctly
        sleep_hour_same_day = sleep_hour if sleep_hour > wake_hour else 24

        # Reserve the last hour before sleep for winding down
        wind_down_start = max(evening_start + 1, sleep_hour_same_day - 1)
        wind_down_start = min(wind_down_start, 23)

        if evening_start < wind_down_start:
            segments.append(
                ScheduleSegment(
                    start_time=time(evening_start, 0),
                    end_time=time(wind_down_start, 0),
                    activity=ActivityType.HOME_EVENING,
                    context=ContextType.HOME_EVENING,
                    location_label="home",
                    phone_accessible=True,
                    duration_multiplier=LiteratureConstants.CONTEXT_DURATION_MULTIPLIERS[
                        ContextType.HOME_EVENING
                    ],
                    frequency_multiplier=LiteratureConstants.CONTEXT_FREQUENCY_MULTIPLIERS[
                        ContextType.HOME_EVENING
                    ],
                )
            )

        # Winding down until midnight if user sleeps after midnight
        if sleep_hour <= wake_hour:
            if wind_down_start < 23:
                segments.append(
                    ScheduleSegment(
                        start_time=time(wind_down_start, 0),
                        end_time=time(23, 59),
                        activity=ActivityType.WINDING_DOWN,
                        context=ContextType.HOME_NIGHT,
                        location_label="home",
                        phone_accessible=True,
                        duration_multiplier=LiteratureConstants.CONTEXT_DURATION_MULTIPLIERS[
                            ContextType.HOME_NIGHT
                        ],
                        frequency_multiplier=LiteratureConstants.CONTEXT_FREQUENCY_MULTIPLIERS[
                            ContextType.HOME_NIGHT
                        ],
                    )
                )
        else:
            if wind_down_start < sleep_hour:
                segments.append(
                    ScheduleSegment(
                        start_time=time(wind_down_start, 0),
                        end_time=time(sleep_hour, 0),
                        activity=ActivityType.WINDING_DOWN,
                        context=ContextType.HOME_NIGHT,
                        location_label="home",
                        phone_accessible=True,
                        duration_multiplier=LiteratureConstants.CONTEXT_DURATION_MULTIPLIERS[
                            ContextType.HOME_NIGHT
                        ],
                        frequency_multiplier=LiteratureConstants.CONTEXT_FREQUENCY_MULTIPLIERS[
                            ContextType.HOME_NIGHT
                        ],
                    )
                )

        return segments

    def _generate_weekend_schedule(
        self, survey: ComprehensiveSurveyInput, parameters: BehavioralParameters
    ) -> List[ScheduleSegment]:
        """
        Generate typical weekend schedule.

        Weekend patterns differ from weekdays:
        - Later wake times (+1-2 hours typically)
        - No commute or work segments
        - More leisure and home time
        - Extended evening sessions
        """
        segments = []

        wake_hour = min(parameters.waking_hour_start + 1, 12)
        sleep_hour = parameters.sleep_hour

        if sleep_hour < wake_hour:
            sleep_hour += 24

        segments.append(
            ScheduleSegment(
                start_time=time(0, 0),
                end_time=time(wake_hour, 0),
                activity=ActivityType.SLEEPING,
                context=ContextType.HOME_NIGHT,
                location_label="home",
                phone_accessible=False,
                duration_multiplier=0.01,
                frequency_multiplier=0.01,
            )
        )

        morning_end = min(wake_hour + 1, 23)
        segments.append(
            ScheduleSegment(
                start_time=time(wake_hour, 0),
                end_time=time(morning_end, 0),
                activity=ActivityType.MORNING_ROUTINE,
                context=ContextType.HOME_MORNING,
                location_label="home",
                phone_accessible=True,
                duration_multiplier=LiteratureConstants.CONTEXT_DURATION_MULTIPLIERS[
                    ContextType.HOME_MORNING
                ],
                frequency_multiplier=LiteratureConstants.CONTEXT_FREQUENCY_MULTIPLIERS[
                    ContextType.HOME_MORNING
                ],
            )
        )

        lunch_hour = max(morning_end, 12)
        if morning_end < lunch_hour:
            segments.append(
                ScheduleSegment(
                    start_time=time(morning_end, 0),
                    end_time=time(lunch_hour, 0),
                    activity=ActivityType.LEISURE,
                    context=ContextType.HOME_MORNING,
                    location_label="home",
                    phone_accessible=True,
                    duration_multiplier=1.20,
                    frequency_multiplier=1.10,
                )
            )

        segments.append(
            ScheduleSegment(
                start_time=time(12, 0),
                end_time=time(13, 0),
                activity=ActivityType.LUNCH_BREAK,
                context=ContextType.HOME_MORNING,
                location_label="home",
                phone_accessible=True,
                duration_multiplier=1.0,
                frequency_multiplier=1.0,
            )
        )

        activity_days = SurveyMappings.PHYSICAL_ACTIVITY_TO_DAYS[
            survey.physical_activity_days
        ]

        if activity_days >= 3:
            segments.append(
                ScheduleSegment(
                    start_time=time(13, 0),
                    end_time=time(14, 30),
                    activity=ActivityType.EXERCISING,
                    context=ContextType.EXERCISING,
                    location_label="gym",
                    phone_accessible=True,
                    duration_multiplier=LiteratureConstants.CONTEXT_DURATION_MULTIPLIERS[
                        ContextType.EXERCISING
                    ],
                    frequency_multiplier=LiteratureConstants.CONTEXT_FREQUENCY_MULTIPLIERS[
                        ContextType.EXERCISING
                    ],
                )
            )
            segments.append(
                ScheduleSegment(
                    start_time=time(14, 30),
                    end_time=time(17, 0),
                    activity=ActivityType.LEISURE,
                    context=ContextType.PUBLIC_PLACE,
                    location_label="outside",
                    phone_accessible=True,
                    duration_multiplier=1.10,
                    frequency_multiplier=1.0,
                )
            )
        else:
            segments.append(
                ScheduleSegment(
                    start_time=time(13, 0),
                    end_time=time(15, 0),
                    activity=ActivityType.ERRANDS,
                    context=ContextType.PUBLIC_PLACE,
                    location_label="errands",
                    phone_accessible=True,
                    duration_multiplier=0.80,
                    frequency_multiplier=0.90,
                )
            )
            segments.append(
                ScheduleSegment(
                    start_time=time(15, 0),
                    end_time=time(17, 0),
                    activity=ActivityType.LEISURE,
                    context=ContextType.HOME_EVENING,
                    location_label="home",
                    phone_accessible=True,
                    duration_multiplier=1.20,
                    frequency_multiplier=1.10,
                )
            )

        wind_down_hour = min(max(20, sleep_hour - 1 if sleep_hour <= 24 else 23), 23)

        segments.append(
            ScheduleSegment(
                start_time=time(17, 0),
                end_time=time(wind_down_hour, 0),
                activity=ActivityType.HOME_EVENING,
                context=ContextType.HOME_EVENING,
                location_label="home",
                phone_accessible=True,
                duration_multiplier=LiteratureConstants.CONTEXT_DURATION_MULTIPLIERS[
                    ContextType.HOME_EVENING
                ]
                * 1.15,
                frequency_multiplier=LiteratureConstants.CONTEXT_FREQUENCY_MULTIPLIERS[
                    ContextType.HOME_EVENING
                ],
            )
        )

        actual_sleep_hour = min(sleep_hour if sleep_hour < 24 else 23, 23)
        if wind_down_hour < actual_sleep_hour:
            segments.append(
                ScheduleSegment(
                    start_time=time(wind_down_hour, 0),
                    end_time=time(actual_sleep_hour, 0),
                    activity=ActivityType.WINDING_DOWN,
                    context=ContextType.HOME_NIGHT,
                    location_label="home",
                    phone_accessible=True,
                    duration_multiplier=LiteratureConstants.CONTEXT_DURATION_MULTIPLIERS[
                        ContextType.HOME_NIGHT
                    ],
                    frequency_multiplier=LiteratureConstants.CONTEXT_FREQUENCY_MULTIPLIERS[
                        ContextType.HOME_NIGHT
                    ],
                )
            )

        return segments

    def _get_commute_context(self, commute_mode: CommuteMode) -> ContextType:
        """
        Treat all commuting as a single generic context for now.
        We intentionally ignore commute_mode in behavioral modeling.
        """
        _ = commute_mode
        return ContextType.COMMUTE_TRANSIT

    def _assign_locations(
        self, segments: List[ScheduleSegment], survey: ComprehensiveSurveyInput
    ) -> List[ScheduleSegment]:
        """
        Assign GPS coordinates to location labels based on the entered city.

        Commute is treated as movement along the home-work corridor rather than
        a single midpoint bubble.
        """
        city_lat, city_lon = self._get_city_coordinates(survey)

        # home = small neighborhood offset from city center
        home_lat = city_lat + float(self.rng.normal(0, 0.008))
        home_lon = city_lon + float(self.rng.normal(0, 0.008))

        work_distance_km = self._get_work_distance_km(survey.commute_time)

        # avoid division issues near poles
        lon_scale = max(0.1, math.cos(math.radians(home_lat)))
        km_per_lon_degree = 111.0 * lon_scale

        angle = float(self.rng.uniform(0, 2 * math.pi))
        work_lat = home_lat + (work_distance_km / 111.0) * math.sin(angle)
        work_lon = home_lon + (work_distance_km / km_per_lon_degree) * math.cos(angle)

        radius_km = LiteratureConstants.RADIUS_OF_GYRATION_KM.get(survey.area_type, 6.0)
        lat_scale = radius_km / 111.0
        lon_scale2 = radius_km / km_per_lon_degree

        location_coords = {
            "home": (home_lat, home_lon),
            "work": (work_lat, work_lon),
            "gym": (
                home_lat + float(self.rng.normal(0, lat_scale * 0.2)),
                home_lon + float(self.rng.normal(0, lon_scale2 * 0.2)),
            ),
            "errands": (
                home_lat + float(self.rng.normal(0, lat_scale * 0.3)),
                home_lon + float(self.rng.normal(0, lon_scale2 * 0.3)),
            ),
            "outside": (
                home_lat + float(self.rng.normal(0, lat_scale * 0.4)),
                home_lon + float(self.rng.normal(0, lon_scale2 * 0.4)),
            ),
            "social": (
                home_lat + float(self.rng.normal(0, lat_scale * 0.4)),
                home_lon + float(self.rng.normal(0, lon_scale2 * 0.4)),
            ),
            "leisure": (
                home_lat + float(self.rng.normal(0, lat_scale * 0.3)),
                home_lon + float(self.rng.normal(0, lon_scale2 * 0.3)),
            ),
        }

        commute_segments = [
            s for s in segments if (s.location_label or "").lower() == "commute"
        ]

        for segment in segments:
            key = (segment.location_label or "home").lower()

            if key == "commute":
                coords = self._get_commute_position(
                    segment=segment,
                    home_coords=(home_lat, home_lon),
                    work_coords=(work_lat, work_lon),
                    all_commute_segments=commute_segments,
                )
            else:
                coords = location_coords.get(
                    key,
                    (
                        home_lat + float(self.rng.normal(0, lat_scale * 0.2)),
                        home_lon + float(self.rng.normal(0, lon_scale2 * 0.2)),
                    ),
                )

            segment.location_coords = (float(coords[0]), float(coords[1]))

        return segments

    def _get_commute_position(
        self,
        segment: ScheduleSegment,
        home_coords: Tuple[float, float],
        work_coords: Tuple[float, float],
        all_commute_segments: List[ScheduleSegment],
    ) -> Tuple[float, float]:
        """
        Calculate a commute position along the home-work corridor.

        Morning commute goes home -> work.
        Evening commute goes work -> home.
        """
        home_lat, home_lon = home_coords
        work_lat, work_lon = work_coords

        segment_hour = segment.start_time.hour if segment.start_time else 8

        if segment_hour < 12:
            start_lat, start_lon = home_lat, home_lon
            end_lat, end_lon = work_lat, work_lon
        else:
            start_lat, start_lon = work_lat, work_lon
            end_lat, end_lon = home_lat, home_lon

        progress = self._calculate_commute_progress(segment, all_commute_segments)

        current_lat = start_lat + (end_lat - start_lat) * progress
        current_lon = start_lon + (end_lon - start_lon) * progress

        # Add a slight perpendicular deviation so commute is not a perfectly straight line
        route_length = math.sqrt(
            (end_lat - start_lat) ** 2 + (end_lon - start_lon) ** 2
        )
        if route_length > 0:
            perp_lat = -(end_lon - start_lon) / route_length
            perp_lon = (end_lat - start_lat) / route_length

            offset_magnitude = 0.002 * math.sin(progress * math.pi)
            current_lat += perp_lat * offset_magnitude * float(self.rng.uniform(-1, 1))
            current_lon += perp_lon * offset_magnitude * float(self.rng.uniform(-1, 1))

        return (float(current_lat), float(current_lon))

    def _calculate_commute_progress(
        self,
        segment: ScheduleSegment,
        all_commute_segments: List[ScheduleSegment],
    ) -> float:
        """
        Return a progress value in [0, 1] for this commute segment.
        """
        if not segment.start_time or not segment.end_time:
            return 0.5

        segment_hour = segment.start_time.hour
        is_morning = segment_hour < 12

        same_direction = [
            s
            for s in all_commute_segments
            if s.start_time and ((s.start_time.hour < 12) == is_morning)
        ]

        if len(same_direction) <= 1:
            return 0.5

        same_direction.sort(key=lambda s: s.start_time)

        try:
            index = same_direction.index(segment)
            return 0.1 + (0.8 * index / (len(same_direction) - 1))
        except ValueError:
            return 0.5

    def _generate_base_location(self, area_type: AreaType) -> Tuple[float, float]:
        """Generate a plausible base location for the area type."""
        if area_type == AreaType.URBAN:
            lat = self.rng.uniform(38.0, 42.0)
            lon = self.rng.uniform(-122.0, -74.0)
        elif area_type == AreaType.SUBURBAN:
            lat = self.rng.uniform(35.0, 45.0)
            lon = self.rng.uniform(-120.0, -75.0)
        else:
            lat = self.rng.uniform(33.0, 47.0)
            lon = self.rng.uniform(-115.0, -80.0)

        return lat, lon

    def _get_city_coordinates(
        self, survey: ComprehensiveSurveyInput
    ) -> Tuple[float, float]:
        city = (survey.city or "").strip()

        if city in self._city_coords_cache:
            return self._city_coords_cache[city]

        if self.context_enhancer is not None:
            coords = self.context_enhancer.get_city_coordinates(city)
        else:
            coords = (40.7128, -74.0060)

        self._city_coords_cache[city] = coords
        return coords

    def _get_work_distance_km(self, commute_time: CommuteTime) -> float:
        """
        Use commute_time only. Keep this simple.
        """
        mapping = {
            CommuteTime.ALMOST_NONE: 1.0,
            CommuteTime.LESS_THAN_30: 5.0,
            CommuteTime.THIRTY_TO_60: 12.0,
            CommuteTime.ONE_TO_TWO_HOURS: 25.0,
            CommuteTime.MORE_THAN_2_HOURS: 40.0,
        }
        return mapping.get(commute_time, 5.0)


class SessionPopulator:
    """
    Populates schedule segments with individual phone sessions.
    Uses literature-grounded distributions to generate realistic
    session timing, duration, and app usage patterns.

    Key sources:
    - Toth et al. (2025): Session structure (78% single, 22% loops)
    - Heitmayer & Lahlou (2021): 5-min rhythm, context invariance
    - Roehrick et al. (2023): Temporal distribution patterns
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.constants = LiteratureConstants()

    def populate(
        self,
        schedule: DailySchedule,
        parameters: BehavioralParameters,
        dimensions: BehavioralDimensions,
    ) -> List[PhoneSession]:
        """
        Generate all phone sessions for a daily schedule and reconcile
        the final total duration to the persona's target daily screen time.
        """

        all_sessions: List[PhoneSession] = []

        for segment in schedule.segments:
            if not segment.phone_accessible:
                continue

            segment_sessions = self._populate_segment(
                segment, parameters, dimensions, schedule.date
            )
            all_sessions.extend(segment_sessions)

        # Sort by timestamp
        all_sessions.sort(key=lambda s: s.timestamp)

        # Reconcile total duration to match target daily minutes
        all_sessions = self._reconcile_total_daily_duration(
            all_sessions,
            target_total_minutes=float(parameters.total_daily_minutes),
        )

        # Assign sequential session IDs
        for i, session in enumerate(all_sessions):
            session.session_id = f"{schedule.date}_{i:04d}"

        return all_sessions

    def _reconcile_total_daily_duration(
        self,
        sessions: List[PhoneSession],
        target_total_minutes: float,
    ) -> List[PhoneSession]:
        """
        Scale generated session durations so the final day total is close
        to the persona's target total_daily_minutes.

        Strategy:
        - Keep glance sessions relatively stable
        - Scale engaged sessions more aggressively
        - Clamp all durations to reasonable bounds
        """
        if not sessions:
            return sessions

        target_total_seconds = max(0.0, float(target_total_minutes) * 60.0)
        current_total_seconds = sum(float(s.duration_seconds) for s in sessions)

        if current_total_seconds <= 0 or target_total_seconds <= 0:
            return sessions

        # If already close enough, leave as-is
        ratio = target_total_seconds / current_total_seconds
        if 0.85 <= ratio <= 1.15:
            return sessions

        glance_sessions = [s for s in sessions if bool(s.is_glance)]
        engaged_sessions = [s for s in sessions if not bool(s.is_glance)]

        current_glance_total = sum(float(s.duration_seconds) for s in glance_sessions)
        current_engaged_total = sum(float(s.duration_seconds) for s in engaged_sessions)

        # Keep glances mostly intact, but allow modest movement
        target_glance_total = current_glance_total * min(max(ratio, 0.75), 1.25)
        target_engaged_total = max(0.0, target_total_seconds - target_glance_total)

        engaged_scale = (
            target_engaged_total / current_engaged_total
            if current_engaged_total > 0
            else 1.0
        )

        glance_scale = (
            target_glance_total / current_glance_total
            if current_glance_total > 0
            else 1.0
        )

        # Clamp scaling so one weird day does not create crazy sessions
        engaged_scale = min(max(engaged_scale, 0.5), 6.0)
        glance_scale = min(max(glance_scale, 0.8), 1.5)

        for session in sessions:
            original = float(session.duration_seconds)

            if bool(session.is_glance):
                new_duration = original * glance_scale
                new_duration = max(
                    LiteratureConstants.GLANCE_DURATION_MIN,
                    min(new_duration, LiteratureConstants.GLANCE_DURATION_MAX),
                )
            else:
                new_duration = original * engaged_scale
                new_duration = max(15.0, min(new_duration, 3600.0))

            session.duration_seconds = float(new_duration)

        # Small second pass to tighten final total if still off
        final_total = sum(float(s.duration_seconds) for s in sessions)
        if final_total > 0:
            correction = target_total_seconds / final_total
            correction = min(max(correction, 0.85), 1.2)

            for session in sessions:
                adjusted = float(session.duration_seconds) * correction
                if bool(session.is_glance):
                    adjusted = max(
                        LiteratureConstants.GLANCE_DURATION_MIN,
                        min(adjusted, LiteratureConstants.GLANCE_DURATION_MAX),
                    )
                else:
                    adjusted = max(15.0, min(adjusted, 3600.0))
                session.duration_seconds = float(adjusted)

        return sessions

    def _populate_segment(
        self,
        segment: ScheduleSegment,
        parameters: BehavioralParameters,
        dimensions: BehavioralDimensions,
        date: str,
    ) -> List[PhoneSession]:
        """
        Populate a schedule segment with phone sessions.

        Session count is based on:
        - expected sessions/day
        - segment duration
        - context frequency multiplier
        - activity suppression
        - temporal peak alignment
        """
        sessions: List[PhoneSession] = []

        if not segment.phone_accessible:
            return sessions

        start_minutes = segment.start_time.hour * 60 + segment.start_time.minute
        end_minutes = segment.end_time.hour * 60 + segment.end_time.minute
        segment_duration = max(0, end_minutes - start_minutes)

        if segment_duration <= 0:
            return sessions

        waking_hours = max(1.0, parameters.sleep_hour - parameters.waking_hour_start)
        total_waking_minutes = waking_hours * 60.0

        # Base expected number of sessions in this segment
        base_session_rate = float(parameters.sessions_per_day) / total_waking_minutes
        adjusted_rate = base_session_rate * float(segment.frequency_multiplier)

        # Apply activity suppression if available
        activity_suppression = LiteratureConstants.ACTIVITY_SUPPRESSION.get(
            segment.activity, 1.0
        )
        adjusted_rate *= float(activity_suppression)

        # Apply temporal peak weighting using the segment midpoint hour
        midpoint_minutes = start_minutes + (segment_duration / 2.0)
        midpoint_hour = int(midpoint_minutes // 60) % 24
        peak_factor = self._calculate_peak_factor(
            midpoint_hour,
            int(parameters.temporal_peak_hour),
        )

        adjusted_rate *= float(peak_factor)

        expected_sessions = adjusted_rate * float(segment_duration)

        # Sample actual number of sessions
        n_sessions = int(self.rng.poisson(max(0.05, expected_sessions)))

        # Hard cap commute so it never floods the day with short checks
        if segment.activity == ActivityType.COMMUTING:
            commute_cap = max(1, min(4, int(round(segment_duration / 20))))
            n_sessions = min(n_sessions, commute_cap)

        if n_sessions <= 0:
            return sessions

        timestamps = self._generate_session_timestamps(
            start_minutes,
            end_minutes,
            n_sessions,
            parameters,
        )

        for ts_minutes in timestamps:
            session = self._generate_single_session(
                ts_minutes,
                date,
                segment,
                parameters,
                dimensions,
            )
            sessions.append(session)

        return sessions

    def _calculate_peak_factor(self, hour: int, peak_hour: int) -> float:
        """
        Calculate temporal peak adjustment factor.

        Source: Roehrick et al. (2023) - diurnal patterns
        Uses Gaussian centered on peak hour.
        """
        # Gaussian with sigma=4 hours
        distance = min(abs(hour - peak_hour), 24 - abs(hour - peak_hour))
        factor = np.exp(-(distance**2) / (2 * 4**2))

        # Scale to range [0.5, 1.5]
        return 0.5 + factor

    def _generate_session_timestamps(
        self,
        start_minutes: int,
        end_minutes: int,
        n_sessions: int,
        parameters: BehavioralParameters,
    ) -> List[int]:
        """
        Generate timestamps for sessions using gamma-distributed intervals.

        Source: Heitmayer & Lahlou (2021) - 290.5 sec mean interval
        Gamma distribution captures the observed inter-check patterns.
        """
        if n_sessions <= 0:
            return []

        # Convert interval from minutes to work within segment
        mean_interval = parameters.mean_inter_session_interval_minutes

        # Gamma parameters (shape=2 gives realistic right-skewed distribution)
        shape = 2.0
        scale = mean_interval / shape

        # Generate intervals
        intervals = self.rng.gamma(shape, scale, size=n_sessions)

        # Convert to timestamps
        timestamps = []
        current_time = start_minutes + self.rng.uniform(0, mean_interval * 0.5)

        for interval in intervals:
            if current_time >= end_minutes:
                break
            timestamps.append(int(current_time))
            current_time += interval

        return timestamps

    def _generate_single_session(
        self,
        ts_minutes: int,
        date: str,
        segment: ScheduleSegment,
        parameters: BehavioralParameters,
        dimensions: BehavioralDimensions,
    ) -> PhoneSession:
        """
        Generate a single phone session with all attributes.

        Determines if session is glance vs engagement,
        duration, app category, and other attributes.
        """
        # Determine if this is a glance or engaged session
        is_glance = self.rng.random() < parameters.glance_probability

        # Determine if user-initiated or notification-triggered
        # Source: Heitmayer & Lahlou (2021) - 89% user-initiated
        is_user_initiated = self.rng.random() < LiteratureConstants.USER_INITIATED_RATIO

        # Generate duration based on session type
        if is_glance:
            # Glance duration: truncated normal
            # Source: Toth et al. (2025) - 5-25 sec for glances
            duration = self.rng.normal(
                LiteratureConstants.GLANCE_DURATION_MEAN,
                LiteratureConstants.GLANCE_DURATION_SD,
            )
            duration = np.clip(
                duration,
                LiteratureConstants.GLANCE_DURATION_MIN,
                LiteratureConstants.GLANCE_DURATION_MAX,
            )
        else:
            # Determine if single-interaction or loop
            # Source: Toth et al. (2025) - 78% single, 22% loops
            is_loop = self.rng.random() < parameters.checking_burst_probability

            if is_loop:
                base_duration = LiteratureConstants.LOOP_SESSION_MEAN
                duration_std = LiteratureConstants.LOOP_SESSION_SD
            else:
                base_duration = LiteratureConstants.SINGLE_SESSION_MEAN
                duration_std = LiteratureConstants.SINGLE_SESSION_SD

            # Apply context multiplier
            duration = self.rng.normal(base_duration, duration_std)
            duration *= segment.duration_multiplier

            # Ensure reasonable bounds
            duration = max(15, min(duration, 1800))  # 15 sec to 30 min

        # Select app category based on weights
        app_category = self._select_app_category(
            parameters,
            dimensions,
            segment,
            is_glance,
        )

        # Convert timestamp to datetime
        hours = ts_minutes // 60
        minutes = ts_minutes % 60
        if hours >= 24:
            hours -= 24

        timestamp = datetime.strptime(
            f"{date} {hours:02d}:{minutes:02d}:00", "%Y-%m-%d %H:%M:%S"
        )

        # Add some seconds randomization
        timestamp += timedelta(seconds=float(self.rng.integers(0, 60)))

        lat: Optional[float] = None
        lon: Optional[float] = None
        if segment.location_coords is not None:
            lat, lon = segment.location_coords[0], segment.location_coords[1]

        return PhoneSession(
            session_id="",  # Will be assigned later
            timestamp=timestamp,
            duration_seconds=float(duration),
            app_category=app_category,
            is_glance=is_glance,
            is_user_initiated=is_user_initiated,
            context=segment.context,
            activity=segment.activity,
            location_label=segment.location_label,
            latitude=lat,
            longitude=lon,
        )

    def _select_app_category(
        self,
        parameters: BehavioralParameters,
        dimensions: BehavioralDimensions,
        segment: ScheduleSegment,
        is_glance: bool,
    ) -> AppCategory:
        """
        Select app category based on persona weights plus light activity-aware rules.
        """

        _ = dimensions  # reserved for future tuning

        weights = {
            AppCategory.SOCIAL_MEDIA: parameters.weight_social,
            AppCategory.MESSAGING: parameters.weight_messaging,
            AppCategory.VIDEO_STREAMING: parameters.weight_video,
            AppCategory.MUSIC_AUDIO: parameters.weight_music,
            AppCategory.MAPS_NAVIGATION: parameters.weight_navigation,
            AppCategory.PRODUCTIVITY_WORK: parameters.weight_productivity,
            AppCategory.NEWS_READING: parameters.weight_news,
            AppCategory.GAMES: parameters.weight_games,
            AppCategory.SHOPPING: parameters.weight_shopping,
        }

        # Light context/activity rules
        if segment.activity == ActivityType.COMMUTING:
            weights[AppCategory.MUSIC_AUDIO] *= 3.0
            weights[AppCategory.MAPS_NAVIGATION] *= 2.2
            weights[AppCategory.MESSAGING] *= 1.4
            weights[AppCategory.PRODUCTIVITY_WORK] *= 0.5
            weights[AppCategory.VIDEO_STREAMING] *= 0.15
            weights[AppCategory.GAMES] *= 0.20
            weights[AppCategory.SHOPPING] *= 0.30

        elif segment.activity == ActivityType.WORKING:
            weights[AppCategory.PRODUCTIVITY_WORK] *= 2.5
            weights[AppCategory.MESSAGING] *= 1.2
            weights[AppCategory.NEWS_READING] *= 1.1
            weights[AppCategory.VIDEO_STREAMING] *= 0.25
            weights[AppCategory.GAMES] *= 0.15
            weights[AppCategory.SHOPPING] *= 0.30

        elif segment.activity == ActivityType.LUNCH_BREAK:
            weights[AppCategory.MESSAGING] *= 1.5
            weights[AppCategory.SOCIAL_MEDIA] *= 1.3
            weights[AppCategory.NEWS_READING] *= 1.2
            weights[AppCategory.PRODUCTIVITY_WORK] *= 0.8

        elif segment.activity == ActivityType.ERRANDS:
            weights[AppCategory.MAPS_NAVIGATION] *= 1.8
            weights[AppCategory.SHOPPING] *= 2.0
            weights[AppCategory.MESSAGING] *= 1.1
            weights[AppCategory.GAMES] *= 0.35
            weights[AppCategory.VIDEO_STREAMING] *= 0.35

        elif segment.activity in {
            ActivityType.HOME_EVENING,
            ActivityType.LEISURE,
            ActivityType.WINDING_DOWN,
        }:
            weights[AppCategory.VIDEO_STREAMING] *= 1.8
            weights[AppCategory.SOCIAL_MEDIA] *= 1.5
            weights[AppCategory.GAMES] *= 1.4
            weights[AppCategory.MUSIC_AUDIO] *= 1.2
            weights[AppCategory.PRODUCTIVITY_WORK] *= 0.5

        # Glances still bias toward quick-check apps
        if is_glance:
            weights[AppCategory.MESSAGING] *= 2.0
            weights[AppCategory.SOCIAL_MEDIA] *= 1.5
            weights[AppCategory.NEWS_READING] *= 1.15
            weights[AppCategory.VIDEO_STREAMING] *= 0.3
            weights[AppCategory.GAMES] *= 0.2

        # Normalize safely
        categories = list(weights.keys())
        safe_weights = [max(weights[c], 0.001) for c in categories]
        total = sum(safe_weights)
        probs = [w / total for w in safe_weights]

        idx = self.rng.choice(len(categories), p=probs)
        return categories[idx]


class BehaviorSimulationEngine:
    """
    Main orchestrator for the behavior simulation pipeline.
    Coordinates the transformation:
    Survey Input → Dimensions → Parameters → Schedule → Sessions

    All transformations are literature-grounded and documented.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize the simulation engine.

        Args:
            seed: Random seed for reproducibility
            llm_client: Optional LLM client for contextual day variation
        """
        self.dimension_extractor = DimensionExtractor()
        self.parameter_deriver = ParameterDeriver()
        self.schedule_generator = ScheduleGenerator(seed)
        self.session_populator = SessionPopulator(seed)
        self.context_enhancer = LLMContextEnhancer(llm_client)
        self.seed = seed

    def process_survey(
        self, survey: ComprehensiveSurveyInput
    ) -> Tuple[BehavioralDimensions, BehavioralParameters]:
        """
        Process survey input to extract dimensions and derive parameters.

        Args:
            survey: Complete survey input

        Returns:
            Tuple of (BehavioralDimensions, BehavioralParameters)
        """
        dimensions = self.dimension_extractor.extract(survey)
        parameters = self.parameter_deriver.derive(dimensions, survey)

        return dimensions, parameters

    DayType = Literal["weekday", "weekend"]

    def generate_day(
        self,
        survey: ComprehensiveSurveyInput,
        parameters: BehavioralParameters,
        dimensions: BehavioralDimensions,
        date: str,
        day_type: DayType = "weekday",
    ) -> Tuple[DailySchedule, List[PhoneSession]]:
        """
        Generate a complete day of simulated phone usage.

        Args:
            survey: Original survey input
            parameters: Derived behavioral parameters
            dimensions: Extracted behavioral dimensions
            date: Date string (YYYY-MM-DD)
            day_type: "weekday" or "weekend"

        Returns:
            Tuple of (DailySchedule, List[PhoneSession])
        """
        day_context = self.context_enhancer.get_day_context(
            survey=survey,
            dimensions=dimensions,
            day_type=day_type,
            date_str=date,
        )

        schedule = self.schedule_generator.generate(
            survey=survey,
            parameters=parameters,
            date=date,
            day_type=day_type,
            context_modifiers=day_context.get("context_modifiers"),
        )

        sessions = self.session_populator.populate(schedule, parameters, dimensions)

        return schedule, sessions

    def generate_week(
        self,
        survey: ComprehensiveSurveyInput,
        start_date: str,
    ) -> List[Tuple[DailySchedule, List[PhoneSession]]]:
        """
        Generate a full week of simulated phone usage.

        Args:
            survey: Complete survey input
            start_date: Start date string (YYYY-MM-DD), should be a Monday

        Returns:
            List of 7 (DailySchedule, List[PhoneSession]) tuples
        """
        dimensions, parameters = self.process_survey(survey)

        results = []
        base_date = datetime.strptime(start_date, "%Y-%m-%d")

        for i in range(7):
            current_date = base_date + timedelta(days=i)
            date_str = current_date.strftime("%Y-%m-%d")
            day_of_week = current_date.weekday()
            day_type = "weekend" if day_of_week >= 5 else "weekday"

            daily_result = self.generate_day(
                survey=survey,
                parameters=parameters,
                dimensions=dimensions,
                date=date_str,
                day_type=day_type,
            )
            results.append(daily_result)

        return results

    def generate_study_period(
        self,
        survey: ComprehensiveSurveyInput,
        start_date: str,
        num_days: int = 14,
    ) -> Dict[str, Any]:
        """
        Generate a full study period of simulated phone usage.

        Args:
            survey: Complete survey input
            start_date: Start date string (YYYY-MM-DD)
            num_days: Number of days to simulate (default 14)

        Returns:
            Dictionary containing:
                - dimensions: BehavioralDimensions
                - parameters: BehavioralParameters
                - daily_data: List of (DailySchedule, List[PhoneSession]) tuples
                - summary_stats: Aggregated statistics
        """
        dimensions, parameters = self.process_survey(survey)

        daily_data = []
        base_date = datetime.strptime(start_date, "%Y-%m-%d")

        for i in range(num_days):
            current_date = base_date + timedelta(days=i)
            date_str = current_date.strftime("%Y-%m-%d")

            day_of_week = current_date.weekday()
            day_type = "weekend" if day_of_week >= 5 else "weekday"

            schedule, sessions = self.generate_day(
                survey, parameters, dimensions, date_str, day_type
            )
            daily_data.append((schedule, sessions))

        # Compute summary statistics
        summary_stats = self._compute_summary_stats(daily_data, parameters)

        return {
            "dimensions": dimensions,
            "parameters": parameters,
            "daily_data": daily_data,
            "summary_stats": summary_stats,
        }

    def _compute_summary_stats(
        self,
        daily_data: List[Tuple[DailySchedule, List[PhoneSession]]],
        parameters: BehavioralParameters,
    ) -> Dict[str, Any]:
        """
        Compute summary statistics across the study period.

        Used for validation against expected parameters.
        """
        all_sessions = []
        daily_session_counts = []
        daily_screen_times = []

        for schedule, sessions in daily_data:
            all_sessions.extend(sessions)
            daily_session_counts.append(len(sessions))
            daily_screen_times.append(sum(s.duration_seconds for s in sessions) / 60)

        # Session statistics
        total_sessions = len(all_sessions)
        mean_sessions_per_day = (
            np.mean(daily_session_counts) if daily_session_counts else 0
        )
        std_sessions_per_day = (
            np.std(daily_session_counts) if daily_session_counts else 0
        )

        # Screen time statistics
        mean_screen_time_min = np.mean(daily_screen_times) if daily_screen_times else 0
        std_screen_time_min = np.std(daily_screen_times) if daily_screen_times else 0

        # Session duration statistics
        durations = [s.duration_seconds for s in all_sessions]
        mean_duration = np.mean(durations) if durations else 0
        median_duration = np.median(durations) if durations else 0

        # Glance statistics
        glance_count = sum(1 for s in all_sessions if s.is_glance)
        glance_ratio = glance_count / total_sessions if total_sessions > 0 else 0

        # App category distribution
        app_counts = defaultdict(int)
        for session in all_sessions:
            app_counts[session.app_category.value] += 1

        app_distribution = {
            cat: count / total_sessions if total_sessions > 0 else 0
            for cat, count in app_counts.items()
        }

        # Context distribution
        context_counts = defaultdict(int)
        for session in all_sessions:
            context_counts[session.context.value] += 1

        context_distribution = {
            ctx: count / total_sessions if total_sessions > 0 else 0
            for ctx, count in context_counts.items()
        }

        # Temporal distribution (hourly)
        hourly_counts = defaultdict(int)
        for session in all_sessions:
            hourly_counts[session.timestamp.hour] += 1

        hourly_distribution = {
            hour: count / total_sessions if total_sessions > 0 else 0
            for hour, count in hourly_counts.items()
        }

        return {
            "total_sessions": total_sessions,
            "num_days": len(daily_data),
            "mean_sessions_per_day": float(mean_sessions_per_day),
            "std_sessions_per_day": float(std_sessions_per_day),
            "mean_screen_time_minutes": float(mean_screen_time_min),
            "std_screen_time_minutes": float(std_screen_time_min),
            "mean_session_duration_seconds": float(mean_duration),
            "median_session_duration_seconds": float(median_duration),
            "glance_ratio": float(glance_ratio),
            "app_distribution": app_distribution,
            "context_distribution": context_distribution,
            "hourly_distribution": hourly_distribution,
            # Validation comparisons
            "expected_sessions_per_day": parameters.sessions_per_day,
            "expected_screen_time_minutes": parameters.total_daily_minutes,
            "expected_glance_probability": parameters.glance_probability,
        }

    def get_citations(self) -> List[LiteratureReference]:
        """
        Get all literature citations used in parameter derivation.

        Returns:
            List of LiteratureReference objects
        """
        return self.parameter_deriver.get_citations()

    @staticmethod
    def create_sample_survey() -> ComprehensiveSurveyInput:
        """
        Create a sample survey input for testing and demonstration.
        Returns:
        ComprehensiveSurveyInput with typical values
        """

        return ComprehensiveSurveyInput(
            # Demographics
            age_range=AgeRange.AGE_25_34,
            city="Portland",
            occupation="Software engineer",
            area_type=AreaType.SUBURBAN,
            # Temporal patterns
            wake_time=WakeTime.SIX_TO_8,
            sleep_time=SleepTime.TEN_TO_12,
            chronotype_self_report=ChronotypeLabel.NEITHER,
            peak_usage_time=PeakUsageTime.EVENING,
            routine_structure=RoutineStructure.SOMEWHAT_STRUCTURED,
            # Mobility
            places_visited_daily=3,
            commute_days=CommuteDays.FIVE_PLUS,
            commute_mode=CommuteMode.PUBLIC_TRANSIT,
            physical_activity_days=PhysicalActivityDays.THREE_FOUR,
            commute_time=CommuteTime.THIRTY_TO_60,
            # Usage patterns
            daily_screen_time=ScreenTime.TWO_TO_4_HOURS,
            checking_frequency=CheckingFrequency.THREE_TO_4_PER_HOUR,
            session_type=SessionType.MIXED,
            glance_frequency=GlanceFrequency.SOMETIMES,
            work_phone_restriction=WorkPhoneRestriction.OCCASIONALLY,
            evening_session_change=EveningSessionChange.SOMEWHAT_LONGER,
            # App preferences
            evening_activities_increase=[
                AppCategory.SOCIAL_MEDIA,
                AppCategory.VIDEO_STREAMING,
                AppCategory.MUSIC_AUDIO,
            ],
            usage_reasons=[
                UsageReason.MESSAGING_TALKING,
                UsageReason.SOCIAL_MEDIA,
                UsageReason.READING_NEWS,
            ],
            commute_activities=[
                AppCategory.MESSAGING,
                AppCategory.MUSIC_AUDIO,
                AppCategory.NEWS_READING,
            ],
            most_used_categories=[
                AppCategory.MESSAGING,
                AppCategory.SOCIAL_MEDIA,
                AppCategory.NEWS_READING,
                AppCategory.VIDEO_STREAMING,
            ],
            exploration_style=ExplorationStyle.SOMETIMES_EXPLORE,
        )

    @staticmethod
    def generate_synthetic_data(
        survey: ComprehensiveSurveyInput,
        start_date: str = "2024-01-08",
        num_days: int = 7,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point for generating synthetic smartphone usage data.
        Args:
            survey: Complete survey input
            start_date: Start date for simulation (YYYY-MM-DD)
            num_days: Number of days to simulate
            seed: Random seed for reproducibility

        Returns:
            Dictionary containing all generated data and metadata
        """
        engine = BehaviorSimulationEngine(seed=seed)

        result = engine.generate_study_period(
            survey=survey,
            start_date=start_date,
            num_days=num_days,
        )

        # Add citations
        result["citations"] = engine.get_citations()

        return result

    @staticmethod
    def sessions_to_dataframe(
        daily_data: List[Tuple[DailySchedule, List[PhoneSession]]],
    ) -> List[Dict[str, Any]]:
        """
        Convert sessions to a list of dictionaries suitable for DataFrame creation.

        Args:
            daily_data: List of (DailySchedule, List[PhoneSession]) tuples

        Returns:
            List of dictionaries, one per session
        """
        records: List[Dict[str, Any]] = []

        for schedule, sessions in daily_data:
            for session in sessions:
                record = {
                    "session_id": session.session_id,
                    "date": schedule.date,
                    "day_type": schedule.day_type,
                    "timestamp": session.timestamp.isoformat(),
                    "hour": session.timestamp.hour,
                    "duration_seconds": session.duration_seconds,
                    "app_category": session.app_category.value,
                    "is_glance": session.is_glance,
                    "is_user_initiated": session.is_user_initiated,
                    "context": session.context.value,
                    "activity": session.activity.value,
                    "location_label": session.location_label,
                    "latitude": session.latitude,
                    "longitude": session.longitude,
                }
                records.append(record)

        return records

    @staticmethod
    def export_to_json(
        result: Dict[str, Any],
        filepath: str,
        include_sessions: bool = True,
    ) -> None:
        """
        Export simulation results to JSON file.
        Args:
        result: Output from generate_synthetic_data()
        filepath: Output file path
        include_sessions: Whether to include individual sessions
        """

        export_data = {
            "dimensions": {
                "chronotype_score": result["dimensions"].chronotype_score,
                "usage_intensity": result["dimensions"].usage_intensity,
                "attentional_granularity": result["dimensions"].attentional_granularity,
                "contextual_sensitivity": result["dimensions"].contextual_sensitivity,
                "social_orientation": result["dimensions"].social_orientation,
                "mobility_diversity": result["dimensions"].mobility_diversity,
                "routine_stability": result["dimensions"].routine_stability,
                "novelty_seeking": result["dimensions"].novelty_seeking,
            },
            "parameters": {
                "waking_hour_start": result["parameters"].waking_hour_start,
                "sleep_hour": result["parameters"].waking_hour_end,
                "temporal_peak_hour": result["parameters"].temporal_peak_hour,
                "total_daily_minutes": result["parameters"].total_daily_minutes,
                "sessions_per_day": result["parameters"].sessions_per_day,
                "mean_session_duration_seconds": result[
                    "parameters"
                ].mean_session_duration_seconds,
                "glance_probability": result["parameters"].glance_probability,
            },
            "summary_stats": result["summary_stats"],
        }

        if include_sessions:
            export_data["sessions"] = BehaviorSimulationEngine.sessions_to_dataframe(
                result["daily_data"]
            )

        # Convert citations
        export_data["citations"] = [
            {
                "parameter": c.parameter,
                "formula": c.formula,
                "sources": c.sources,
                "notes": c.notes,
            }
            for c in result.get("citations", [])
        ]

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)


class ValidationReport:
    """
    Generate validation reports comparing simulated data to literature benchmarks.
    """

    def __init__(self, summary_stats: Dict[str, Any], parameters: BehavioralParameters):
        self.stats = summary_stats
        self.params = parameters

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a validation report with comparisons to expected values.

        Returns:
            Dictionary containing validation metrics and pass/fail indicators
        """
        report = {
            "session_count_validation": self._validate_session_count(),
            "screen_time_validation": self._validate_screen_time(),
            "glance_ratio_validation": self._validate_glance_ratio(),
            "temporal_pattern_validation": self._validate_temporal_patterns(),
            "overall_validity": True,
        }

        # Check if all validations passed
        for key, value in report.items():
            if isinstance(value, dict) and not value.get("passed", True):
                report["overall_validity"] = False
                break

        return report

    def _validate_session_count(self) -> Dict[str, Any]:
        """Validate session counts against expected values."""
        observed = self.stats["mean_sessions_per_day"]
        expected = self.params.sessions_per_day

        # Allow 30% deviation (natural variation)
        tolerance = 0.30
        lower_bound = expected * (1 - tolerance)
        upper_bound = expected * (1 + tolerance)

        passed = lower_bound <= observed <= upper_bound

        return {
            "observed": observed,
            "expected": expected,
            "tolerance": tolerance,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "passed": passed,
            "message": f"Session count {'within' if passed else 'outside'} expected range",
        }

    def _validate_screen_time(self) -> Dict[str, Any]:
        """Validate screen time against expected values."""
        observed = self.stats["mean_screen_time_minutes"]
        expected = self.params.total_daily_minutes

        tolerance = 0.35  # Screen time can vary more
        lower_bound = expected * (1 - tolerance)
        upper_bound = expected * (1 + tolerance)

        passed = lower_bound <= observed <= upper_bound

        return {
            "observed": observed,
            "expected": expected,
            "tolerance": tolerance,
            "passed": passed,
            "message": f"Screen time {'within' if passed else 'outside'} expected range",
        }

    def _validate_glance_ratio(self) -> Dict[str, Any]:
        """Validate glance ratio against expected probability."""
        observed = self.stats["glance_ratio"]
        expected = self.params.glance_probability

        tolerance = 0.15  # Absolute tolerance for ratios

        passed = abs(observed - expected) <= tolerance

        return {
            "observed": observed,
            "expected": expected,
            "tolerance": tolerance,
            "passed": passed,
            "message": f"Glance ratio {'within' if passed else 'outside'} expected range",
        }

    def _validate_temporal_patterns(self) -> Dict[str, Any]:
        """Validate that temporal patterns show expected peak."""
        hourly = self.stats.get("hourly_distribution", {})

        if not hourly:
            return {"passed": True, "message": "No hourly data to validate"}

        # Find observed peak hour
        peak_hour = max(hourly.keys(), key=lambda h: hourly.get(h, 0))
        expected_peak = self.params.temporal_peak_hour

        # Allow 3-hour deviation
        tolerance_hours = 3
        distance = min(
            abs(peak_hour - expected_peak), 24 - abs(peak_hour - expected_peak)
        )

        passed = distance <= tolerance_hours

        return {
            "observed_peak_hour": peak_hour,
            "expected_peak_hour": expected_peak,
            "tolerance_hours": tolerance_hours,
            "passed": passed,
            "message": f"Peak hour {'near' if passed else 'far from'} expected",
        }


# ============================================================
# PERSONA ENGINE (Main Interface)
# ============================================================


# ============================================================
# PERSONA ENGINE (Main Interface)
# ============================================================

from datetime import date, datetime, time, timedelta
from typing import Optional, Tuple


class PersonaEngine:
    """
    Main engine that orchestrates persona generation.

    This is the primary interface used by the API server.
    """

    def __init__(self, llm_client: Optional[Any] = None, seed: Optional[int] = None):
        """Initialize the engine components."""
        self.dimension_extractor = DimensionExtractor()
        self.parameter_deriver = ParameterDeriver()
        self.context_enhancer = LLMContextEnhancer(llm_client, seed=seed)
        self.schedule_generator = ScheduleGenerator(
            seed=seed,
            context_enhancer=self.context_enhancer,
        )
        self.session_populator = SessionPopulator(seed)

    def generate_persona(
        self, survey: ComprehensiveSurveyInput
    ) -> Tuple[BehavioralDimensions, BehavioralParameters]:
        """
        Generate behavioral dimensions and parameters from survey input.

        Args:
            survey: The comprehensive survey input from the user

        Returns:
            Tuple of (BehavioralDimensions, BehavioralParameters)
        """
        # Step 1: Extract behavioral dimensions from survey
        dimensions = self.dimension_extractor.extract(survey)

        # Step 2: Derive concrete parameters from dimensions
        # Note: Check your ParameterDeriver.derive() method signature
        parameters = self.parameter_deriver.derive(dimensions, survey)

        return dimensions, parameters

    def generate_schedule(
        self,
        dimensions: BehavioralDimensions,
        parameters: BehavioralParameters,
        survey: ComprehensiveSurveyInput,
        day_type: Literal["weekday", "weekend"] = "weekday",
        target_date: Optional[date] = None,
    ) -> Optional[DailySchedule]:
        """
        Generate a daily schedule with phone sessions.

        Args:
            dimensions: The behavioral dimensions
            parameters: The behavioral parameters
            survey: The original survey input
            day_type: Either "weekday" or "weekend"
            target_date: The date for the schedule (defaults to today)

        Returns:
            DailySchedule with segments and phone sessions, or None on error
        """
        # Set default date if not provided
        if target_date is None:
            target_date = date.today()

        date_str = target_date.isoformat()

        try:
            # Step 1: Generate contextual modifiers for this day
            day_context = self.context_enhancer.get_day_context(
                survey=survey,
                dimensions=dimensions,
                day_type=day_type,
                date_str=date_str,
            )
            # TODO --- LLM Call
            # Step 2: Generate the daily schedule structure
            schedule = self.schedule_generator.generate(
                survey=survey,
                parameters=parameters,
                date=date_str,
                day_type=day_type,
                context_modifiers=day_context.get("context_modifiers"),
            )

            # Step 3: Populate with phone sessions
            sessions = self.session_populator.populate(
                schedule=schedule,
                parameters=parameters,
                dimensions=dimensions,
            )
            # TODO --- LLM Call
            # Step 4: Create final DailySchedule with sessions
            final_schedule = DailySchedule(
                date=date_str,
                day_type=day_type,
                segments=schedule.segments,
                sessions=sessions,
            )

            return final_schedule

        except Exception as e:
            print(f"Error generating schedule: {e}")
            import traceback

            traceback.print_exc()
            return None

    def generate_full_persona(
        self,
        survey: ComprehensiveSurveyInput,
        include_weekday: bool = True,
        include_weekend: bool = True,
    ) -> dict:
        """
        Generate a complete persona with all schedules.

        Args:
            survey: The comprehensive survey input
            include_weekday: Whether to generate weekday schedule
            include_weekend: Whether to generate weekend schedule

        Returns:
            Dictionary containing all persona data
        """
        # Generate core persona
        dimensions, parameters = self.generate_persona(survey)

        today = date.today()

        result = {
            "survey": survey,
            "dimensions": dimensions,
            "parameters": parameters,
            "weekday_schedule": None,
            "weekend_schedule": None,
        }

        # Generate schedules
        if include_weekday:
            result["weekday_schedule"] = self.generate_schedule(
                dimensions=dimensions,
                parameters=parameters,
                survey=survey,
                day_type="weekday",
                target_date=today,
            )

        if include_weekend:
            result["weekend_schedule"] = self.generate_schedule(
                dimensions=dimensions,
                parameters=parameters,
                survey=survey,
                day_type="weekend",
                target_date=today,
            )

        return result
