# Add to engine.py


from models import (
    AreaType,
    BehavioralDimensions,
    CommuteDays,
    ComprehensiveSurveyInput,
    WorkPhoneRestriction,
)


class ContextProfile:
    """
    Abstracted context profile for LLM consumption.
    Contains NO personally identifiable information.
    """

    @staticmethod
    def from_survey(
        survey: ComprehensiveSurveyInput, dimensions: BehavioralDimensions
    ) -> dict:
        """
        Create an anonymized context profile from survey data.

        This can be safely sent to an LLM without privacy concerns.
        """

        # Abstract occupation to category
        occupation_type = ContextProfile._categorize_occupation(survey.occupation)

        # Abstract commute pattern
        commute_pattern = {
            CommuteDays.ZERO: "remote_only",
            CommuteDays.ONE_TWO: "hybrid_minimal",
            CommuteDays.THREE_FOUR: "hybrid_regular",
            CommuteDays.FIVE_PLUS: "office_full_time",
        }.get(survey.commute_days, "hybrid_regular")

        # Abstract area
        area_density = {
            AreaType.URBAN: "high_density",
            AreaType.SUBURBAN: "medium_density",
            AreaType.RURAL: "low_density",
        }.get(survey.area_type, "medium_density")

        # Mobility level (no specific counts)
        mobility_level = (
            "high"
            if dimensions.mobility_diversity > 0.6
            else "medium" if dimensions.mobility_diversity > 0.3 else "low"
        )

        # Work flexibility
        work_phone_freedom = {
            WorkPhoneRestriction.USE_FREELY: "unrestricted",
            WorkPhoneRestriction.OCCASIONALLY: "moderate",
            WorkPhoneRestriction.BRIEFLY_WHEN_NECESSARY: "restricted",
            WorkPhoneRestriction.NOT_APPLICABLE: "no_workplace",
        }.get(survey.work_phone_restriction, "moderate")

        # Schedule regularity
        routine_type = (
            "structured"
            if dimensions.routine_stability > 0.6
            else "variable" if dimensions.routine_stability < 0.4 else "semi_structured"
        )

        # Chronotype (affects timing reasonableness)
        chronotype = (
            "morning_person"
            if dimensions.chronotype_score < 0.35
            else (
                "evening_person"
                if dimensions.chronotype_score > 0.65
                else "intermediate"
            )
        )

        return {
            "occupation_type": occupation_type,
            "commute_pattern": commute_pattern,
            "commute_mode": survey.commute_mode.value,
            "area_density": area_density,
            "mobility_level": mobility_level,
            "work_phone_freedom": work_phone_freedom,
            "routine_type": routine_type,
            "chronotype": chronotype,
            "exercise_frequency": survey.physical_activity_days.value,
        }

    @staticmethod
    def _categorize_occupation(occupation: str) -> str:
        """
        Categorize occupation into broad types without specifics.
        Uses simple keyword matching - could be enhanced.
        """
        occupation_lower = occupation.lower()

        # Knowledge worker categories
        if any(
            kw in occupation_lower
            for kw in [
                "engineer",
                "developer",
                "programmer",
                "software",
                "data",
                "analyst",
            ]
        ):
            return "tech_knowledge_worker"
        if any(
            kw in occupation_lower
            for kw in ["manager", "director", "executive", "consultant"]
        ):
            return "management_professional"
        if any(
            kw in occupation_lower
            for kw in ["teacher", "professor", "instructor", "educator"]
        ):
            return "education"
        if any(
            kw in occupation_lower
            for kw in ["doctor", "nurse", "medical", "health", "therapist"]
        ):
            return "healthcare"
        if any(kw in occupation_lower for kw in ["sales", "marketing", "account"]):
            return "sales_marketing"
        if any(
            kw in occupation_lower
            for kw in ["designer", "artist", "creative", "writer"]
        ):
            return "creative_professional"
        if any(
            kw in occupation_lower
            for kw in ["retail", "service", "restaurant", "hospitality"]
        ):
            return "service_industry"
        if any(kw in occupation_lower for kw in ["student"]):
            return "student"
        if any(kw in occupation_lower for kw in ["retired", "homemaker"]):
            return "non_workforce"

        return "general_professional"

    @staticmethod
    def build_llm_prompt(context_profile: dict, day_type: str, date_str: str) -> str:
        """
        Build a prompt for the LLM that requests contextual details
        WITHOUT revealing personal information.
        """

        prompt = f"""Generate realistic daily context variations for a simulated smartphone user.

USER PROFILE (abstracted):
- Occupation type: {context_profile['occupation_type']}
- Work arrangement: {context_profile['commute_pattern']}
- Commute mode: {context_profile['commute_mode']}
- Area: {context_profile['area_density']}
- Mobility level: {context_profile['mobility_level']}
- Work phone policy: {context_profile['work_phone_freedom']}
- Routine: {context_profile['routine_type']}
- Chronotype: {context_profile['chronotype']}
- Exercise frequency: {context_profile['exercise_frequency']}

DAY INFO:
- Type: {day_type}
- Date: {date_str}

TASK:
Based on this profile, generate 2-3 realistic contextual variations for this day. 
For example: "meeting-heavy day", "work from home day", "social evening planned", etc.

Return as JSON:
{{
    "day_variation": "string describing the day type",
    "context_modifiers": {{
        "work_intensity": float 0.5-1.5,  // multiplier for work phone suppression
        "social_evening": bool,  // if true, evening has social context
        "exercise_today": bool,  // if true, include exercise block
        "errands_needed": bool,  // if true, include errand time
        "commutes_today": bool  // for hybrid workers, are they going to office?
    }}
}}

Generate realistic variation, not extreme cases. Most days are typical."""

        return prompt


class LLMContextEnhancer:
    """
    Optional component that uses LLM to add realistic daily variation.
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client  # Optional LLM client

    def get_day_context(
        self,
        survey: ComprehensiveSurveyInput,
        dimensions: BehavioralDimensions,
        day_type: str,
        date_str: str,
    ) -> dict:
        """
        Get contextual modifiers for a specific day.
        Falls back to rule-based logic if no LLM available.
        """

        # Create anonymized profile
        profile = ContextProfile.from_survey(survey, dimensions)

        if self.llm_client is None:
            # Rule-based fallback
            return self._rule_based_context(profile, day_type, date_str)

        # Build prompt with NO personal data
        prompt = ContextProfile.build_llm_prompt(profile, day_type, date_str)

        # Call LLM (implementation depends on your LLM client)
        try:
            response = self.llm_client.generate(prompt)
            return self._parse_llm_response(response)
        except Exception:
            return self._rule_based_context(profile, day_type, date_str)

    def _rule_based_context(self, profile: dict, day_type: str, date_str: str) -> dict:
        """
        Deterministic fallback based on profile.
        """
        import random

        # Determine if this is a commute day
        commutes_today = True
        if profile["commute_pattern"] == "remote_only":
            commutes_today = False
        elif profile["commute_pattern"] == "hybrid_minimal":
            commutes_today = random.random() < 0.25  # ~1-2 days/week
        elif profile["commute_pattern"] == "hybrid_regular":
            commutes_today = random.random() < 0.70  # ~3-4 days/week
        # office_full_time stays True

        if day_type == "weekend":
            commutes_today = False

        return {
            "day_variation": "typical_day",
            "context_modifiers": {
                "work_intensity": 1.0,
                "social_evening": random.random() < 0.15,  # 15% chance
                "exercise_today": random.random()
                < (0.5 if profile["exercise_frequency"] == "three_four" else 0.3),
                "errands_needed": random.random() < 0.20,
                "commutes_today": commutes_today,
            },
        }
