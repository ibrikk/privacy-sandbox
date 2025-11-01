import os
import configparser
import openai
import warnings

from langchain.chains.llm import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

import json
from datetime import datetime
from random import randint
from collections import defaultdict
import re
import base64

from openai import OpenAI
import requests


warnings.filterwarnings("ignore", category=DeprecationWarning)

# 忽略 FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)


# Load config.ini from parent directory
current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)
config_path = os.path.join(current_dir_path, "../config.ini")

config = configparser.ConfigParser()
config.read(config_path)

api_key = config["openai"]["OPENAI_API_KEY"]
openai.api_key = api_key


# load prompts and few shot examples
with open(os.path.join(current_dir_path, "./prompts.json")) as prompts_file:
    prompts = json.load(prompts_file)

with open(os.path.join(current_dir_path, "./examples.json")) as examples_file:
    fewshot_examples = json.load(examples_file)


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
    sensors: list[str] = Field(
        description="List of spoofed sensors used for this persona. Options include: accelerometer, step_counter, step_detector, gyroscope, magnetic_field, linear_acceleration, light, cell_station"
    )
    traits: dict = Field(
        description="""
    Nested trait attributes for the persona, including lifestyle and sensor behavior.

    Example:
    {
          "fitness_level": "high",
          "exercise_frequency_per_week": 2,
          "usual_exercise_time": ["07:30", "11:30", "19:15"],
          "exercise_type": ["hiking"],
          "commute_pattern": "drives to gym",
          "sleep_schedule": {
            "bed_time": "21:00",
            "wake_time": "05:00"
          },
          "diet_focus": "plant-based",
          "screen_time_hours": 10,
          "shopping_behavior": "protein snacks, gym wear",
          "sensors_behavior": {
            "accelerometer": {
              "mean": 0,
              "drift": 4.8
            },
            "gyroscope": {
              "mean": 0,
              "drift": 1.6
            },
            "light": {
              "mean": 10686,
              "drift": 1999
            },
            "step_counter": {
              "start": 5084,
              "rate": 2.4
            },
            "step_detector": {
              "mean": 0,
              "drift": 0.47
            },
            "linear_acceleration": {
              "mean": 0,
              "drift": 0.45
            },
            "cell_station": {
              "mean": 4,
              "drift": 0
            },
            "magnetic_field": {
              "mean": 54,
              "drift": 12
            }
          }
        }
    """
    )


class EventSet(BaseModel):
    event: list[str] = Field(description="A list of calendar event content")


class ScheduleEntry(BaseModel):
    start_time: str = Field(
        description="The start time of a schedule event in the format of YYYY-MM-DD hh:mm:ss"
    )
    end_time: str = Field(
        description="The end time of a schedule event in the format of YYYY-MM-DD hh:mm:ss"
    )
    event: str = Field(description="A schedule event in the calendar")
    address: str = Field(
        description="The place where the event happen, must including four parts: street, city, state, and zipcode"
    )
    latitude: float = Field(description="The corresponding latitude of the address")
    longitude: float = Field(description="The corresponding longitude of the address")


class Schedule(BaseModel):
    schedule: list[ScheduleEntry]


class BrowsingHistoryEntry(BaseModel):
    time: str = Field(
        description="The time for browser history entry in the format of YYYY-MM-DD hh:mm:ss"
    )
    title: str = Field(description="The website title of the browser history entry")
    url: str = Field(description="The url of the browsing history entry")


class BrowsingHistory(BaseModel):
    browsing_history: list[BrowsingHistoryEntry]


class Generator:
    def __init__(self):
        self.persona_profile = None
        self.persona_variant = None
        self.persona_attributes = None
        self._event_sets = None
        self.schedule = None
        self.browsing_history = []

    def get_persona_profile(self, guidance) -> str:
        """
        Generate the persona description from LLM

        Parameters:
            guidance(str): The text briefly describes a persona.

        Return
            str: persona description
        """
        examples = fewshot_examples["profile"]

        example_prompt = PromptTemplate(
            input_variables=["persona"], template=prompts["profile"]["prompt"]
        )

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prompts["profile"]["prefix"],
            suffix=prompts["profile"]["suffix"],
            input_variables=["guidance", "template"],
            example_separator="\n\n",
        )

        chain = LLMChain(
            llm=ChatOpenAI(
                model_name="gpt-4o-mini", temperature=0.9, openai_api_key=api_key
            ),
            prompt=few_shot_prompt,
        )
        persona_profile = chain.invoke(
            input={"guidance": guidance, "template": prompts["profile"]["template"]}
        )
        self.persona_profile = persona_profile["text"]

        return self.persona_profile

    def _gen_persona_variant(self, profile: str, version: str) -> str:
        """
        Generate the persona variation based on a given profile and the version

        Parameters:
            profile(str): The base persona profile for variation.
            attr(str): The version str for getting variation.

        Return
            str: variant persona profile
        """
        examples = fewshot_examples["profile_variant"]

        example_prompt = PromptTemplate(
            input_variables=["persona", "version", "variant"],
            template=prompts["profile_variant"]["prompt"],
        )

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prompts["profile_variant"]["prefix"],
            suffix=prompts["profile_variant"]["suffix"],
            input_variables=["template", "persona", "version"],
            example_separator="\n\n",
        )

        chain = LLMChain(
            llm=ChatOpenAI(
                model_name="gpt-4o-mini", temperature=0.9, openai_api_key=api_key
            ),
            prompt=few_shot_prompt,
        )
        persona_variant = chain.invoke(
            input={
                "persona": profile,
                "version": version,
                "template": prompts["profile"]["template"],
            }
        )

        return persona_variant["text"]

    def get_persona_variant(
        self, profile: str, attr: str, attr_type: str, whole_attr
    ) -> str:
        """
        Generate the persona variation based on a given profile and a privacy attributes

        Parameters:
            profile(str): The base persona profile for variation.
            attr(str): The privacy attributes for getting variation.

        Return
            str: variant persona description
        """

        attr_dict = {
            "Age": ["young", "mid-aged", "old"],
            "Gender": ["male", "non-binary", "female"],
            "Location (urbanization)": ["urban", "suburb", "countryside"],
            "Income": ["low", "medium", "high"],
            "Educational level": ["low", "medium", "high"],
        }

        res = []
        # versions = [value for value in attr_dict[attr] if value != attr_type]
        for v in attr_dict[attr]:
            if v == attr_type:
                res.append({"value": v, "profile": profile, "attr": whole_attr})
            else:
                var_profile = self._gen_persona_variant(profile, v)
                res.append(
                    {
                        "value": v,
                        "profile": var_profile,
                        "attr": self.get_attributes(var_profile),
                    }
                )

        return res

    def _get_persona_short_profile(self, attribute: dict) -> str:
        #  if attribute "error" exists inside the dictionary, then get "raw" values
        short_profile = ""
        if "error" in attribute:
            raw_string = attribute["raw"].replace("\\'", "'")
            raw = json.loads(raw_string)
            if isinstance(raw, str):
                raw = json.loads(raw)  # parse string into dictionary
                short_profile = f"A {raw['age']}-year-old {raw['race']} {raw['gender']} {raw['job']}"
        else:
            short_profile = f"A {attribute['age']}-year-old {attribute['race']} {attribute['gender']} {attribute['job']}"

        return short_profile

    def _get_profile_img_prompt(self, short_profile: str) -> str:
        prompt_template = PromptTemplate(
            input_variables=["short_profile"], template=prompts["profile_img"]["prompt"]
        )

        chain = LLMChain(
            llm=ChatOpenAI(
                model_name="gpt-4o-mini", temperature=0.9, openai_api_key=api_key
            ),
            prompt=prompt_template,
        )
        profile_img_prompt = chain.run(short_profile=short_profile)

        return profile_img_prompt

    def get_profile_img(self, attribute):
        """
        Generate the persona's profile image
        """
        short_profile = self._get_persona_short_profile(attribute)
        profile_img_prompt = self._get_profile_img_prompt(short_profile)
        print(profile_img_prompt)

        client = OpenAI(api_key=api_key)

        response = client.images.generate(
            model="dall-e-2",
            prompt=profile_img_prompt,
            size="256x256",
            n=1,
        )

        self.profile_img_url = response.data[0].url
        self.profile_img = base64.b64encode(requests.get(self.profile_img_url).content)

        return {"base64_json": self.profile_img, "url": self.profile_img_url}

    def get_attributes(self, profile: str) -> dict:
        """
        Generate the persona's privacy attributes from the profile

        Input
            profile(str): the profile description of the persona to support generate its weekly event sets.

        Return
            attributes(dict): a dict of the parsed attributes
        """
        curated_examples = []
        for i, v in enumerate(fewshot_examples["attributes"]):
            example = v["attributes_example"]
            if "sensors" not in example:
                example["sensors"] = [
                    "accelerometer",
                    "step_counter",
                    "step_detector",
                    "gyroscope",
                    "magnetic_field",
                    "linear_acceleration",
                    "light",
                    "cell_station",
                ]
            curated_examples.append(
                {
                    "persona": v["persona"],
                    "attributes_example": PrivacyAttributes.model_validate(
                        v["attributes_example"]
                    )
                    .model_dump_json()
                    .replace("{", "{{")
                    .replace("}", "}}"),
                    "sensors": [
                        "accelerometer",
                        "step_counter",
                        "step_detector",
                        "gyroscope",
                        "magnetic_field",
                        "linear_acceleration",
                        "light",
                        "cell_station",
                    ],
                }
            )
        examples = curated_examples

        example_prompt = PromptTemplate(
            input_variables=["persona", "attributes_example"],
            template=prompts["attributes"]["prompt"],
        )

        attribute_parser = PydanticOutputParser(pydantic_object=PrivacyAttributes)

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="",
            suffix=prompts["attributes"]["suffix"],
            input_variables=["persona"],
            partial_variables={
                "format_instructions": attribute_parser.get_format_instructions()
            },
        )

        chain = LLMChain(
            llm=ChatOpenAI(
                model_name="gpt-4o-mini", temperature=0.9, openai_api_key=api_key
            ),
            prompt=few_shot_prompt,
        )
        persona_attributes = chain.invoke(input={"persona": profile})
        self.persona_attributes = persona_attributes["text"]

        pattern = r"```json\n(.+?)\n```"
        match = re.search(pattern, self.persona_attributes, re.DOTALL)

        if match:
            try:
                res = json.loads(match.group(1))
            except json.JSONDecodeError as e:
                print("⚠️ JSON parsing failed from code block:", e)
                res = {"error": "JSON parsing failed", "raw": match.group(1)}
        else:
            try:
                res = json.loads(self.persona_attributes)
            except json.JSONDecodeError as e:
                print("⚠️ JSON parsing failed from raw text:", e)
                res = {"error": "JSON parsing failed", "raw": self.persona_attributes}

        return res

    def _gen_events(self, profile: str) -> list:
        """
        Generate the persona's event sets to support generate the schedule.

        Input
            profile(str): the profile description of the persona to support generate its weekly event sets.

        Return
            list: a event set for the input persona.
        """
        event_parser = PydanticOutputParser(pydantic_object=EventSet)

        human_prompt = HumanMessagePromptTemplate.from_template(
            "{request}\n{profile}\n{format_instructions}"
        )
        chat_prompt = ChatPromptTemplate.from_messages([human_prompt])

        request = chat_prompt.format_prompt(
            profile=profile,
            request=prompts["event"]["prompt"],
            format_instructions=event_parser.get_format_instructions(),
        ).to_messages()

        model = ChatOpenAI(
            model_name="gpt-4o-mini", temperature=0.9, openai_api_key=api_key
        )
        results = model(request)
        results_value = event_parser.parse(results.content)
        self._event_sets = results_value

        return self._event_sets

    def get_schedule(
        self, persona: str, start_date: str = "2024-05-01", end_date: str = "2024-05-01"
    ) -> list:
        """
        Generate the persona's schedule.

        Input
            person(str): the profile description of the persona,
            start_date(str): the start date of the schedule in the format of YYYY-MM-DD
            end_date(str): the end date of the schedule in the format of YYYY-MM-DD

        Return
            list: the persona's schedule
        """

        # We need to add .replace("{", "{{").replace("}", "}}") after serialising as JSON so that the curly brackets in the JSON won’t be mistaken for prompt variables by LangChain
        curated_examples = []
        for i, v in enumerate(fewshot_examples["schedule"]):
            curated_examples.append(
                {
                    "persona": v["persona"],
                    "event_example": v["event_example"],
                    "schedule_example": [],
                }
            )
            for entry in v["schedule_example"]:
                curated_examples[i]["schedule_example"].append(
                    ScheduleEntry.model_validate(entry)
                    .model_dump_json()
                    .replace("{", "{{")
                    .replace("}", "}}")
                )

        examples = curated_examples

        schedule_parser = PydanticOutputParser(pydantic_object=Schedule)

        example_prompt = PromptTemplate(
            input_variables=["persona", "event_example", "schedule_example"],
            template=prompts["schedule"]["prompt"],
        )

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prompts["schedule"]["prefix"],
            suffix=prompts["schedule"]["suffix"],
            input_variables=["persona", "start_date", "end_date", "event_example"],
            partial_variables={
                "format_instructions": schedule_parser.get_format_instructions()
            },
        )

        chain = LLMChain(
            llm=ChatOpenAI(
                model_name="gpt-4o-mini", temperature=0.5, openai_api_key=api_key
            ),
            prompt=few_shot_prompt,
        )
        generated_schedule = chain.invoke(
            input={
                "persona": persona,
                "event_example": self._gen_events(persona),
                "start_date": start_date,
                "end_date": end_date,
            }
        )

        self.schedule = generated_schedule["text"]

        # self.schedule = eval(generated_schedule['text'])
        # self.schedule = self.schedule["schedule"]

        pattern = r"```json\n(.+?)\n```"
        match = re.search(pattern, generated_schedule["text"], re.DOTALL)

        if match:
            res = eval(match.group(1))
        else:
            res = eval(generated_schedule["text"])

        return res["schedule"]

    def get_device_browser(self, persona: str) -> dict:
        pass

    def _get_one_day_browsing_history(
        self, persona: str, date: str, schedule: list
    ) -> list:
        """
        Generate the persona's browsing history for one day.

        Input
            person(str): the profile description of the persona,
            start_date(str): the start date of the schedule in the format of YYYY-MM-DD
            end_date(str): the end date of the schedule in the format of YYYY-MM-DD
            schedule(list): the persona's one-day schedule to guide the browsing history generation

        Return
            browsing history(list): the persona's schedule
        """
        curated_examples = []
        for i, v in enumerate(fewshot_examples["browsing_history"]):
            curated_examples.append(
                {
                    "persona": v["persona"],
                    "schedule": [],
                    "browsing_history_example": [],
                }
            )
            for s in v["schedule"]:
                curated_examples[i]["schedule"].append(
                    ScheduleEntry.model_validate(s)
                    .model_dump_json()
                    .replace("{", "{{")
                    .replace("}", "}}")
                )

            for bh in v["browsing_history_example"]:
                curated_examples[i]["browsing_history_example"].append(
                    BrowsingHistoryEntry.model_validate(bh)
                    .model_dump_json()
                    .replace("{", "{{")
                    .replace("}", "}}")
                )

        examples = curated_examples

        browsing_history_parser = PydanticOutputParser(pydantic_object=BrowsingHistory)

        example_prompt = PromptTemplate(
            input_variables=["persona", "schedule", "browsing_history_example"],
            template=prompts["browsing_history"]["prompt"],
        )

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prompts["browsing_history"]["prefix"],
            suffix=prompts["browsing_history"]["suffix"],
            input_variables=["persona", "schedule", "date", "num"],
            partial_variables={
                "format_instructions": browsing_history_parser.get_format_instructions()
            },
        )

        chain = LLMChain(
            llm=ChatOpenAI(
                model_name="gpt-4o-mini", temperature=0.5, openai_api_key=api_key
            ),
            prompt=few_shot_prompt,
        )
        generated_browsing_hitory = chain.invoke(
            input={"persona": persona, "num": "10", "date": date, "schedule": schedule}
        )

        pattern = r"```json\n(.+?)\n```"
        match = re.search(pattern, generated_browsing_hitory["text"], re.DOTALL)

        if match:
            res = eval(match.group(1))
        else:
            res = eval(generated_browsing_hitory["text"])

        return res["browsing_history"]

    def get_browsing_history(
        self,
        persona: str,
        schedule: list,
        start_date: str = "2024-05-01",
        end_date: str = "2024-05-01",
    ) -> list:
        """
        Generate the persona's browsing history.

        Input
            person(str): the profile description of the persona,
            schedule(list): the persona's schedule to guide the browsing history generation,
            start_date(str): the start date of the schedule in the format of YYYY-MM-DD,
            end_date(str): the end date of the schedule in the format of YYYY-MM-DD

        Return
            browsing history(list): the persona's schedule
        """
        events_by_day = defaultdict(list)
        for event in schedule:
            start_date = datetime.strptime(
                event["start_time"], "%Y-%m-%d %H:%M:%S"
            ).date()
            events_by_day[str(start_date)].append(event)
        events_by_day = dict(events_by_day)

        for date in events_by_day:
            # one_day_history = json.loads(self._get_one_day_browsing_history(persona, date, events_by_day[date]))
            # one_day_history = one_day_history["browsing_history"]
            # self.browsing_history.extend(one_day_history)
            self.browsing_history.extend(
                self._get_one_day_browsing_history(persona, date, events_by_day[date])
            )

        return self.browsing_history

    def get_traits(self, profile: str) -> dict:
        full_prompt = prompts["traits"]["prompt"].replace("{persona}", profile)

        prompt_template = ChatPromptTemplate.from_messages(
            [HumanMessagePromptTemplate.from_template("{message}")]
        )

        chain = LLMChain(
            llm=ChatOpenAI(
                model_name="gpt-4o-mini", temperature=0.7, openai_api_key=api_key
            ),
            prompt=prompt_template,
        )

        traits_response = chain.run({"message": full_prompt})

        try:
            return json.loads(traits_response)
        except json.JSONDecodeError:
            print("⚠️ Could not parse traits, returning raw.")
            return {"raw_traits": traits_response}


if __name__ == "__main__":
    gen = Generator()

    # guidance = "a 25-year old male, the living address is in urban area"
    guidance = "Generate a mobile user profile."
    profile = gen.get_persona_profile(guidance)
    print("Profile: ")
    print(profile)

    attribute = gen.get_attributes(profile)
    print("Attribute: ")
    print(attribute)

    traits = gen.get_traits(profile)
    print("Traits: ")
    print(json.dumps(traits, indent=2))

    image = gen.get_profile_img(attribute)
    print(image["url"])

    with open("spoof_config.json", "w") as f:
        if "sensors_behavior" in traits:
            json.dump(traits["sensors_behavior"], f, indent=2)

    # attribute = {
    #     "first_name": "Michael",
    #     "last_name": "Johnson",
    #     "age": "25",
    #     "gender": "male",
    #     "race": "Black",
    #     "street": "456 Elm St",
    #     "city": "Atlanta",
    #     "state": "GA",
    #     "zip_code": "30312",
    #     "spoken_language": "English",
    #     "education_background": "Bachelor degree in Computer Science",
    #     "education_level": "bachelor degree",
    #     "birthday": "04/15/1998",
    #     "job": "Software Developer",
    #     "income": "75,000",
    #     "income_level": "moderate income",
    #     "marital_status": "single",
    #     "parental_status": "not parents",
    #     "online_behavior": "Enjoys gaming and streaming content on computer; uses gaming headset and mechanical keyboard; frequently uses social media apps like Instagram and TikTok.",
    #     "industry": "technology",
    #     "employer_size": "large employer (250-10,000 employees)",
    #     "homeownership": "renter",
    #     "short_profile": "Michael Johnson, 25, is a single software developer from Atlanta, GA, earning $75,000, enjoys gaming and streaming, and stays connected through social media.",
    #     "age_type": "young adult",
    #     "gender_type": "male",
    #     "location_type": "urban",
    #     "income_type": "medium",
    #     "edu_type": "medium",
    # }

    # Combine everything into a string
    persona_log = f"""=== Persona Profile ===
    Name: {attribute['first_name']} {attribute['last_name']}
    Age: {attribute['age']}
    Race: {attribute['race']}
    Gender: {attribute['gender']}
    Job: {attribute['job']}
    Location: {attribute['city']}, {attribute['state']} {attribute['zip_code']}
    Income: ${attribute['income']} ({attribute['income_level']})
    Marital Status: {attribute['marital_status']}
    Parental Status: {attribute['parental_status']}
    Education: {attribute['education_background']}
    Spoken Language: {attribute['spoken_language']}
    Industry: {attribute['industry']}
    Employer Size: {attribute['employer_size']}
    Homeownership: {attribute['homeownership']}

    Short Profile: {attribute.get("short_profile", "")}
    Online Behavior: {attribute['online_behavior']}

    --- Traits ---
    {json.dumps(attribute['traits'], indent=2)}

    Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """

    # Save to a .txt file
    filename = f"persona_{attribute['first_name'].lower()}_{attribute['last_name'].lower()}.txt"
    filepath = os.path.join("personas", filename)

    # Ensure the directory exists
    os.makedirs("personas", exist_ok=True)

    # Write to file
    with open(filepath, "w") as f:
        f.write(persona_log)

    print(f"[📄] Persona saved to {filepath}")