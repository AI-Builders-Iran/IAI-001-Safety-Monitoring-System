from LLM.models import RuleEngineData
from LLM.llm_client import LLMModel, download_model
from LLM.models import RuleEngineData
from LLM.prompt import HSEPromptGenerator

# download_model()

data = RuleEngineData(
    time="2026-07-28T10:35:42Z",
    events=[
        {
            "event_type": "ForkliftNearPerson",
            "zone": "A-03",
            "timestamp": "00:01:42",
            "severity": "High"
        },
        {
            "event_type": "BlockedAisle",
            "zone": "B-01",
            "timestamp": "00:03:15",
            "severity": "Medium",
            "duration": 180
        }
    ],
    statistics={
        "persons": 4,
        "forklifts": 2,
        "pallets": 18,
        "alerts": 3
    }
)

model = LLMModel()

prompt_generator = HSEPromptGenerator()

print(
    model.generate(
        prompt=prompt_generator.executive_summary(data=data)
    )
)