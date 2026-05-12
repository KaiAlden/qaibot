from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.nlp.clarification import ClarificationDecider
from app.nlp.general_intent import GeneralIntentParser
from app.nlp.intent_parser import IntentParser


CASES = [
    {
        "name": "single conditioning type",
        "message": "运动方式方面有什么建议吗",
        "session": {"constitution": "阴虚体质", "area": "华东长江以北"},
    },
    {
        "name": "multi conditioning types",
        "message": "在穴位保健、药浴调理和运动方式方面有什么建议吗",
        "session": {"constitution": "阴虚体质", "area": "华东长江以北"},
    },
    {
        "name": "constitution explain",
        "message": "特禀体质有什么特点呢",
        "session": {},
    },
    {
        "name": "general greeting",
        "message": "你好",
        "session": {},
    },
    {
        "name": "general capability",
        "message": "你能做什么",
        "session": {},
    },
    {
        "name": "external realtime",
        "message": "今天天气怎么样？",
        "session": {},
    },
    {
        "name": "greeting with domain signal",
        "message": "你好，我最近乏力怎么调理",
        "session": {},
    },
]


def main() -> None:
    general_parser = GeneralIntentParser()
    parser = IntentParser()
    clarifier = ClarificationDecider()

    for index, case in enumerate(CASES, start=1):
        general = general_parser.parse(case["message"])
        parsed = parser.parse(case["message"], case["session"])
        clarification = clarifier.decide(parsed, case["session"])
        print(f"\n--- Case {index}: {case['name']} ---")
        print(f"message: {case['message']}")
        print(
            json.dumps(
                {
                    "general_intent": general.intent,
                    "general_answer": general.answer,
                    "parsed": parsed.model_dump(),
                    "clarification": clarification,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
