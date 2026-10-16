import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from rag_answer_norm import load_all, GEMINI_MODEL

load_dotenv()
llm = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

records = load_all()
departments = sorted({r["department"] for r in records})
services = sorted({r["service"] for r in records})


def translate_list(items, label):
    prompt = (
        f"Translate each of the following {label} names from Arabic to English. "
        f"Return ONLY a JSON array of strings, same order, same length, no extra text.\n\n"
        + "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))
    )
    response = llm.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


dept_en = translate_list(departments, "government department")
service_en = translate_list(services, "government service")

translations = {
    "departments": dict(zip(departments, dept_en)),
    "services": dict(zip(services, service_en)),
}

Path("translations.json").write_text(
    json.dumps(translations, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"Translated {len(departments)} departments and {len(services)} services.")
print("Saved to translations.json")