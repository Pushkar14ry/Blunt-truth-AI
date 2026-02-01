from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import re
import traceback



app = FastAPI(
    title="BluntTruth AI (Offline)",
    description="An offline AI that brutally challenges opinions with logic",
    version="3.1"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow all origins (safe for demo)
    allow_credentials=True,
    allow_methods=["*"],          # allow POST, OPTIONS, etc.
    allow_headers=["*"],
)

# =========================
# MODELS (OFFLINE)
# =========================

opinion_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

text_generator = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=180,
    do_sample=True,
    temperature=0.6
)

# =========================
# REQUEST MODEL
# =========================

class OpinionInput(BaseModel):
    text: str

# =========================
# FALLACY DETECTION
# =========================

FALLACY_PATTERNS = {
    "Overgeneralization": r"\b(always|never|everyone|no one)\b",
    "Emotional Appeal": r"\b(obviously|clearly|any idiot|stupid)\b",
    "False Dilemma": r"\b(either|only two options)\b",
    "Unsupported Claim": r"\b(i think|i feel|probably)\b"
}

def detect_fallacies(text: str):
    found = []
    for name, pattern in FALLACY_PATTERNS.items():
        if re.search(pattern, text.lower()):
            found.append(name)
    return found if found else ["None detected"]

# =========================
# OPINION CHECK
# =========================

def is_opinion(text: str):
    result = opinion_classifier(
        text,
        candidate_labels=["opinion", "fact"]
    )
    return result["labels"][0].lower() == "opinion"

# =========================
# GENERATION HELPERS
# =========================

def generate_after_marker(prompt: str, marker: str):
    output = text_generator(prompt)[0]["generated_text"]
    if marker in output:
        return output.split(marker, 1)[1].strip()
    return output.strip()

def clean_lines(text: str, max_lines=3):
    lines = text.split("\n")
    lines = [l.strip("- ").strip() for l in lines if l.strip()]
    return "\n".join(lines[:max_lines])

# =========================
# ARGUMENT GENERATION
# =========================

def blunt_counter_argument(text: str):
    marker = "CRITIQUE:"
    prompt = (
        f"STATEMENT:\n{text}\n\n"
        f"{marker}\n"
        "- "
    )
    raw = generate_after_marker(prompt, marker)
    return clean_lines(raw)


def steelman_argument(text: str):
    marker = "DEFENSE:"
    prompt = (
        f"STATEMENT:\n{text}\n\n"
        f"{marker}\n"
        "- "
    )
    raw = generate_after_marker(prompt, marker)
    return clean_lines(raw)


def truth_verdict(text: str):
    marker = "VERDICT:"
    prompt = (
        f"STATEMENT:\n{text}\n\n"
        f"{marker}\n"
    )
    raw = generate_after_marker(prompt, marker)
    return raw.split("\n")[0].strip()

# =========================
# CONFIDENCE SCORE
# =========================

def confidence_score(text: str):
    marker = "SCORE:"
    prompt = (
        f"STATEMENT:\n{text}\n\n"
        f"{marker}\n"
    )
    output = generate_after_marker(prompt, marker)
    match = re.search(r"\b([0-9]{1,3})\b", output)
    score = int(match.group()) if match else 50
    return max(0, min(score, 100))

# =========================
# API ENDPOINTS
# =========================

@app.post("/analyze")
def analyze(data: OpinionInput):
    try:
        text = data.text.strip()

        if not is_opinion(text):
            return {
                "error": "This is a factual statement. Facts are not debatable."
            }

        return {
            "statement": text,
            "confidence_score": confidence_score(text),
            "detected_fallacies": detect_fallacies(text),
            "counter_argument": blunt_counter_argument(text),
            "steelman_argument": steelman_argument(text),
            "final_verdict": truth_verdict(text)
        }

    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal error")


@app.post("/debate")
def debate(data: OpinionInput):
    text = data.text.strip()
    return {
        "statement": text,
        "debate": [
            {"round": 1, "response": blunt_counter_argument(text)},
            {"round": 2, "response": steelman_argument(text)},
            {"round": 3, "response": truth_verdict(text)}
        ]
    }