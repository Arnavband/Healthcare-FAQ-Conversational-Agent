import os
import re
import json
import urllib.request
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="MediAssist Clinical Triage AI")

# Enable CORS for cross-device support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

# Red flag symptoms screening
RED_FLAGS = [
    "chest pain", "crushing chest pain", "heart attack", "can't breathe",
    "shortness of breath", "difficulty breathing", "slurred speech",
    "facial droop", "unconscious", "severe bleeding", "stiff neck",
    "paralysis", "anaphylaxis"
]

def check_red_flags(query_text: str) -> bool:
    """Checks for emergency red flags while respecting negation phrases."""
    text_lower = query_text.lower()
    for flag in RED_FLAGS:
        if flag in text_lower:
            negation_pattern = rf"(no|not|without|denies|zero|never)\s+(?:any\s+)?{re.escape(flag)}"
            if re.search(negation_pattern, text_lower):
                continue
            if "breathing" in flag and re.search(r"breathing\s+(?:is\s+)?(?:completely\s+)?(normal|fine|good|okay)", text_lower):
                continue
            return True
    return False

DISCLAIMER_TEXT = (
    "\n\n---\n⚠️ **Standard Clinical Disclaimer:**\n"
    "*This guidance is strictly educational and does not constitute a formal diagnosis, prescription, or emergency service. "
    "Please consult a registered medical practitioner immediately for formal assessment.*"
)

SYSTEM_PROMPT = (
    "You are MediAssist, an evidence-based clinical triage and healthcare AI assistant.\n"
    "Analyze the patient symptoms and provide a comprehensive, empathetic, and clear response following this exact structure:\n\n"
    "### 1. 🩺 Triage Assessment & Urgency Level\n"
    "- State the triage level: Level 1 (Emergency), Level 2 (Urgent Doctor Visit), or Level 3 (Supportive Care / Non-Urgent).\n"
    "- Acknowledge the patient's stated age or demographics.\n\n"
    "### 2. 🌿 Home Precautions & Supportive Care\n"
    "- Practical non-pharmacological care (e.g., hydration, gargling, rest, positioning).\n\n"
    "### 3. 💊 Over-The-Counter (OTC) Guidance & Dosages\n"
    "- Detail specific standard OTC medications, exact adult or pediatric dosages, dosing intervals, and maximum daily limits.\n"
    "- Mention contraindications (e.g., avoid NSAIDs with ulcers or kidney disease; Paracetamol precautions for liver).\n\n"
    "### 4. 🚨 Red-Flag Symptoms to Monitor\n"
    "- Specific warning signs requiring immediate emergency or ER care.\n"
)

MODELS_TO_TRY = [
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]

def call_gemini_api(prompt_text: str, api_key: str) -> str:
    """Calls Google Gemini using the google.genai Client with fallback."""
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not configured.")

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        for model in MODELS_TO_TRY:
            try:
                res = client.models.generate_content(model=model, contents=prompt_text)
                if res and res.text:
                    return res.text
            except Exception as e:
                print(f"[-] Model '{model}' error: {e}")
                continue
    except Exception as e:
        print(f"[!] SDK error: {e}")

    # REST fallback
    for model in MODELS_TO_TRY:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            payload = json.dumps({
                "contents": [{"parts": [{"text": prompt_text}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 2048
                }
            }).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=12) as response:
                result = json.loads(response.read().decode("utf-8"))
                candidates = result.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"]
        except Exception:
            continue
    raise RuntimeError("All Gemini models failed.")

# Local clinical FAQ fallback
FAQ_KNOWLEDGE = [
    {
        "keywords": ["covid", "corona", "symptoms"],
        "topic": "Covid-19 Symptoms",
        "guidance": "Primary Covid-19 symptoms include fever/chills, dry cough, shortness of breath, fatigue, and body aches. Isolate and test if symptoms persist."
    },
    {
        "keywords": ["flu", "influenza", "fever", "cough"],
        "topic": "Influenza / Viral Infection",
        "guidance": "Common flu symptoms include high fever, chills, persistent dry cough, sore throat, and severe fatigue. Prioritize hydration, bed rest, and Paracetamol (500-650mg every 4-6h for adults)."
    },
    {
        "keywords": ["allergy", "pollen", "allergic"],
        "topic": "Seasonal Allergy Care",
        "guidance": "Minimize morning outdoor exposure during peak pollen times, keep windows closed, use saline nasal sprays, and consider non-drowsy oral antihistamines (e.g., Cetirizine 10mg)."
    },
    {
        "keywords": ["burn", "scald", "heat"],
        "topic": "Minor First-Degree Burn",
        "guidance": "Immediately cool under cool running tap water for 10-15 minutes. Never apply ice directly. Apply pure aloe vera gel or petroleum jelly and keep clean."
    }
]

def get_faq_fallback(query: str) -> str:
    query_l = query.lower()
    for item in FAQ_KNOWLEDGE:
        if any(kw in query_l for kw in item["keywords"]):
            return (
                f"### 🩺 MediAssist Clinical Guidance (Knowledge Fallback)\n\n"
                f"**Clinical Topic:** {item['topic']}\n"
                f"- **Guidance:** {item['guidance']}\n\n"
                f"**Standard OTC Note:** Paracetamol 500-650 mg every 4-6 hours as needed for adults (maximum 3000 mg/day). Avoid NSAIDs if history of gastritis or renal impairment."
            )
    return (
        "### 🩺 MediAssist Clinical Assessment\n\n"
        "- **Triage Urgency Level:** Level 3 (Supportive Home Care)\n"
        "- **General Supportive Measures:** Maintain strict oral hydration (water, electrolyte solutions), monitor body temperature twice daily, and rest.\n"
        "- **OTC Guidance:** For fever or discomfort, Paracetamol 500-650 mg every 4-6 hours for adults (max 3000 mg/day). Consult a pediatrician for weight-based child dosing."
    )

class ChatRequest(BaseModel):
    message: str

@app.get("/api/health")
def health_check():
    key_configured = bool(GOOGLE_API_KEY)
    return {
        "status": "healthy",
        "service": "MediAssist Clinical AI",
        "google_api_key_configured": key_configured
    }

@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    user_query = payload.message.strip()
    if not user_query:
        return {"reply": "Please describe your symptoms, duration, and patient age."}

    # Step 1: Emergency Red-Flag Screening
    if check_red_flags(user_query):
        emergency_reply = (
            "🚨 **EMERGENCY FAST-TRACK ALERT: LEVEL 1 (CRITICAL)**\n\n"
            "Your description matches potential high-acuity red flag medical symptoms.\n\n"
            "**ACTIONS TO TAKE IMMEDIATELY:**\n"
            "1. **CALL EMERGENCY SERVICES (112 / 108 / 911) IMMEDIATELY.**\n"
            "2. Do not attempt to drive yourself to the hospital.\n"
            "3. Have someone stay with you while awaiting emergency personnel."
            + DISCLAIMER_TEXT
        )
        return {
            "reply": emergency_reply,
            "triage_level": 1,
            "is_emergency": True
        }

    # Step 2: Clinical Gemini Analysis
    api_key = os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY).strip()
    prompt = f"{SYSTEM_PROMPT}\n\nPatient Query: {user_query}"

    try:
        if not api_key:
            raise ValueError("No GOOGLE_API_KEY configured in environment.")
        reply_text = call_gemini_api(prompt, api_key)
    except Exception as err:
        print(f"[!] AI API fallback triggered: {err}")
        reply_text = get_faq_fallback(user_query) + f"\n\n*(Notice: Clinical knowledge fallback active - {err})*"

    if "Standard Clinical Disclaimer" not in reply_text:
        reply_text += DISCLAIMER_TEXT

    return {
        "reply": reply_text,
        "triage_level": 3 if "Level 3" in reply_text else (2 if "Level 2" in reply_text else 1),
        "is_emergency": False
    }

# Serve the static frontend on root
@app.get("/")
def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return HTMLResponse("<h1>MediAssist Clinical AI is running</h1>")