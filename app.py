import os
import re
from dotenv import load_dotenv

import gradio as gr
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Environment & API Key
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()

# 2. Vector DB & Embeddings Setup
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Local medical FAQ sample fallback knowledge
sample_faq_path = "medical_faq.txt"
if not os.path.exists(sample_faq_path):
    with open(sample_faq_path, "w", encoding="utf-8") as f:
        f.write(
            "Q: What are the primary symptoms of Covid-19?\n"
            "A: Primary symptoms include fever or chills, dry cough, shortness of breath, fatigue, and body aches.\n\n"
            "Q: What steps should be taken for seasonal allergy prevention?\n"
            "A: Minimize outdoor exposure during high-pollen morning hours, keep windows closed, and use HEPA air purifiers.\n\n"
            "Q: What are common symptoms of influenza (flu)?\n"
            "A: High fever, chills, persistent dry cough, sore throat, runny nose, and severe muscle fatigue.\n\n"
            "Q: How can I treat a minor first-degree burn at home?\n"
            "A: Cool under running tap water for 10 to 15 minutes. Never apply ice directly. Apply aloe vera gel or petroleum jelly.\n"
        )

loader = TextLoader(sample_faq_path, encoding="utf-8")
raw_docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)
doc_chunks = splitter.split_documents(raw_docs)
vector_db = Chroma.from_documents(doc_chunks, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# 3. Model Fallback Hierarchy
MODELS_TO_TRY = [
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash"
]

def extract_text_safely(res):
    """Safely extracts text whether res is a string, dict, or AIMessage block list."""
    if hasattr(res, "text") and res.text:
        return res.text
    if hasattr(res, "content"):
        content = res.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif hasattr(item, "text"):
                    parts.append(item.text)
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts).strip()
    return str(res).strip()

def invoke_gemini_with_fallback(prompt_text):
    """Tries each model in order until one succeeds."""
    last_error = None
    for model_name in MODELS_TO_TRY:
        try:
            print(f"[*] Trying model: {model_name}...")
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.2,
                max_output_tokens=3000,
                google_api_key=google_api_key if google_api_key else "dummy_key"
            )
            res = llm.invoke(prompt_text)
            text_out = extract_text_safely(res)
            if text_out and text_out != "[]":
                print(f"[+] Successfully generated with {model_name}")
                return text_out
        except Exception as e:
            print(f"[-] Model '{model_name}' failed: {e}")
            last_error = e
            continue
    raise RuntimeError(f"All models failed. Last error: {last_error}")

# 4. Red-Flag Emergency Screening with Negation Handling
RED_FLAGS = [
    "chest pain", "crushing chest pain", "heart attack", "can't breathe",
    "shortness of breath", "difficulty breathing", "slurred speech",
    "facial droop", "unconscious", "severe bleeding", "stiff neck",
    "paralysis", "anaphylaxis"
]

def check_red_flags(query_text):
    """Checks for red flags while ignoring negations (e.g., 'no difficulty breathing')."""
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
    "Please consult a registered medical practitioner.*"
)

# 5. Core Clinical Triage Generator (Uses dictionary messages format for Gradio 5/6)
def triage_consultation(user_message, chat_history):
    if chat_history is None:
        chat_history = []
    
    if not user_message or not user_message.strip():
        return "", chat_history

    # Step A: Emergency Screening
    if check_red_flags(user_message):
        emergency_response = (
            "🚨 **EMERGENCY FAST-TRACK ALERT: LEVEL 1 (CRITICAL)**\n\n"
            "Your description matches potential high-acuity red flag symptoms.\n\n"
            "**ACTIONS TO TAKE IMMEDIATELY:**\n"
            "1. **CALL EMERGENCY SERVICES (112 / 108 / 911) IMMEDIATELY.**\n"
            "2. Do not attempt self-treatment or drive yourself to the hospital.\n"
            "3. Have someone stay with you while awaiting emergency personnel."
            + DISCLAIMER_TEXT
        )
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": emergency_response})
        return "", chat_history

    # Step B: Clinical System Prompt
    system_prompt = (
        "You are MediAssist, an evidence-based clinical triage and healthcare AI assistant.\n"
        "Provide a comprehensive, empathetic, and clear response following this exact structure:\n\n"
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

    full_query = f"{system_prompt}\n\nPatient Query: {user_message}"

    try:
        response_text = invoke_gemini_with_fallback(full_query)
    except Exception as err:
        print(f"Fallback to vector DB due to: {err}")
        docs = retriever.invoke(user_message)
        context = "\n".join([d.page_content for d in docs])
        response_text = (
            f"🏥 **MediAssist Clinical Guidance (RAG Fallback)**\n\n"
            f"**Retrieved Clinical Reference:**\n{context}\n\n"
            f"*(Notice: Direct AI service temporary capacity notice - {err})*\n"
            f"1. **Triage Urgency Level:** Level 3 (Supportive Care)\n"
            f"2. **Practical Care:** Rest, hydration, and salt-water gargles for throat irritation.\n"
            f"3. **Medication Note:** Paracetamol 500-650 mg every 4-6 hours as needed for adults (max 3000 mg/day)."
        )

    if "Standard Clinical Disclaimer" not in response_text:
        response_text += DISCLAIMER_TEXT

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": response_text})
    return "", chat_history

# 6. UI Dark Styling
custom_css = """
body, .gradio-container {
    background-color: #0b1120 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
"""

# 7. Gradio Blocks Interface
with gr.Blocks(title="MediAssist Clinical Triage") as demo:
    gr.Markdown(
        """
        # 🏥 MediAssist Clinical Triage & Intelligence
        ### Evidence-based clinical assessment, red-flag screening, and OTC guidance powered by Gemini.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=9):
            chatbot = gr.Chatbot(
                label="Clinical Consultation Session",
                height=520
            )
        
    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="Describe symptoms, duration, and patient age (e.g., '21-year-old with throat pain and mild fever')...",
            label="Patient Query",
            scale=8,
            lines=2
        )
        submit_btn = gr.Button("Consult", variant="primary", scale=2)

    with gr.Row():
        clear_btn = gr.Button("Clear Consultation", variant="secondary")

    # Wire interactive actions
    submit_btn.click(triage_consultation, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
    msg_input.submit(triage_consultation, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
    clear_btn.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, css=custom_css)