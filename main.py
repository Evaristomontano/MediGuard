#.\venv\Scripts\activate
#python -m uvicorn main:app 
#PS C:\Users\evari\Documents\MediGuard> git push github master:main
import os
import io
import json
import asyncio
import logging
import base64

# Third-party imports
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import pillow_heif
import httpx
import google.generativeai as genai

# --- 1. CONFIGURATION & LOGGING ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Register HEIF opener to support iPhone photos
pillow_heif.register_heif_opener()

app = FastAPI(
    title="MediGuard API",
    description="Backend for drug interaction detection using Gemini 2.0 Flash.",
    version="2.0.0"
)

# CORS Configuration for local and remote access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 2. UTILITY FUNCTIONS ---

def clean_json_string(text: str) -> str:
    """Removes Markdown code blocks from AI responses to ensure valid JSON parsing."""
    return text.replace("```json", "").replace("```", "").strip()

async def fetch_fda_data(client: httpx.AsyncClient, drug: str):
    """Fetches drug interaction warnings from the official openFDA API."""
    url = f'https://api.fda.gov/drug/label.json?search=drug_interactions:"{drug}"&limit=1'
    try:
        response = await client.get(url, timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            if results:
                interactions = results[0].get('drug_interactions', [""])
                return drug, interactions[0].lower()
    except Exception as e:
        logger.error(f"FDA API Error for {drug}: {str(e)}")
    return drug, None

# --- 3. AI LOGIC (GEMINI INTEGRATION) ---

async def extract_medications_with_ai(image_bytes: bytes) -> list:
    """Uses Gemini Vision to perform OCR and extract generic drug names."""
    prompt = """
    Analyze this medical image.
    1. Extract all specific drug names.
    2. Return ONLY the generic English names separated by commas (e.g., 'Ibuprofen, Paracetamol').
    3. If no drugs are visible, return 'NONE'.
    """
    try:
        logger.info("Attempting medication extraction with Gemini...")
        model = genai.GenerativeModel('gemini-2.0-flash') # Updated to Flash 2.0
        img = Image.open(io.BytesIO(image_bytes))
        response = await model.generate_content_async([prompt, img])
        text = response.text.strip()
        
        if "NONE" in text.upper() or not text:
            return []
        return [med.strip() for med in text.split(",")]
    except Exception as e:
        logger.error(f"Gemini OCR failed: {e}")
        return []

async def generate_safety_report(interactions_data: list, lang: str) -> str:
    """Generates a clinical safety report based on FDA data using Gemini."""
    lang_map = {"en": "English", "es": "Spanish", "de": "German"}
    target_lang = lang_map.get(lang, "English")

    context = ""
    for item in interactions_data:
        context += f"Pair: {item['pair']}\nFDA Info: {item['snippet'][:800]}\n---\n"

    system_prompt = f"""
    You are an expert clinical pharmacist. Analyze the FDA warnings provided.
    Requirements:
    1. Language: Write the 'risk' and 'action' fields strictly in {target_lang}.
    2. Format: Return ONLY a valid JSON object.
    3. Severity: Use 'CRITICAL', 'CAUTION', or 'INFO'.
    
    JSON Structure:
    {{
        "interactions_found": true,
        "details": [
            {{
                "pair": "Drug A + Drug B",
                "severity": "CRITICAL",
                "risk": "Explanation in {target_lang}",
                "action": "Advice in {target_lang}"
            }}
        ],
        "disclaimer": "Medical disclaimer in {target_lang}"
    }}
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = await model.generate_content_async(system_prompt + "\n" + context)
        return clean_json_string(response.text)
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return json.dumps({"interactions_found": False, "disclaimer": "Error generating report."})

# --- 4. ROUTES & ENDPOINTS ---

@app.get("/")
async def serve_index():
    """Serves the main frontend page."""
    return FileResponse('index.html')

@app.get("/settings")
async def serve_settings():
    """Serves the settings/configuration page."""
    return FileResponse('settings.html')

@app.get("/health")
async def health_check():
    """Endpoint for monitoring service status."""
    return {"status": "active", "service": "MediGuard AI"}

@app.post("/analyze-prescription/")
async def analyze_prescription(
    file: UploadFile = File(...), 
    lang: str = "en", 
    x_gemini_key: str = Header(None)
):
    """
    Main Pipeline:
    1. Image Validation -> 2. AI OCR -> 3. openFDA Fetching -> 4. Safety Analysis
    """
    if not x_gemini_key:
        raise HTTPException(status_code=401, detail="Gemini API Key missing in headers.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    # Configure Gemini with the user's provided API Key
    genai.configure(api_key=x_gemini_key)
    image_data = await file.read()

    # Step 1: OCR Extraction
    detected_meds = await extract_medications_with_ai(image_data)
    if not detected_meds:
        return {
            "status": "success",
            "detected_medications": [],
            "analysis": {"interactions_found": False, "disclaimer": "No medications identified."}
        }

    # Step 2: Parallel Data Fetching from FDA
    async with httpx.AsyncClient() as client:
        tasks = [fetch_fda_data(client, drug) for drug in detected_meds]
        results = await asyncio.gather(*tasks)
        fda_map = {drug: text for drug, text in results if text}

    # Step 3: Cross-referencing detected drugs
    interaction_evidence = []
    for i, drug_a in enumerate(detected_meds):
        for drug_b in detected_meds[i+1:]:
            # Check bi-directional interactions in the FDA map
            if drug_a in fda_map and drug_b.lower() in fda_map[drug_a]:
                interaction_evidence.append({"pair": f"{drug_a} + {drug_b}", "snippet": fda_map[drug_a]})
            elif drug_b in fda_map and drug_a.lower() in fda_map[drug_b]:
                interaction_evidence.append({"pair": f"{drug_a} + {drug_b}", "snippet": fda_map[drug_b]})

    # Step 4: Final Report Generation
    if interaction_evidence:
        report_json = await generate_safety_report(interaction_evidence, lang)
        try:
            final_report = json.loads(report_json)
        except:
            final_report = {"interactions_found": True, "error": "AI response format error."}
    else:
        final_report = {"interactions_found": False, "disclaimer": "No interactions found in FDA records."}

    return {
        "status": "success",
        "detected_medications": detected_meds,
        "analysis": final_report
    }