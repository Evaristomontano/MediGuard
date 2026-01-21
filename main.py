#.\venv\Scripts\activate
#python -m uvicorn main:app 
import os
import io
import json
import asyncio
import logging
import base64

# Third-party imports
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pillow_heif
import httpx
from fastapi.responses import FileResponse # Añade esto a tus imports
from fastapi.staticfiles import StaticFiles # Añade esto a tus imports
# AI Providers
import google.generativeai as genai





# --- 1. CONFIGURATION & SETUP ---

# Configure logging (Standard practice for production apps)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# Register HEIF opener for iPhone photos support
pillow_heif.register_heif_opener()

# Initialize FastAPI app
app = FastAPI(
    title="MediGuard API",
    description="Backend for drug interaction detection using Multi-LLM architecture.",
    version="2.0.0"
)

@app.get("/")
async def read_index():
    return FileResponse('index.html')

# 3. Ruta para el settings.html
@app.get("/settings")
async def read_settings():
    return FileResponse('settings.html')

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. HELPER FUNCTIONS (BUSINESS LOGIC) ---

def clean_json_string(text: str) -> str:
    """
    Utility to strip Markdown formatting from AI responses
    to ensure valid JSON parsing.
    """
    return text.replace("```json", "").replace("```", "").strip()

async def fetch_fda_data(client: httpx.AsyncClient, drug: str):
    """
    Fetches drug interaction data from the openFDA API asynchronously.
    """
    url = f'https://api.fda.gov/drug/label.json?search=drug_interactions:"{drug}"&limit=1'
    try:
        response = await client.get(url, timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            # Extract the interaction section safely
            results = data.get('results', [])
            if results:
                interactions = results[0].get('drug_interactions', [""])
                return drug, interactions[0].lower()
    except Exception as e:
        logger.error(f"FDA API Error for {drug}: {str(e)}")
    return drug, None

async def extract_medications_with_ai(image_bytes: bytes) -> list:
    """
    Step 1: OCR & Entity Extraction.
    Strategy: Try Gemini Vision first.
    """
    prompt = """
    Analyze this medical image.
    1. Extract all specific drug names.
    2. Return ONLY the generic English names separated by commas (e.g., 'Ibuprofen, Paracetamol').
    3. If no drugs are visible, return 'NONE'.
    """

    # --- PLAN A: GEMINI VISION ---
    try:
        logger.info("Attempting medication extraction with Gemini Vision...")
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        # Open image with Pillow
        img = Image.open(io.BytesIO(image_bytes))
        
        # Gemini handles the image object directly
        response = await model.generate_content_async([prompt, img])
        text = response.text.strip()
        
        if "NONE" in text.upper():
            return []
        return [med.strip() for med in text.split(",")]

    except Exception as e:
        logger.error(f"Gemini Vision failed: {e}. Initiating fallback...")

   
    
    return []

async def generate_safety_report(interactions_data: list, lang: str) -> str:
    """
    Step 3: AI Analysis & Summarization.
    Strategy: Try Gemini first.
    """
    # Map language codes to full names for the prompt
    lang_map = {"en": "English", "es": "Spanish", "de": "German"}
    target_lang = lang_map.get(lang, "English")

    # Build context from FDA data
    context = ""
    for item in interactions_data:
        context += f"Drug Pair: {item['pair']}\nFDA Warning Snippet: {item['snippet'][:800]}...\n---\n"

    system_prompt = f"""
    You are an expert clinical pharmacist.
    Task: Analyze the provided FDA warning snippets for drug interactions.
    
    Output Requirements:
    1. Language: Write the 'risk' and 'action' fields strictly in {target_lang}.
    2. Format: Return ONLY a valid JSON object. Do not include markdown blocks.
    3. Severity Levels: Use only "CRITICAL", "CAUTION", or "INFO".
    
    JSON Structure:
    {{
        "interactions_found": true,
        "details": [
            {{
                "pair": "Drug A + Drug B",
                "severity": "CRITICAL",
                "risk": "Concise risk explanation in {target_lang}",
                "action": "Actionable advice in {target_lang}"
            }}
        ],
        "disclaimer": "Standard medical disclaimer in {target_lang}"
    }}
    """
    
    user_prompt = f"Here is the data found from FDA database:\n{context}"

    # --- PLAN A: GEMINI ---
    try:
        logger.info("Generating report with Gemini...")
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = await model.generate_content_async(system_prompt + "\n" + user_prompt)
        return clean_json_string(response.text)

    except Exception as e:
        logger.error(f"Gemini generation failed: {e}. Initiating fallback...")

    

    # Final fallback if both fail
    return json.dumps({
        "interactions_found": False,
        "disclaimer": "Service currently unavailable. Please consult a doctor."
    })

# --- 3. API ENDPOINTS ---

@app.get("/")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "active", "providers": ["Gemini"]}

@app.post("/analyze-prescription/")
async def analyze_prescription(file: UploadFile = File(...), lang: str = "en",x_gemini_key: str = Header(None)):
    """
    Main pipeline:
    1. Validate Image -> 2. Extract Meds (OCR) -> 3. Fetch FDA Data -> 4. AI Analysis
    """


    # 1. Validation
    if not x_gemini_key:
        raise HTTPException(status_code=401, detail="API Key missing. Please configure it in Settings.")

 
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
       # IMPORTANTE: Re-configurar Gemini con la clave del usuario para esta petición
    genai.configure(api_key=x_gemini_key)
    image_data = await file.read()

    # 2. Extraction (OCR)
    detected_meds = await extract_medications_with_ai(image_data)
    logger.info(f"Medications detected: {detected_meds}")

    if not detected_meds:
        return {
            "status": "success",
            "detected_medications": [],
            "analysis": {"interactions_found": False, "disclaimer": "No medications detected."}
        }

    # 3. Data Fetching (FDA) - Parallel Execution
    # We use httpx.AsyncClient to make concurrent requests to the FDA API
    fda_matches = []
    async with httpx.AsyncClient() as client:
        tasks = [fetch_fda_data(client, drug) for drug in detected_meds]
        # Gather allows us to run all requests at the same time
        results = await asyncio.gather(*tasks)
        
        # Build a dictionary for quick lookup
        fda_map = {drug: text for drug, text in results if text}

    # 4. Interaction Logic (Cross-referencing)
    interaction_evidence = []
    for i, drug_a in enumerate(detected_meds):
        for drug_b in detected_meds[i+1:]:
            # Check A -> B
            if drug_a in fda_map and drug_b.lower() in fda_map[drug_a]:
                interaction_evidence.append({
                    "pair": f"{drug_a} + {drug_b}",
                    "snippet": fda_map[drug_a]
                })
            # Check B -> A (Interaction might be listed on the other label)
            elif drug_b in fda_map and drug_a.lower() in fda_map[drug_b]:
                interaction_evidence.append({
                    "pair": f"{drug_a} + {drug_b}",
                    "snippet": fda_map[drug_b]
                })

    # 5. Final Report Generation
    if interaction_evidence:
        raw_json_report = await generate_safety_report(interaction_evidence, lang)
        try:
            final_report = json.loads(raw_json_report)
        except json.JSONDecodeError:
             logger.error("Failed to parse AI JSON response.")
             final_report = {"interactions_found": True, "error": "Parsing Error"}
    else:
        final_report = {
            "interactions_found": False, 
            "disclaimer": "No interactions found in FDA labels."
        }

    return {
        "status": "success",
        "detected_medications": detected_meds,
        "analysis": final_report
    }

# Run with: uvicorn main:app --reload