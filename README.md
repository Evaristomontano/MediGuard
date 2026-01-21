## Live Demo
You can try the application live on Hugging Face Spaces: 
[MediGuard on Hugging Face](https://huggingface.co/spaces/EvaMontano/MediGuard)


# MediGuard: AI-Powered Drug Interaction Analyzer

MediGuard is a web application designed to improve patient safety by identifying potential drug-drug interactions...
# MediGuard: AI-Powered Drug Interaction Analyzer

MediGuard is a web application designed to improve patient safety by identifying potential drug-drug interactions. The system uses Gemini 2.0 Flash to extract medication names from images and cross-references them with official FDA databases to provide clinical safety insights.

---

## Key Features

- Multimodal OCR: Extraction of drug names from photos using AI vision capabilities.
- FDA Integration: Automated data fetching from the openFDA API for clinical accuracy.
- Privacy-First Architecture: Users provide their own API keys to ensure data control and scalability.
- Multilingual Support: Full interface and analysis available in English, Spanish, and German.
- Responsive Design: Built with Tailwind CSS for compatibility across mobile and desktop devices.

---

## Architecture and Tech Stack

- Backend: Python / FastAPI using Asynchronous I/O.
- Frontend: Vanilla JavaScript / HTML5 / Tailwind CSS.
- AI Engine: Google Gemini 2.0 Flash API.
- Deployment: Dockerized environment for Hugging Face Spaces.
- Data Source: openFDA (U.S. Food and Drug Administration).

---

## Instructions for Use

1. Obtain a Gemini API Key: Access Google AI Studio to generate a free API key.
2. Configure Settings: Open the Configuration page (Settings) in the application and input your key.
3. Upload Image: Provide a clear photo of the medication label or prescription.
4. Review Report: Analyze the generated findings regarding potential risks and recommended actions.

---

## Medical Disclaimer

Important: This application is a tool for informational purposes only and does not provide medical advice. MediGuard is designed to identify potential risks but cannot replace the professional judgment of a healthcare provider. Always consult a doctor or pharmacist before making any changes to your medical treatment.

---

## Installation and Local Development

To run this project in a local environment:

1. Clone the repository:
   git clone https://huggingface.co/spaces/YOUR_USER/mediguard

2. Install dependencies:
   pip install -r requirements.txt

3. Start the development server:
   uvicorn main:app --reload

---

## License

This project is licensed under the MIT License. Use, modification, and distribution are permitted provided that the original author is credited.

---

Developed for Patient Safety Research.
