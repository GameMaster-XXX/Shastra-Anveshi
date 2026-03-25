import os
import json
import time
import re
from openai import OpenAI
from dotenv import load_dotenv

# Assuming your data_ingestion and parser are in the same directory
from data_ingestion import load_file
from parser import parse_shlokas

load_dotenv()

# --- CONFIGURATION ---
SARVAM_API_KEY = os.getenv("SARVAM_M_API")
# Using the updated 2026 Sarvam-M endpoint and model name
client = OpenAI(base_url="https://api.sarvam.ai/v1", api_key=SARVAM_API_KEY)

OUTPUT_JSON = "gita_saint_ontology.json"
CHAPTER_FOLDER = "Chapters"

def get_sarvam_theme(chapter_no, shloka_no, shloka_text, explanation_text):
    """
    Synthesizes a 3-4 sentence theme based on the Shloka AND the Saint's Bhashya.
    """
    # The prompt explicitly asks to prioritize the Saint's POV found in the explanation
    prompt = f"""
    You are a Vedic Research Assistant. Your task is to extract a 'Thematic Summary' of a Bhagavad Gita verse based STRICTLY on the provided Saint's Commentary (Bhashya).

    CONTEXT:
    Chapter: {chapter_no}
    Shloka Number: {shloka_no}
    
    SANSKRIT SHLOKA:
    {shloka_text}

    SAINT'S EXPLANATION (BHASHYA):
    {explanation_text}

    TASK:
    1. Analyze how the Saint interprets this specific Shloka.
    2. Identify any cross-references to other verses mentioned in the Bhashya.
    3. Generate a 3-4 sentence "Theme" in English that captures the unique spiritual insight of the Saint.

    RULES:
    - Do NOT use outside knowledge.
    - Do NOT transliterate; keep Sanskrit words in Devanagari.
    - If no cross-references are found, return an empty list.

    RETURN JSON ONLY:
    {{
        "theme": "3-4 sentence summary of the Saint's POV...",
        "speaker": "Krishna/Arjuna/Sanjaya/Dhritarashtra",
        "cross_references": ["Ch.Verse", "Ch.Verse"]
    }}
    """
    
    try:
        # Using Sarvam-M's reasoning capability for better synthesis
        response = client.chat.completions.create(
            model="sarvam-m",
            messages=[
                {"role": "system", "content": "You are a precise textual scholar specializing in Indian scriptures."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, # Lower temperature for consistency
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  [Error] API call failed for {chapter_no}.{shloka_no}: {e}")
        return None

def generate_master_ontology():
    master_ontology = {}
    
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            master_ontology = json.load(f)

    files = sorted([f for f in os.listdir(CHAPTER_FOLDER) if f.endswith(('.txt', '.docx', '.pdf'))])

    for filename in files:
        path = os.path.join(CHAPTER_FOLDER, filename)
        match = re.search(r'(\d+)', filename)
        ch_no = int(match.group(1)) if match else 0
        
        print(f"\n>>> Processing Chapter {ch_no}...")
        lines = load_file(path)
        units = parse_shlokas(lines, chapter_no=ch_no)

        if str(ch_no) not in master_ontology:
            master_ontology[str(ch_no)] = {}

        for unit in units:
            # Handle both Shlokas and Chapter Summaries (where shloka_no might be None)
            shloka_id = str(unit['shloka_no']) if unit['shloka_no'] else "summary"
            
            if shloka_id in master_ontology[str(ch_no)]:
                continue
                
            print(f"  Generating Saint's POV for {ch_no}.{shloka_id}...")
            
            # Key change: Pass Shloka and Explanation separately to the prompt
            data = get_sarvam_theme(ch_no, shloka_id, unit['shloka'], unit['explanation'])
            
            if data:
                # Store everything: Original text + Generated Theme
                master_ontology[str(ch_no)][shloka_id] = {
                    "shloka": unit['shloka'],
                    "explanation": unit['explanation'],
                    **data
                }
                
                with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                    json.dump(master_ontology, f, indent=4, ensure_ascii=False)
                
                time.sleep(2) # Balanced cooldown for Sarvam-M API

if __name__ == "__main__":
    generate_master_ontology()