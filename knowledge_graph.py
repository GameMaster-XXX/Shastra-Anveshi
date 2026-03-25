#knowledge_graph.py
import os
import time
import csv
import re
import json
import dotenv
from docx import Document
from unidecode import unidecode
from indic_transliteration import sanscript
from openai import OpenAI

# Load environment variables
dotenv.load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_M_API") 
client = OpenAI(base_url="https://api.sarvam.ai/v1", api_key=SARVAM_API_KEY)

# Configuration
INPUT_FOLDER = "Chapter_Final"
OUTPUT_CSV = "gita_knowledge_graph_triplets.csv"

def transliterate_and_normalize(text):
    """Convert Devanagari to IAST and then to Unidecode."""
    if not text: return ""
    # Translating from DEVANAGARI to IAST first
    iast_text = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
    return unidecode(iast_text).strip()

def parse_only_shlokas(file_path):
    """Extracts raw Sanskrit Shloka text from .docx files."""
    doc = Document(file_path)
    shlokas = []
    
    # Extract Chapter Number from filename (e.g., 'Chapter_02.docx' -> 2)
    match = re.search(r'(\d+)', os.path.basename(file_path))
    chapter_no = int(match.group(1)) if match else 0

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text: continue
        
        # Updated Regex to match Devanagari numerals (०-९) or standard digits
        # Matches formats like: ॥ २.४७ ॥ or | ४७ | or standard 2.47
        shloka_match = re.search(r'([०-९\d]+)\.([०-९\d]+)|[।॥]\s*([०-९\d]+)\s*[।॥]', text)
        
        if shloka_match:
            # Extract numbers and convert Devanagari digits to standard int string
            raw_no = shloka_match.group(2) or shloka_match.group(3)
            # Normalize Devanagari digits to standard Western digits for the ID
            shloka_no = unidecode(raw_no) 
            
            # Normalize the full text for the API
            normalized_text = transliterate_and_normalize(text)
            
            shlokas.append({
                "id": f"BG_{chapter_no}_{shloka_no}",
                "text": normalized_text
            })
            
    return shlokas

def get_sarvam_triplets(shloka_batch):
    """Extracts triplets using Sarvam-M."""
    system_prompt = "Act as a formal logic extractor. Extract semantic triplets (Subject, Predicate, Object) only. Output valid JSON list of objects only."
    
    user_content = "Extract triplets for these Shlokas:\n" + \
                   "\n".join([f"ID: {s['id']} | Text: {s['text']}" for s in shloka_batch]) + \
                   "\n\nOutput format: [{\"shloka_id\": \"ID\", \"triplets\": [[\"S\", \"P\", \"O\"]]}]"

    try:
        # Note: Using 'client' defined in global scope
        response = client.chat.completions.create(
            model="sarvam-m",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=2048, # Increased for batch processing
            temperature=0.1
        )
        
        raw_output = response.choices[0].message.content
        json_match = re.search(r'\[.*\]', raw_output, re.DOTALL)
        return json.loads(json_match.group()) if json_match else []
    except Exception as e:
        print(f"Sarvam API Error: {e}")
        return []

def main():
    # Ensure sequential processing by sorting files
    files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(".docx")])
    
    # Initialize CSV if it doesn't exist
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Chapter_Shloka_id", "Subject", "Predicate", "Object"])

    for filename in files:
        file_path = os.path.join(INPUT_FOLDER, filename)
        print(f"--- Sequential Processing: {filename} ---")
        
        shlokas = parse_only_shlokas(file_path)
        if not shlokas:
            print(f"Warning: No shlokas found in {filename}. Check your regex or document formatting.")
            continue
            
        # Process in batches of 10
        for i in range(0, len(shlokas), 10):
            batch = shlokas[i:i+10]
            print(f"  Sending batch {i//10 + 1} ({batch[0]['id']} to {batch[-1]['id']})...")
            
            results = get_sarvam_triplets(batch)
            
            # Append results to CSV immediately to prevent data loss
            with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for item in results:
                    sid = item.get("shloka_id")
                    for t in item.get("triplets", []):
                        if len(t) == 3:
                            writer.writerow([sid, t[0], t[1], t[2]])
            
            # Cooldown logic
            if i + 10 < len(shlokas):
                print("  Cooldown: Waiting 15 seconds...")
                time.sleep(15)

if __name__ == "__main__":
    main()