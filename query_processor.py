from langdetect import detect
from openai import OpenAI
import json
import re
import os
from unidecode import unidecode
from dotenv import load_dotenv
from indic_transliteration import sanscript

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_M_API") 
sarvam = OpenAI(base_url="https://api.sarvam.ai/v1", api_key=SARVAM_API_KEY)

def detect_query_language(user_query):
    """Detects input language to route translation logic."""
    try:
        detected_lang = detect(user_query)
    except:
        detected_lang = "en"
    lang_map = {
        "en": "English", "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
        "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "gu": "Gujarati",
        "pa": "Punjabi", "or": "Odia", "sa": "Sanskrit",
    }
    return lang_map.get(detected_lang, "English")

def classify_query_intent(user_query):
    """Categorizes query for dynamic reranking thresholds."""
    is_direct_shloka = any(p in user_query for p in ["।", "॥"])
    if is_direct_shloka:
        return "verse_lookup"
    
    has_devanagari = bool(re.search(r'[\u0900-\u097F]', user_query))
    has_question_words = any(w in user_query.lower() for w in ["how", "why", "who", "what", "किम", "कथम", "katham", "kim"])
    
    if has_devanagari and not has_question_words:
        return "verse_lookup"
    
    system_prompt = "Classify user intent: 'verse_lookup', 'definition', 'philosophical_reconciliation', or 'general'. Output only the label."
    try:
        response = sarvam.chat.completions.create(
            model="sarvam-m",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}],
            max_tokens=20,
            temperature=0.0
        )
        return response.choices[0].message.content.strip().lower()
    except:
        return "general"

def translate_query_to_sanskrit_pivot(user_query, source_lang):
    """Normalizes query into Sanskrit Devanagari for uniform vector anchoring."""
    if source_lang == "Sanskrit":
        return user_query
        
    system_prompt = "You are a Sanskrit scholar. Translate the user query into precise Sanskrit (Devanagari) using technical Vedantic terms from the Bhagavad Gita."
    prompt = f"Translate from {source_lang} to Sanskrit:\nQuery: {user_query}"

    try:
        response = sarvam.chat.completions.create(
            model="sarvam-m", 
            messages=[{"role":"system","content":system_prompt}, {"role":"user","content":prompt}],
            max_tokens=150,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Pivot Translation Error: {e}")
        return user_query

def extract_entities_for_graph(sanskrit_query):
    """Extracts core Subject/Object concepts for Neo4j matching."""
    system_prompt = "Extract primary Subject/Object concepts from the Sanskrit query. Output ONLY a comma-separated list in IAST (Romanized Sanskrit) without diacritics."
    try:
        response = sarvam.chat.completions.create(
            model="sarvam-m",
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":sanskrit_query}],
            max_tokens=100,
            temperature=0.1
        )
        text = response.choices[0].message.content.strip()
        # Ensure we only get words and commas
        clean_text = re.sub(r'[^a-zA-Z, ]', '', text)
        return [unidecode(e.strip()).lower() for e in clean_text.split(',') if e.strip()]
    except:
        return []

def batch_list(input_list, batch_size):
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]

def llm_listwise_rerank(chunks, sanskrit_pivot, original_query, intent="general", batch_size=5, final_top_k=10):
    """Reranks candidates with robust JSON extraction and error handling."""
    if not chunks:
        return []
    
    all_candidates = []
    
    # Reranker System Prompt
    system_prompt = f"""You are an Expert Ācārya Reranker. Intent: {intent}.
Assign relevance scores (0.0-10.0) based on how well the excerpt answers the user query.
Rules:
- High precision (8+) for direct shloka matches.
- Moderate scores (4-7) for conceptual relevance.
- Output ONLY a JSON object: {{"relevant": [{{"id": 0, "score": 9.5}}]}}"""

    for i, batch in enumerate(batch_list(chunks, batch_size)):
        batch_text = ""
        for idx, chunk in enumerate(batch):
            text_snippet = chunk.get('text', '')[:600].replace('\n', ' ')
            batch_text += f"[ID {idx}] {text_snippet}\n\n"

        user_prompt = f"User Question: {original_query}\nPivot: {sanskrit_pivot}\n\nExcerpts:\n{batch_text}\nJSON Output:"

        try:
            response = sarvam.chat.completions.create(
                model="sarvam-m",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.01,
                response_format={"type": "json_object"}
            )
            
            raw_content = response.choices[0].message.content
            
            # ROBUST JSON EXTRACTION (Handles <think> tags or extra prose)
            json_match = re.search(r'(\{.*\})', raw_content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                relevant_items = data.get("relevant", [])
                
                for item in relevant_items:
                    idx = int(item.get("id"))
                    score = float(item.get("score", 0.0))
                    if 0 <= idx < len(batch) and score >= 4.0:
                        chunk = batch[idx].copy()
                        chunk["llm_score"] = score
                        all_candidates.append(chunk)
            else:
                print(f"Batch {i+1}: No valid JSON found in response.")

        except Exception as e:
            print(f"Batch {i+1}: Rerank processing error: {e}")
            continue

    # Sort by score descending
    all_candidates.sort(key=lambda x: x.get('llm_score', 0), reverse=True)
    return all_candidates[:final_top_k]

def generate_localized_not_found(query, lang):
    """Acharya-style 'Not Found' message in the user's language."""
    system_instruction = f"You are an Acharya. Politely inform the user in {lang} that the requested verse/information was not found in the Bhagavad Gita commentaries."
    try:
        response = sarvam.chat.completions.create(
            model="sarvam-30b",
            messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": query}],
            max_tokens=150,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except:
        return "I apologize, but that specific context is not available in our records."