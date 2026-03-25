# generator.py
import os
import re
import asyncio
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

sarvam_api_key = os.getenv("SARVAM_M_API")

sarvam = OpenAI(
    base_url="https://api.sarvam.ai/v1",
    api_key=sarvam_api_key
)

LLM_SYSTEM_PROMPT_BASE = """
Before answering, you MUST first internally infer the INTENT of the user’s question.

Possible intents include (but are not limited to):
• Ontological explanation
• Cosmological process
• Soteriological attainment
• Upāsanā / comparative practice

Your answer MUST be structured ONLY according to the inferred intent.
Do NOT include other doctrinal layers unless the question explicitly demands them.

You are an Acharya, a revered scholar deeply knowledgeable in the tradition of Advaita Vedanta. Your purpose is to illuminate the user's query by drawing **EXCLUSIVELY** from the provided excerpts of the Bhashya (commentary) on the Bhagavad Gita.
 
**CRITICAL GUIDING PRINCIPLES:**
1.  **Scholarly Perspective**: Speak exclusively from the perspective of the Bhashya. Do NOT label specific schools like 'Advaita Vedanta states...'. Instead use 'According to the Bhashya...'
2.  **Source STRICTLY Limited**: Use ONLY the provided 'Scholarly Material'. No external knowledge.
3.  **Concise & Targeted Synthesis**: Prioritize relevant chunks. Synthesize a coherent explanation.
4.  **MANDATORY CITATION & SHLOKA PRINTING**: Every quote or summary MUST be followed by [Source: Chapter X Shloka Y].
5.  **SANSKRIT VERSE RULE**: If the Sanskrit Shloka text is provided, you MUST print it exactly as provided before the commentary.
6.  **Honesty**: If the answer isn't in the context, state: "Based on the provided context, I cannot answer this question."
"""

try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except:
    tokenizer = tiktoken.get_encoding("p50k_base")

def estimate_tokens(text):
    if not isinstance(text, str):
         text = str(text)
    return len(tokenizer.encode(text))

def limit_context_sliding_window(context_lines, system_prompt, user_query, reply_lang, max_tokens=7500):
    """
    Issue 5 Solution: Sliding Window Context Assembly.
    Uses a smaller safety margin (300) to maximize context.
    """
    instructions = f"""Answer STRICTLY in {reply_lang}.
    1. CONTEXT IS KING: Use ONLY scholarly material.
    2. INTENT-DRIVEN: Infer user intent and choose excerpts that serve it.
    3. CITE EVERYTHING: End every derived sentence with its [Source: Chapter X Shloka Y].
    """

    fixed_parts = f"{system_prompt}\n{instructions}\n\nUser's question:\n{user_query}\n\nAnswer:"
    fixed_tokens = estimate_tokens(fixed_parts)
    
    # Formula Update (Issue 5): Reduced safety_margin from 1000 to 300
    safety_margin = 300
    available_tokens = max_tokens - fixed_tokens - safety_margin

    if available_tokens <= 0:
        return [], instructions

    selected_lines = []
    current_tokens = 0
    separator_tokens = estimate_tokens("\n------\n")

    for line in context_lines:
        line_tokens = estimate_tokens(line) + separator_tokens
        if current_tokens + line_tokens <= available_tokens:
            selected_lines.append(line)
            current_tokens += line_tokens
        else:
            # Smart Truncation: If the chunk is highly relevant but too large, 
            # we take the top slice instead of dropping it
            remaining = available_tokens - current_tokens
            if remaining > 100:
                chars_to_take = int(len(line) * (remaining / line_tokens))
                truncated = line[:chars_to_take] + "... [truncated for window]"
                selected_lines.append(truncated)
            break

    return selected_lines, instructions

def construct_prompt_with_citations(retrieved_chunks, user_query, query_word, reply_lang):
    """Constructs final RAG prompt using sliding window assembly."""
    MAX_PROMPT_TOKENS = 7500

    if not retrieved_chunks:
        return f"{LLM_SYSTEM_PROMPT_BASE}\nRespond ONLY with: 'I could not find relevant information...'", []

    context_lines = []
    for chunk in retrieved_chunks:
        chapter = chunk.get('chapter', 'Unknown')
        shloka = chunk.get('shloka_no', 'None')
        source_info = f"[Source: Chapter {chapter} Shloka {shloka}]"
        text_content = chunk.get('text', '')
        context_lines.append(f"{source_info}\n{text_content}")

    # Use sliding window logic
    selected_lines, instructions = limit_context_sliding_window(
        context_lines, LLM_SYSTEM_PROMPT_BASE, user_query, reply_lang, MAX_PROMPT_TOKENS
    )

    context = "\n------\n".join(selected_lines)
    used_chunks = []
    
    # Matching logic to track which chunks were actually sent
    for line, chunk in zip(context_lines, retrieved_chunks):
        for sel in selected_lines:
            if line[:50] in sel: # Match based on start of chunk
                used_chunks.append(chunk)
                break

    prompt = f"{LLM_SYSTEM_PROMPT_BASE}\n{instructions}\n\nScholarly Material:\n{context}\n\nUser: {user_query}\nAnswer:"
    return prompt, used_chunks

def call_llm_api(prompt):
    """Calls Sarvam-M with enough head-room to finish long Bhashya explanations."""
    try:
        response = sarvam.chat.completions.create(
            model="sarvam-m",
            messages=[{"role": "user", "content": prompt}],
            # Increased to 2500 to prevent mid-sentence stops
            max_tokens=5000, 
            temperature=0.01,
            # Adding stop sequences if necessary to prevent infinite loops 
            # but usually not needed for Sarvam-M
        )
        content = response.choices[0].message.content.strip()
        
        # Check if response seems incomplete (doesn't end with punctuation)
        if content and content[-1] not in ['.', '!', '।', '॥', '"', '”']:
            content += "... [Response truncated by model]"
            
        return content
    except Exception as e:
        return f"Error during generation: {str(e)}"