# parser.py
import re
from collections import defaultdict
import uuid
# --- NEW IMPORT ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
# --- Regex patterns (Unchanged) ---
shloka_end_pattern = re.compile(r"॥\s*([\d\u0966-\u096F]+)\s*॥")
reference_pattern = re.compile(r"\([\u0900-\u097F\. ]+\)")
 
# --- Utilities (Unchanged) ---
def devanagari_to_ascii_num(s: str) -> str:
    return "".join(str(ord(ch) - 0x0966) if '०' <= ch <= '९' else ch for ch in s)
 
def is_poetic_line(line: str) -> bool:
    return len(line) < 100 and (line.endswith("।") or line.endswith("॥") or "उवाच" in line or "आह" in line)
 
def is_explanation_block(text: str) -> bool:
    if reference_pattern.search(text):
        return True
    if len(text) > 350 and "।" not in text[:120]:
        return True
    return False
 
def is_explanation_start(line: str) -> bool:
    if line.startswith("‘") and "(भ. गी." in line:
        return True
    if line.startswith("(") and "।" in line and "भ. गी." in line:
        return True
    return False
 
# --- Main parser (Unchanged) ---
def parse_shlokas(lines: list[str], chapter_no: int = 1, source_file: str = None) -> list[dict]:
    """
    Parse shloka + explanation units from docx lines.
    Returns a list of dicts with metadata.
    This list of units *is* our list of Parent Documents.
    """
    units = []
    seen_counts = defaultdict(int)
    current_shloka = []
    inside_shloka = False
 
    for line in lines:
        if not line.strip():
            continue
 
        if is_explanation_start(line):
            if units:
                units[-1]["explanation"] += "\n" + line
            else:
                units.append({
                    "chapter": chapter_no, "shloka_no": None, "shloka": "",
                    "explanation": line, "source_file": source_file
                })
            continue
 
        if is_explanation_block(line):
            if units:
                units[-1]["explanation"] += "\n" + line
            else:
                units.append({
                    "chapter": chapter_no, "shloka_no": None, "shloka": "",
                    "explanation": line, "source_file": source_file
                })
            continue
 
        m = shloka_end_pattern.search(line)
        if m and is_poetic_line(line):
            num = devanagari_to_ascii_num(m.group(1))
            seen_counts[num] += 1
            current_shloka.append(line)
 
            if seen_counts[num] == 1:
                units.append({
                    "chapter": chapter_no, "shloka_no": num,
                    "shloka": "\n".join(current_shloka).strip(),
                    "explanation": "", "source_file": source_file
                })
                current_shloka = []
                inside_shloka = False
            else:
                if units:
                    units[-1]["explanation"] += "\n" + line
            continue
 
        if is_poetic_line(line) or inside_shloka:
            current_shloka.append(line)
            inside_shloka = True
        else:
            if units:
                units[-1]["explanation"] += "\n" + line
            else:
                units.append({
                    "chapter": chapter_no, "shloka_no": None, "shloka": "",
                    "explanation": line, "source_file": source_file
                })
 
    return units
 
 
def create_parent_and_child_documents(
    units: list[dict],
    child_chunk_size: int = 400,
    child_chunk_overlap: int = 50
) -> (list[dict], list[dict]):
    """
    Creates parent and child documents based on a hybrid strategy:
    - Parents: Full "Sloka + Bhashya" text.
    - Children:
        1. The full Shloka text (as one chunk).
        2. Small chunks of the Bhashya (Explanation) for embedding.
    """
    parent_documents = []
    child_documents = []
    global_child_idx = 1 # We will use a global index for children
 
    # This splitter is for creating the small, searchable child chunks from the Bhashya
    child_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "।", "॥", "\n", " ", ""],
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap
    )
 
    for unit in units:
        # --- Common Data Extraction ---
        shloka_text = unit.get("shloka", "").strip()
        explanation_text = unit.get("explanation", "").strip()
        parent_text = (shloka_text + "\n\n" + explanation_text).strip()
       
        if not parent_text:
            continue
 
        # Generate a unique ID for this parent
        parent_id = str(uuid.uuid4())
       
        # Get metadata
        chapter = unit.get("chapter")
        shloka_no = unit.get("shloka_no")
        source_file = unit.get("source_file")
       
        common_metadata = {
            "chapter": chapter,
            "shloka_no": str(shloka_no) if shloka_no is not None else "None",
            "source_file": source_file,
            "parent_id": parent_id # Foreign key to parent
        }
 
        # 1. Create the Parent Document
        parent_doc = {
            "parent_id": parent_id,
            "text": parent_text,
            **common_metadata
        }
        parent_documents.append(parent_doc)
 
        # 2. Create Child Documents
       
        # --- A. Child Chunk for the whole Shloka ---
        if shloka_text:
            child_doc_shloka = {
                "global_index": global_child_idx, # Primary key for child collection
                "chunk_type": "shloka", # New metadata to identify the chunk type
                "text": shloka_text,
                "has_bhashya": bool(explanation_text),
                "start_char": 0,
                "end_char": len(shloka_text),
                **common_metadata
            }
            child_documents.append(child_doc_shloka)
            global_child_idx += 1
 
 
        # --- B. Child Chunks for the Explanation (Bhashya) ---
        if explanation_text:
            child_chunks = child_splitter.split_text(explanation_text)
           
            # Approximate starting position for Bhashya in the parent_text
            # It starts right after the Shloka text and the two newlines
            explanation_offset = len(shloka_text) + 2 if shloka_text else 0
 
            for i, chunk_text in enumerate(child_chunks):
                # Calculate the approximate start/end of the chunk within the Bhashya text
                approx_start_in_bhashya = i * (child_chunk_size - child_chunk_overlap)
                approx_end_in_bhashya = approx_start_in_bhashya + len(chunk_text)
               
                child_doc_bhashya = {
                    "global_index": global_child_idx,
                    "chunk_type": "bhashya", # New metadata to identify the chunk type
                    "text": chunk_text,
                    "has_bhashya": True,
                    # Approximate character range within the *full parent text*
                    "start_char": explanation_offset + approx_start_in_bhashya,
                    "end_char": explanation_offset + approx_end_in_bhashya,
                    **common_metadata
                }
                child_documents.append(child_doc_bhashya)
                global_child_idx += 1
 
    return parent_documents, child_documents