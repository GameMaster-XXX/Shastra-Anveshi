import os
import docx
import pymupdf  # PyMuPDF for PDFs
from PIL import Image  # For image handling in OCR
import pytesseract  # For OCR on images/Sanskrit scripts
import io  # For BytesIO in PDF image rendering

# Set Tesseract path if not in system PATH (adjust as needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set TESSDATA_PREFIX for language data (adjust if custom tessdata dir)
if 'TESSDATA_PREFIX' not in os.environ:
    os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

def load_docx(path):
    doc = docx.Document(path)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]

def load_pdf(path):
    doc = pymupdf.open(path)
    lines = []
    page_count = len(doc)
    has_text = False
    text_lines_per_page = []
    
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").split("\n")
        stripped = [t.strip() for t in text if t.strip()]
        if stripped:
            lines.extend(stripped)
            has_text = True
        text_lines_per_page.append(len(stripped))
    
    # Calculate average lines per page
    avg_lines_per_page = sum(text_lines_per_page) / page_count if page_count > 0 else 0
    
    # Fallback to OCR if minimal text extracted (scanned PDF with images)
    if not has_text or avg_lines_per_page < 10:  # Heuristic: <10 lines/page suggests scanned/images
        print(f"Low text in PDF '{os.path.basename(path)}' (avg {avg_lines_per_page:.1f} lines/page); using OCR fallback...")
        lines = []
        for page_num, page in enumerate(doc, start=1):
            # Render page to image (pixmap, 1.5 DPI scale for better OCR clarity)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5))  # Scale up for sharper input
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            
            # OCR the image with Sanskrit/Devanagari config
            ocr_text = pytesseract.image_to_string(img, lang='san',  # Use 'san' for pure Sanskrit; 'san+hin' if mixed
                                                   config='--oem 3 --psm 6 -c tessedit_char_whitelist=०-९अ-हऽा-ौैंऔं।॥')  # Whitelist Devanagari + punctuation
            page_lines = [line.strip() for line in ocr_text.split("\n") if line.strip()]
            lines.extend(page_lines)
            print(f"OCR'd PDF page {page_num}: {len(page_lines)} lines")
    
    doc.close()
    return lines

def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def load_image(path):
    if not os.path.exists(path):
        raise ValueError(f"Image file not found: {path}")
    
    # Open image with PIL
    img = Image.open(path)
    
    # OCR with Sanskrit focus
    ocr_text = pytesseract.image_to_string(img, lang='san',  # Pure Sanskrit
                                           config='--oem 3 --psm 6 -c tessedit_char_whitelist=०-९अ-हऽा-ौैंऔं।॥')
    
    # Split into lines, strip empty
    lines = [line.strip() for line in ocr_text.split("\n") if line.strip()]
    print(f"OCR'd image '{os.path.basename(path)}': {len(lines)} lines extracted")
    
    # Optional post-processing: Clean to Devanagari/Sanskrit only (uncomment if needed)
    # import re
    # lines = [re.sub(r'[^\u0900-\u097F\s।॥]', '', line) for line in lines]
    
    return lines

def load_file(path):
    if path.endswith(".docx"):
        return load_docx(path)
    elif path.endswith(".pdf"):
        return load_pdf(path)
    elif path.endswith((".txt", ".text")):
        return load_txt(path)
    elif path.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        return load_image(path)
    else:
        raise ValueError(f"Unsupported file type for {path}. Supported: DOCX, PDF (text or OCR fallback), TXT, JPG/JPEG/PNG/BMP/TIFF (OCR)")