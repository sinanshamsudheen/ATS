import fitz  # PyMuPDF
import re
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file using PyMuPDF.
    
    Args:
        pdf_path (str): Path to the PDF file.
        
    Returns:
        str: Extracted clean text from the PDF.
        
    Raises:
        ValueError: If the file is not found or is not a valid PDF.
    """
    try:
        doc = fitz.open(pdf_path)
        full_text = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            full_text.append(text)
            
        doc.close()
        
        raw_text = "\n".join(full_text)
        cleaned_text = clean_text(raw_text)
        
        return cleaned_text
        
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        raise ValueError(f"Failed to process PDF: {str(e)}")

def clean_text(text: str) -> str:
    """
    Cleans extracted text by normalizing whitespace and removing artifacts.
    """
    # Replace multiple newlines with a single newline (preserve paragraphs roughly)
    # But often PDF extraction leaves weird line breaks. 
    # For now, let's normalize multiple spaces to single space 
    # and handle multiple newlines cautiously.
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Fix extensive whitespace
    # Replace 3+ newlines with 2 (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Trim lines
    lines = [line.strip() for line in text.split('\n')]
    
    # Rejoin
    text = '\n'.join(lines)
    
    return text.strip()

if __name__ == "__main__":
    # Simple manual test
    import sys
    if len(sys.argv) > 1:
        print(extract_text_from_pdf(sys.argv[1]))
