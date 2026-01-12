from typing import Dict, List, Any
import fitz
import re

class FormattingChecker:
    """
    Checks for ATS-friendly formatting issues.
    """
    
    def check(self, pdf_path: str) -> Dict[str, Any]:
        issues = []
        score_deductions = 0
        
        try:
            doc = fitz.open(pdf_path)
            
            # 1. Check for basic metadata issues
            if doc.is_encrypted:
                issues.append({"type": "critical", "message": "PDF is encrypted/password protected."})
                return {"issues": issues, "score": 0}

            # 2. Check for Tables (heuristic)
            # PyMuPDF doesn't explicitly detect tables easily without structure analysis,
            # but we can check for drawing paths that look like grids or many horizontal/vertical lines.
            # Simplified heuristic: check if text is extracted comfortably.
            
            # 3. Check for Images (Text vs Image ratio)
            total_text_len = 0
            has_images = False
            
            for page in doc:
                text = page.get_text()
                total_text_len += len(text)
                
                # Check for images
                if page.get_images():
                    has_images = True
            
            if total_text_len < 100 and has_images:
                issues.append({"type": "high", "message": "Resume appears to be an image (scanned). ATS cannot read this."})
                score_deductions += 100
            elif has_images:
                 issues.append({"type": "medium", "message": "Resume contains images/graphics which ATS may ignore."})
                 score_deductions += 15

            # 4. Check for Columns (heuristic: large gaps in lines?)
            # Hard to detect reliably without complex logic. Skip for MVP or use simple line analysis (if many lines have > 2 gaps)
            
            # 5. Check for Headers/Footers (heuristic: same text top/bottom of pages)
            if len(doc) > 1:
                p1_blocks = doc[0].get_text("blocks")
                p2_blocks = doc[1].get_text("blocks")
                # Very rough check... maybe skipping for MVP to keep it simple and robust.

            doc.close()
            
            # Score calculation
            final_score = max(0, 100 - score_deductions)
            
            return {
                "issues": issues,
                "score": final_score
            }

        except Exception as e:
            return {
                "issues": [{"type": "error", "message": f"Error analyzing formatting: {str(e)}"}],
                "score": 0
            }
