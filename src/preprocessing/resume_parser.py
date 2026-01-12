import re
from typing import Dict, List, Any

class ResumeParser:
    """
    Parses resume text into structured sections.
    """
    
    SECTION_HEADERS = {
        "experience": ["experience", "work history", "employment", "professional experience"],
        "education": ["education", "academic background", "qualifications"],
        "skills": ["skills", "technical skills", "competencies", "technologies"],
        "projects": ["projects", "personal projects", "academic projects"],
        "summary": ["summary", "profile", "objective", "about me"],
        "certifications": ["certifications", "licenses", "courses"],
        "contact": ["contact"] # heuristic, usually at top
    }

    def __init__(self):
        pass

    def parse(self, text: str) -> Dict[str, Any]:
        """
        Main parse method to convert raw text into a structured dictionary.
        """
        lines = text.split('\n')
        sections = {key: "" for key in self.SECTION_HEADERS.keys()}
        sections["contact"] = [] # specialized handling if needed
        
        current_section = "contact" # Default starting section (usually top info)
        
        # Simple finite state machine for section tracking
        for line in lines:
            line_clean = line.strip().lower()
            
            # Check if line is a header
            is_header = False
            for section, headers in self.SECTION_HEADERS.items():
                if any(h == line_clean for h in headers) or \
                   any(f"{h}:" == line_clean for h in headers):
                    current_section = section
                    is_header = True
                    break
            
            if not is_header:
                if isinstance(sections[current_section], list):
                    sections[current_section].append(line)
                else:
                    sections[current_section] += line + "\n"
        
        # Post-processing
        structured_data = {
            "contact_info": self._extract_contact_info("\n".join(sections["contact"]) if isinstance(sections["contact"], list) else sections["contact"]),
            "sections": {}
        }
        
        for key in sections:
            if key != "contact":
                structured_data["sections"][key] = {
                    "text": sections[key].strip(),
                    "bullets": self._extract_bullets(sections[key])
                }
                
        return structured_data

    def _extract_bullets(self, text: str) -> List[str]:
        """
        Extracts bullet points from text using regex for common bullet markers.
        """
        bullets = []
        lines = text.split('\n')
        # Regex for common bullets: dot, dash, star, etc.
        bullet_pattern = re.compile(r'^(\s*[\u2022\u2023\u25E6\u2043\u2219\*\-\+]\s+)')
        
        for line in lines:
            if bullet_pattern.match(line):
                # Remove the bullet char
                clean_line = bullet_pattern.sub('', line).strip()
                if clean_line:
                    bullets.append(clean_line)
            # Handle also lines that look like list items but might not have bullet chars if structured well?
            # For now strict bullet detection to avoid noise.
            
        return bullets

    def _extract_contact_info(self, text: str) -> Dict[str, str]:
        """
        Extracts email and phone using regex.
        """
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        phone_pattern = r'(\+\d{1,2}\s?)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}'
        
        emails = re.findall(email_pattern, text)
        phones = re.findall(phone_pattern, text)
        
        return {
            "email": emails[0] if emails else None,
            "phone": phones[0] if phones else None # simplistic phone match, tuple issue with findall group need generic join
        }

if __name__ == "__main__":
    # Test
    sample_text = """
    John Doe
    john@example.com
    
    Experience
    • Worked at Google
    • Built search engine
    
    Education
    BS CS
    """
    parser = ResumeParser()
    print(parser.parse(sample_text))
