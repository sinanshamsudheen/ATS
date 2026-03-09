"""Resume text parser — splits raw text into structured sections and bullets."""

import re
from typing import Dict, List, Any


class ResumeParser:
    """Parses resume text into structured sections with bullet extraction."""

    SECTION_HEADERS = {
        "experience": ["experience", "work history", "employment", "professional experience"],
        "education": ["education", "academic background", "qualifications"],
        "skills": ["skills", "technical skills", "competencies", "technologies"],
        "projects": ["projects", "personal projects", "academic projects"],
        "summary": ["summary", "profile", "objective", "about me"],
        "certifications": ["certifications", "licenses", "courses"],
        "contact": ["contact"],
    }

    def parse(self, text: str) -> Dict[str, Any]:
        """Convert raw resume text into a structured dictionary."""
        lines = text.split('\n')
        sections: Dict[str, Any] = {key: "" for key in self.SECTION_HEADERS}
        sections["contact"] = []

        current_section = "contact"

        for line in lines:
            line_clean = line.strip().lower()

            is_header = False
            for section, headers in self.SECTION_HEADERS.items():
                if any(h == line_clean or f"{h}:" == line_clean for h in headers):
                    current_section = section
                    is_header = True
                    break

            if not is_header:
                if isinstance(sections[current_section], list):
                    sections[current_section].append(line)
                else:
                    sections[current_section] += line + "\n"

        contact_text = (
            "\n".join(sections["contact"])
            if isinstance(sections["contact"], list)
            else sections["contact"]
        )

        structured_data: Dict[str, Any] = {
            "contact_info": self._extract_contact_info(contact_text),
            "sections": {},
        }

        for key in sections:
            if key != "contact":
                structured_data["sections"][key] = {
                    "text": sections[key].strip() if isinstance(sections[key], str) else "",
                    "bullets": self._extract_bullets(
                        sections[key] if isinstance(sections[key], str) else ""
                    ),
                }

        return structured_data

    def _extract_bullets(self, text: str) -> List[str]:
        """Extract bullet points from text using common bullet markers."""
        bullets: List[str] = []
        bullet_pattern = re.compile(
            r'^(\s*[\u2022\u2023\u25E6\u2043\u2219\u25CF\u25CB\u25C6\u25C7'
            r'\u25A0\u25A1\u25AA\u25AB\*\-\+\u2022\u25CB\u25CF\u25C6\u25C7'
            r'\u25A0\u25A1\u25AA\u25AB]\s+)'
        )

        for line in text.split('\n'):
            if bullet_pattern.match(line):
                clean_line = bullet_pattern.sub('', line).strip()
                if clean_line:
                    bullets.append(clean_line)
            elif re.match(r'^\s*\d+[\.\)]\s+', line):
                clean_line = re.sub(r'^\s*\d+[\.\)]\s+', '', line).strip()
                if clean_line:
                    bullets.append(clean_line)

        return bullets

    def _extract_contact_info(self, text: str) -> Dict[str, Any]:
        """Extract email and phone from contact text."""
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        phone_pattern = r'(\+\d{1,2}\s?)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}'

        emails = re.findall(email_pattern, text)
        phones = re.findall(phone_pattern, text)

        return {
            "email": emails[0] if emails else None,
            "phone": phones[0] if phones else None,
        }
