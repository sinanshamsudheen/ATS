import streamlit as st
import sys
import os
import tempfile
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.preprocessing.pdf_extractor import extract_text_from_pdf
from src.preprocessing.resume_parser import ResumeParser
from src.analysis.similarity import SimilarityAnalyzer
from src.analysis.formatting_check import FormattingChecker
from src.config import APP_TITLE, APP_VERSION

# Page Config
st.set_page_config(page_title=APP_TITLE, layout="wide")

# Title
st.title(f"📄 {APP_TITLE} v{APP_VERSION}")
st.markdown("Optimize your resume for Applicant Tracking Systems (ATS)")

# Sidebar
with st.sidebar:
    st.header("Instructions")
    st.info("1. Upload your Resume (PDF)\n2. Paste the Job Description\n3. Click Analyze")
    
    st.divider()
    st.caption("Powered by Local LLM & Embeddings")

# Main Interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Resume")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
with col2:
    st.subheader("2. Job Description")
    job_description = st.text_area("Paste JD text here", height=200, placeholder="Copy specific job requirements here...")

# Analyze Button
if st.button("Analyze Resume", type="primary"):
    if not uploaded_file:
        st.error("Please upload a resume first.")
    elif not job_description:
        st.error("Please provide a Job Description.")
    else:
        with st.spinner("Analyzing... (This may take a moment to load models)"):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # 1. Processing
                text = extract_text_from_pdf(tmp_path)
                
                # 2. Parsing
                parser = ResumeParser()
                parsed_data = parser.parse(text)
                
                # 3. Formatting Check
                fmt_checker = FormattingChecker()
                fmt_result = fmt_checker.check(tmp_path)
                
                # 4. Analysis
                analyzer = SimilarityAnalyzer() # This loads model, might be slow first time
                
                # Extract keywords from JD (Simplified assumption: JD is text, we compare text vs text mostly, 
                # but for keywords we might need a keyword extractor. 
                # For MVP, let's treat top frequent words or just use simple string splitting logic, 
                # OR just rely on SimilarityAnalyzer's full text match for now if keyword extraction isn't built.
                # PRD said "Identify missing keywords". We implemented `analyze_keywords`.
                # But we need a `jd_keywords` list. 
                # Let's add a simple keyword extractor helper for MVP in this block.
                
                def simple_extract_keywords(text):
                    # Very naive extractor for MVP demo
                    # In real app, use KeyBERT or similar
                    words = [w.strip() for w in text.split() if len(w) > 4]
                    return list(set(words))[:20] # Take first 20 unique long words (dumb heuristic)

                jd_keywords = simple_extract_keywords(job_description)
                
                keyword_analysis = analyzer.analyze_keywords(text, jd_keywords)
                overall_match = analyzer.calculate_overall_match(text, job_description)

                # Cleanup
                os.unlink(tmp_path)

                # --- Results Display ---
                st.divider()
                st.header("Analysis Results")
                
                # Metrics Row
                m1, m2, m3 = st.columns(3)
                m1.metric("Overall Match", f"{overall_match}%", delta_color="normal")
                m2.metric("Keyword Coverage", f"{keyword_analysis['score']}%")
                m3.metric("Formatting Score", f"{fmt_result['score']}/100")
                
                # Detailed View
                tab1, tab2, tab3 = st.tabs(["Missing Keywords", "Formatting Issues", "Resume Content"])
                
                with tab1:
                    if keyword_analysis["missing"]:
                        st.warning(f"Missing {len(keyword_analysis['missing'])} potential keywords:")
                        st.write(", ".join(keyword_analysis["missing"]))
                    else:
                        st.success("Great! No obvious missing keywords found based on simple heuristic.")
                        
                with tab2:
                    if fmt_result["issues"]:
                        for issue in fmt_result["issues"]:
                            if issue["type"] == "critical":
                                st.error(issue["message"])
                            elif issue["type"] == "high":
                                st.error(issue["message"])
                            else:
                                st.warning(issue["message"])
                    else:
                        st.success("Formatting looks good!")
                        
                with tab3:
                    st.json(parsed_data)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                import traceback
                st.text(traceback.format_exc())
