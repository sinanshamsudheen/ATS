import streamlit as st
import sys
import os
import tempfile
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.parsing.pdf_extractor import extract_text_from_pdf
from src.parsing.resume_parser import ResumeParser
from src.parsing.jd_parser import extract_keywords
from src.scoring.similarity_engine import SimilarityAnalyzer
from src.scoring.formatting_check import FormattingChecker
from src.rewriting.bullet_rewriter import BulletRewriter
from src.app.components.results_display import (
    display_metrics_overview,
    display_rewrite_summary,
    display_missing_keywords,
    display_formatting_issues,
    display_loading_progress
)
from src.config import APP_TITLE, APP_VERSION

# Page Config
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📄")

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 12px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown(f"""
<div class="main-header">
    <h1>📄 {APP_TITLE}</h1>
    <p style="margin: 0; font-size: 16px;">v{APP_VERSION} - AI-Powered Resume Optimizer</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🚀 How It Works")
    st.info("""
    1. 📤 Upload your Resume (PDF)
    2. 📋 Paste the Job Description  
    3. 🔍 Click "Analyze & Optimize"
    4. ✨ Get AI-powered rewrite suggestions
    """)
    
    st.divider()
    
    # Advanced Options
    with st.expander("⚙️ Advanced Settings"):
        enable_llm = st.checkbox("Enable AI Rewrites", value=True, 
                                 help="Generate LLM-powered bullet point improvements")
        only_weak_bullets = st.checkbox("Only Rewrite Weak Bullets", value=True,
                                       help="Skip bullets that are already strong")
    
    st.divider()
    st.caption("🤖 Powered by Phi-3-mini & Sentence Transformers")
    st.caption("⚡ Local Phi-3 LoRA Fine-Tuned")

# Main Interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ Upload Resume")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", 
                                     help="Max 5MB, ATS-friendly format recommended")
    
with col2:
    st.subheader("2️⃣ Job Description")
    job_description = st.text_area("Paste JD text here", height=200, 
                                   placeholder="Paste the complete job description including requirements, responsibilities, and qualifications...")

# Analyze Button
st.markdown("---")
col_analyze = st.columns([1, 2, 1])[1]  # Center the button

with col_analyze:
    analyze_button = st.button("🚀 Analyze & Optimize Resume", type="primary")

if analyze_button:
    if not uploaded_file:
        st.error("❌ Please upload a resume first.")
    elif not job_description:
        st.error("❌ Please provide a Job Description.")
    else:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Step 1: Extract text
            status_text.info("📄 Extracting text from PDF...")
            progress_bar.progress(10)
            text = extract_text_from_pdf(tmp_path)
            
            # Step 2: Parse resume
            status_text.info("🔍 Parsing resume structure...")
            progress_bar.progress(20)
            parser = ResumeParser()
            parsed_data = parser.parse(text)
            
            # Step 3: Formatting check
            status_text.info("📋 Checking ATS formatting...")
            progress_bar.progress(30)
            fmt_checker = FormattingChecker()
            fmt_result = fmt_checker.check(tmp_path)
            
            # Step 4: Similarity analysis
            status_text.info("🎯 Analyzing keyword match...")
            progress_bar.progress(45)
            analyzer = SimilarityAnalyzer()
            jd_keywords = extract_keywords(job_description)
            keyword_analysis = analyzer.analyze_keywords(text, jd_keywords)
            overall_match = analyzer.calculate_overall_match(text, job_description)
            
            # Step 5: LLM-powered rewrite suggestions (if enabled)
            rewrite_results = None
            if enable_llm:
                status_text.info("🤖 Loading AI model for rewrite suggestions... (This may take a moment)")
                progress_bar.progress(60)
                
                rewrite_engine = BulletRewriter()
                
                status_text.info("✨ Generating AI-powered improvements...")
                progress_bar.progress(75)
                
                rewrite_results = rewrite_engine.analyze_resume_sections(
                    parsed_data,
                    job_description
                )
            
            # Cleanup
            os.unlink(tmp_path)
            
            # Complete
            progress_bar.progress(100)
            status_text.success("✅ Analysis complete!")
            
            # --- Results Display ---
            st.markdown("---")
            
            # Calculate bullet quality score if LLM was used
            bullet_quality_score = 0
            if rewrite_results:
                bullet_quality_score = rewrite_results.get("overall_stats", {}).get("average_score", 0)
            else:
                # Fallback: use simple heuristic based on other scores
                bullet_quality_score = (overall_match + keyword_analysis['score']) / 2
            
            # Display overview metrics
            display_metrics_overview(
                overall_match=overall_match,
                keyword_score=keyword_analysis['score'],
                formatting_score=fmt_result['score'],
                bullet_quality_score=bullet_quality_score
            )
            
            st.markdown("---")
            
            # Tabbed interface for detailed results
            if enable_llm and rewrite_results:
                tabs = st.tabs(["🎯 AI Rewrites", "🔑 Keywords", "📋 Formatting", "📊 Raw Data"])
                
                with tabs[0]:
                    display_rewrite_summary(rewrite_results)
                
                with tabs[1]:
                    display_missing_keywords(
                        keyword_analysis["missing"],
                        keyword_analysis["matched"]
                    )
                
                with tabs[2]:
                    display_formatting_issues(fmt_result)
                
                with tabs[3]:
                    st.subheader("Parsed Resume Data")
                    st.json(parsed_data)
            
            else:
                # Simplified view without LLM
                tabs = st.tabs(["🔑 Keywords", "📋 Formatting", "📊 Resume Content"])
                
                with tabs[0]:
                    display_missing_keywords(
                        keyword_analysis["missing"],
                        keyword_analysis["matched"]
                    )
                
                with tabs[1]:
                    display_formatting_issues(fmt_result)
                
                with tabs[2]:
                    st.subheader("Parsed Resume Data")
                    st.json(parsed_data)

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            
            with st.expander("🐛 Debug Information"):
                import traceback
                st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 20px;">
    <p>Built with ❤️ using Streamlit, Phi-3-mini, and Sentence Transformers</p>
    <p style="font-size: 12px;">Powered by Phi-3-mini LoRA fine-tuning & Sentence Transformers</p>
</div>
""", unsafe_allow_html=True)
