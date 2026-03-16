import streamlit as st
import sys
import os
import tempfile
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.parsing.pdf_extractor import extract_text_from_pdf
from src.parsing.resume_parser import ResumeParser
from src.app.components.results_display import display_llm_ats_report
from src.config import APP_TITLE, APP_VERSION, GGUF_MODEL_PATH, BASE_MODEL_NAME, LORA_ADAPTER_PATH
from src.inference.groq_inference import generate_with_groq, is_groq_available


@st.cache_resource(show_spinner=False)
def _load_local_inference():
    """Load the best available local model.

    Returns (generate_fn, model_label) or (None, None).
    Priority: GGUF (fast CPU) → Groq.
    """
    if Path(GGUF_MODEL_PATH).exists():
        try:
            from src.inference.gguf_inference import load_model, generate
            model = load_model(GGUF_MODEL_PATH)
            def _gen(resume, jd, _m=model):
                return generate(_m, resume, jd)
            return _gen, "Phi-3 GGUF Q8_0"
        except Exception:
            pass

    if is_groq_available():
        def _groq_gen(resume, jd):
            return generate_with_groq(resume, jd)
        return _groq_gen, "Groq (Llama 3.1 8B)"

    return None, None


# Page Config
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📄")

# Custom CSS
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
    <p style="margin: 0; font-size: 16px;">v{APP_VERSION} - AI-Powered ATS Report</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🚀 How It Works")
    st.info("""
    1. 📤 Upload your Resume (PDF)
    2. 📋 Paste the Job Description
    3. 🔍 Click "Analyze"
    4. ✨ Get your ATS report
    """)

    st.divider()

    _local_gen, _local_label = _load_local_inference()
    if _local_gen is not None:
        st.success(f"✅ Local model: {_local_label}")
    else:
        st.warning("⚠️ Local model: Unavailable")

    if is_groq_available():
        st.success("✅ Groq API: Connected")
    else:
        st.warning("⚠️ Groq API: Not configured")

    if _local_gen is not None:
        st.caption(f"🤖 Inference: {_local_label} (Groq fallback)")
    elif is_groq_available():
        st.caption("🤖 Inference: Groq + Llama 3.1 8B")
    else:
        st.caption("🤖 Inference: Disabled (no model available)")

# Main Interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ Upload Resume")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf",
                                     key="resume_upload",
                                     help="Max 5MB, ATS-friendly format recommended")

with col2:
    st.subheader("2️⃣ Job Description")
    job_description = st.text_area("Paste JD text here", height=200,
                                   key="job_description",
                                   placeholder="Paste the complete job description including requirements, responsibilities, and qualifications...")

# Analyze Button
st.markdown("---")
col_analyze = st.columns([1, 2, 1])[1]

with col_analyze:
    analyze_button = st.button("🚀 Analyze Resume", type="primary")

if analyze_button:
    if not uploaded_file:
        st.error("❌ Please upload a resume first.")
    elif not job_description:
        st.error("❌ Please provide a Job Description.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Step 1: Extract text
            status_text.info("📄 Extracting text from PDF...")
            progress_bar.progress(20)
            text = extract_text_from_pdf(tmp_path)

            # Step 2: Parse resume
            status_text.info("🔍 Parsing resume structure...")
            progress_bar.progress(40)
            parser = ResumeParser()
            parsed_data = parser.parse(text)

            # Step 3: LLM inference — try local model first, fall back to Groq
            llm_report = None
            progress_bar.progress(60)
            _local_gen, _local_label = _load_local_inference()

            if _local_gen is not None:
                status_text.info(f"🤖 Running {_local_label} inference...")
                try:
                    llm_report = _local_gen(text, job_description)
                    progress_bar.progress(90)
                except Exception as _local_err:
                    st.warning(f"⚠️ Local model failed: {_local_err} — falling back to Groq")

            # Also fall back to Groq if local model returned invalid/unparseable JSON
            if llm_report is not None and not llm_report.get("valid_json"):
                st.warning(f"⚠️ Local model output invalid — falling back to Groq")
                llm_report = None

            if llm_report is None:
                if is_groq_available():
                    status_text.info("🤖 Running Groq + Llama 3.1 8B inference...")
                    try:
                        llm_report = generate_with_groq(text, job_description)
                        progress_bar.progress(90)
                    except Exception as _llm_err:
                        st.warning(f"⚠️ Groq inference failed: {_llm_err}")
                else:
                    st.warning("⚠️ No inference backend available. Set GROQ_API_KEY or run merge_and_convert.py.")

            os.unlink(tmp_path)
            progress_bar.progress(100)
            status_text.success("✅ Analysis complete!")

            st.markdown("---")

            if llm_report:
                tabs = st.tabs(["🤖 AI Report", "📊 Raw Data"])
                with tabs[0]:
                    display_llm_ats_report(llm_report)
                with tabs[1]:
                    st.subheader("Parsed Resume Data")
                    st.json(parsed_data)
            else:
                st.error("❌ No report generated. Check that a model is available.")
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
    <p>Built with ❤️ using Streamlit, Phi-3, and Groq</p>
</div>
""", unsafe_allow_html=True)
