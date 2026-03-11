"""
Results Display Component
Enhanced UI components for showing ATS analysis results with LLM suggestions
"""

import streamlit as st
from typing import Dict, Any, List


def display_metrics_overview(
    overall_match: float,
    keyword_score: float,
    formatting_score: float,
    bullet_quality_score: float
):
    """
    Display key metrics in a card-style layout.
    
    Args:
        overall_match: Overall semantic match percentage
        keyword_score: Keyword coverage percentage
        formatting_score: Formatting compliance score
        bullet_quality_score: Average bullet point quality score
    """
    st.markdown("### 📊 Overall Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        _metric_card("Overall Match", overall_match, "%")
    
    with col2:
        _metric_card("Keyword Coverage", keyword_score, "%")
    
    with col3:
        _metric_card("Format Score", formatting_score, "/100")
    
    with col4:
        _metric_card("Bullet Quality", bullet_quality_score, "/100")


def _metric_card(label: str, value: float, suffix: str = ""):
    """Helper to create a styled metric card."""
    color = _get_score_color(value)
    
    st.markdown(f"""
    <div style="
        background-color: {color}15;
        border-left: 4px solid {color};
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    ">
        <p style="margin:0; font-size:12px; color:gray;">{label}</p>
        <p style="margin:0; font-size:28px; font-weight:bold; color:{color};">
            {value}{suffix}
        </p>
    </div>
    """, unsafe_allow_html=True)


def _get_score_color(score: float) -> str:
    """Return color based on score value."""
    if score >= 80:
        return "#28a745"  # Green
    elif score >= 60:
        return "#ffc107"  # Yellow
    else:
        return "#dc3545"  # Red


def display_bullet_comparison(result: Dict[str, Any], index: int, section_name: str = ""):
    """
    Display original vs. improved bullet point comparison.
    
    Args:
        result: Dictionary with bullet analysis and suggestions
        index: Bullet index for unique keys
        section_name: Section identifier for unique keys across sections
    """
    original = result.get("bullet", "")
    improved = result.get("improved", original)
    score = result.get("score", 0)
    issues = result.get("issues", [])
    llm_analysis = result.get("llm_analysis", "")
    
    # Color based on score
    status_color = _get_score_color(score)
    
    with st.container():
        st.markdown(f"""
        <div style="
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #f8f9fa;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0;">Bullet Point #{index + 1}</h4>
                <span style="
                    background-color: {status_color};
                    color: white;
                    padding: 5px 12px;
                    border-radius: 15px;
                    font-size: 12px;
                    font-weight: bold;
                ">Score: {score}/100</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📝 Original**")
            st.text_area(
                "Original",
                value=original,
                height=100,
                key=f"orig_{section_name}_{index}",
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown("**✨ Improved**")
            st.text_area(
                "Improved",
                value=improved,
                height=100,
                key=f"improved_{section_name}_{index}",
                label_visibility="collapsed"
            )
            
            # Copy button
            if st.button("📋 Copy Improved", key=f"copy_{section_name}_{index}"):
                st.code(improved, language=None)
        
        # Show issues and analysis
        if issues or llm_analysis:
            with st.expander("🔍 Detailed Analysis"):
                if issues:
                    st.markdown("**Issues Found:**")
                    for issue in issues:
                        st.markdown(f"- ⚠️ {issue}")
                
                if llm_analysis:
                    st.markdown("**AI Feedback:**")
                    st.info(llm_analysis)


def display_section_results(section_name: str, section_data: Dict[str, Any]):
    """
    Display results for a specific resume section.
    
    Args:
        section_name: Name of the section (e.g., "experience")
        section_data: Dictionary with bullets and stats
    """
    bullets = section_data.get("bullets", [])
    stats = section_data.get("stats", {})
    
    if not bullets:
        return
    
    st.markdown(f"### 📌 {section_name.title()} Section")
    
    # Section stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Bullets", stats.get("total_bullets", 0))
    
    with col2:
        st.metric("Need Improvement", stats.get("weak_bullets", 0))
    
    with col3:
        st.metric("Avg Quality", f"{stats.get('average_score', 0)}/100")
    
    st.markdown("---")
    
    # Display each bullet
    for idx, result in enumerate(bullets):
        display_bullet_comparison(result, idx, section_name)


def display_missing_keywords(missing: List[str], matched: List[str]):
    """
    Display missing and matched keywords.
    
    Args:
        missing: List of missing keywords
        matched: List of matched keywords
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ❌ Missing Keywords")
        if missing:
            for kw in missing:
                st.markdown(f"- `{kw}`")
        else:
            st.success("No critical keywords missing!")
    
    with col2:
        st.markdown("### ✅ Matched Keywords")
        if matched:
            for kw in matched[:10]:  # Show first 10
                st.markdown(f"- `{kw}`")
            if len(matched) > 10:
                st.caption(f"... and {len(matched) - 10} more")
        else:
            st.info("No keywords matched yet")


def display_formatting_issues(fmt_result: Dict[str, Any]):
    """
    Display formatting issues in a structured way.
    
    Args:
        fmt_result: Formatting check results
    """
    st.markdown("### 📋 Formatting Analysis")
    
    score = fmt_result.get("score", 0)
    issues = fmt_result.get("issues", [])
    
    # Score display
    _metric_card("Formatting Compliance", score, "/100")
    
    # Issues list
    if issues:
        st.markdown("**Issues Found:**")
        for issue in issues:
            issue_type = issue.get("type", "info")
            message = issue.get("message", "")
            
            if issue_type == "critical":
                st.error(f"🚫 {message}")
            elif issue_type == "high":
                st.error(f"⚠️ {message}")
            elif issue_type == "medium":
                st.warning(f"⚡ {message}")
            else:
                st.info(f"ℹ️ {message}")
    else:
        st.success("✅ No formatting issues detected!")


def display_rewrite_summary(rewrite_results: Dict[str, Any]):
    """
    Display a summary of all rewrite suggestions.
    
    Args:
        rewrite_results: Complete rewrite analysis results
    """
    st.markdown("## 🎯 Resume Optimization Results")
    
    overall_stats = rewrite_results.get("overall_stats", {})
    
    # Overall stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Bullets Analyzed",
            overall_stats.get("total_bullets", 0)
        )
    
    with col2:
        st.metric(
            "Bullets Needing Work",
            overall_stats.get("weak_bullets", 0),
            delta=f"-{overall_stats.get('improvement_rate', 0)}%"
        )
    
    with col3:
        st.metric(
            "Average Quality Score",
            f"{overall_stats.get('average_score', 0)}/100"
        )
    
    with col4:
        improvement_rate = overall_stats.get("improvement_rate", 0)
        status = "🟢 Excellent" if improvement_rate < 20 else "🟡 Good" if improvement_rate < 40 else "🔴 Needs Work"
        st.metric(
            "Resume Health",
            status
        )
    
    st.markdown("---")
    
    # Section-by-section results
    sections = rewrite_results.get("sections", {})
    
    for section_name, section_data in sections.items():
        if section_data.get("bullets"):
            with st.expander(f"📂 {section_name.title()} Section", expanded=True):
                display_section_results(section_name, section_data)


def display_loading_progress(stage: str):
    """
    Display a loading progress indicator.
    
    Args:
        stage: Current loading stage description
    """
    st.info(f"⏳ {stage}...")


def display_llm_ats_report(report: Dict[str, Any]):
    """
    Display the full structured ATS evaluation produced by the fine-tuned Phi-3 LoRA model.

    Args:
        report: Dict returned by src.inference.generate_report.generate()
    """
    if not report.get("valid_json"):
        st.warning("⚠️ Model output could not be parsed as structured JSON.")
        raw = report.get("raw_output", "")
        if raw:
            st.markdown("**Raw Model Output:**")
            st.code(raw[:2000], language=None)
        return

    st.markdown("## 🤖 Fine-Tuned Model ATS Report")

    # ── Score cards ──────────────────────────────────────────────────────────
    ats_score = report.get("ats_score", 0)
    breakdown = report.get("score_breakdown", {})

    cols = st.columns(5)
    labels = [
        ("ATS Score",        ats_score,                               "/100"),
        ("Keyword Coverage", breakdown.get("keyword_coverage", 0),   "/100"),
        ("Bullet Quality",   breakdown.get("bullet_quality",   0),   "/100"),
        ("Formatting",       breakdown.get("formatting",       0),   "/100"),
        ("Structure",        breakdown.get("structure",        0),   "/100"),
    ]
    for col, (label, value, suffix) in zip(cols, labels):
        with col:
            _metric_card(label, value, suffix)

    st.markdown("---")

    # ── Matched / Missing Skills ──────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ✅ Matched Skills")
        matched = report.get("matched_skills", [])
        if matched:
            for skill in matched:
                st.markdown(f"- `{skill}`")
        else:
            st.info("No matched skills reported.")

    with col2:
        st.markdown("### ❌ Missing Skills")
        missing = report.get("missing_skills", [])
        if missing:
            for skill in missing:
                st.markdown(f"- `{skill}`")
        else:
            st.success("No critical skills missing!")

    st.markdown("---")

    # ── Weak bullets with AI improvements ────────────────────────────────────
    weak_bullets = report.get("weak_bullets", [])
    if weak_bullets:
        st.markdown("### 📝 Weak Bullets & AI Improvements")
        for i, item in enumerate(weak_bullets):
            if isinstance(item, dict):
                original = item.get("original", item.get("bullet", ""))
                improved = item.get("improved", item.get("rewrite", original))
                reason   = item.get("reason",   item.get("feedback", ""))
            else:
                original = str(item)
                improved = original
                reason   = ""

            preview = (original[:70] + "…") if len(original) > 70 else original
            with st.expander(f"Bullet #{i + 1}: {preview}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**📝 Original**")
                    st.text_area("orig", value=original, height=90,
                                 key=f"llm_orig_{i}", label_visibility="collapsed")
                with c2:
                    st.markdown("**✨ AI Improved**")
                    st.text_area("impr", value=improved, height=90,
                                 key=f"llm_impr_{i}", label_visibility="collapsed")
                    if st.button("📋 Copy", key=f"llm_copy_{i}"):
                        st.code(improved, language=None)
                if reason:
                    st.info(f"💡 {reason}")

    # ── AI-detected formatting issues ────────────────────────────────────────
    ai_fmt = report.get("formatting_issues", [])
    if ai_fmt:
        st.markdown("---")
        st.markdown("### 📋 AI-Detected Formatting Issues")
        for issue in ai_fmt:
            if isinstance(issue, dict):
                msg = issue.get("message", issue.get("issue", str(issue)))
            else:
                msg = str(issue)
            st.warning(f"⚡ {msg}")

    # ── Overall feedback ─────────────────────────────────────────────────────
    feedback = report.get("overall_feedback", "")
    if feedback:
        st.markdown("---")
        st.markdown("### 💬 Overall AI Feedback")
        st.success(feedback)
