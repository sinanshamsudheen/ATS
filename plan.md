# Strategic 3-Phase Implementation Plan: ATS Resume Compliance Checker

## Executive Summary
This document outlines a consolidated, high-impact roadmap to deliver the ATS Resume Compliance Checker. Condensed from the initial 5-phase proposal into 3 strategic phases, this plan prioritizes "Time to Value" by establishing a working deterministic baseline before layering in complex generative AI capabilities.

---

## Phase 1: Core Architecture & Deterministic Analysis Engine
**Goal:** Deliver a functional MVP that parses resumes, analyzes formatting, and calculates a baseline "Content Match" score using semantic similarity (Embeddings), without the heavy LLM dependency.

### Key Objectives
*   **Infrastructure Setup:** Initialize repository, environment, and Streamlit automated CI/CD pipeline.
*   **Data Ingestion:** Robust PDF text extraction (PyMuPDF) with fallback and section segmentation (Regex/Heuristic).
*   **Vector Analysis:** Implement `sentence-transformers/all-MiniLM-L6-v2` to compute cosine similarity between Resume and JD.
*   **Rule-Based Checks:** Implement the `Formatting Checker Module` (Tables, Images, Header/Footer verification).
*   **MVP Interface:** Streamlit dashboard to upload files, view extracted text, and see preliminary scores (Keyword Match, Formatting).

### Technical Deliverables
*   `src/preprocessing/pdf_extractor.py`: Multi-page handling + text cleaning.
*   `src/analysis/embeddings.py`: Vector generation and caching.
*   `src/analysis/similarity.py`: Keyword coverage logic (missing keywords detection).
*   `src/app/`: Basic Streamlit UI showing "Similarity Score" and "Missing Keywords".

**Success Criteria:**
*   [x] System accepts PDF and JD text.
*   [x] Returns specific "Missing Keywords" with >90% accuracy against ground truth.
*   [x] Identifies formatting errors (e.g., tables) correctly.
*   [x] End-to-end latency < 5 seconds (CPU inference).

---

## Phase 2: Generative AI Intelligence & Model Fine-Tuning
**Goal:** Transform the tool from a "checker" to an "optimizer" by integrating the fine-tuned LLM for specific, actionable rewrite suggestions.

### Key Objectives
*   **Data Factory:** Synthesize and curate the training dataset (Resume-JD pairs with expert annotations).
*   **Model Engineering:** Fine-tune `Phi-3-mini` (or `Llama-3.2-3B`) using LoRA/QLoRA for the specific task of "Critique & Rewrite".
*   **Inference Pipeline:** Optimize model loading and inference for the Streamlit environment (quantization).
*   **Feedback & Rewrite:** Implement the logic to suggest specific improvements for weak bullet points.

### Technical Deliverables
*   `data/processed/`: Prepared dataset with "weak vs. strong" bullet point pairs.
*   `models/fine-tuned/`: LoRA adapters for the selected LLM.
*   `src/model/llm_inference.py`: Helper class for efficient local/cloud inference.
*   `src/app/components/results_display.py`: Updated UI to show "Original vs. Improved" side-by-side.

**Success Criteria:**
*   [x] Model generates actionable rewrites (not just generic advice).
*   [x] Inference time remains acceptable (e.g., < 30s on T4 GPU).
*   [x] Score accuracy (MAE) improved by aligning LLM judgment with ATS rules.

---

## Phase 3: Production Polish, Optimization & Deployment
**Goal:** Harden the application for real-world usage, addressing edge cases, UI/UX refinement, and extensive testing.

### Key Objectives
*   **UX/UI Overhaul:** Implement the "Premium" design aesthetic – responsive layouts, progress bars, and clear visual hierarchy (Green/Yellow/Red indicators).
*   **Performance Optimization:** Asynchronous processing for the heavy LLM tasks; caching mechanism for repeated analyses.
*   **Comprehensive Testing:** Unit tests for parsers, integration tests for the full pipeline, and manual edge-case verification (bad PDFs, empty JDs).
*   **Documentation & Handover:** Complete API documentation, user guide, and final code cleanup.

### Technical Deliverables
*   **Final Dashboard:** Polished Streamlit app with export functionality (PDF Report).
*   **Optimized Pipeline:** Async task queue (optional) or optimized sequential flow.
*   `tests/`: Full test suite coverage (>80%).
*   `README.md` & `docs/`: Comprehensive deployment instructions.

**Success Criteria:**
*   [x] Zero critical crashes on malformed inputs.
*   [x] UI is intuitive (no training required for users).
*   [x] Full analysis pipeline completes < 45 seconds total.
*   [x] Deployment ready (Dockerized or Streamlit Cloud configuration).
