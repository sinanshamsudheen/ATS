"""Thin entry-point: run ATS inference with fine-tuned Phi + LoRA."""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.generate_report import load_config, load_model, generate

_SAMPLE_RESUME = """John Smith
Software Engineer | john.smith@email.com | (555) 123-4567

EXPERIENCE
Software Engineer, TechCorp Inc. - Jan 2021 - Present
- Developed RESTful APIs using Python and Flask serving 10K daily users
- Managed PostgreSQL databases with 50+ tables and optimized query performance
- Collaborated with cross-functional teams to deliver 3 major product releases
- Responsible for maintaining CI/CD pipelines using Jenkins and Docker

Junior Developer, StartupXYZ - Jun 2019 - Dec 2020
- Built frontend components using React and TypeScript
- Helped with bug fixes and code reviews

EDUCATION
B.S. Computer Science, State University - 2019

SKILLS
Python, JavaScript, React, Flask, PostgreSQL, Docker, Git, AWS"""

_SAMPLE_JD = """Senior Software Engineer - TechGlobal Inc.
Requirements:
- 5+ years of experience in software development
- Strong proficiency in Python and Java
- Experience with RESTful API design and microservices architecture
- Proficiency with SQL databases (PostgreSQL preferred)
- Experience with cloud services (AWS or GCP)
- Familiarity with Docker and Kubernetes
- Strong understanding of CI/CD pipelines
Responsibilities:
- Design and implement scalable backend services
- Write clean, maintainable, and well-tested code
- Mentor junior engineers"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ATS inference")
    parser.add_argument("--adapter-path", default="models/ats_phi_lora")
    parser.add_argument("--resume", default=None, help="Path to resume text file")
    parser.add_argument("--job-desc", default=None, help="Path to JD text file")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    config = load_config()
    model_name = config["model_name"]

    model, tokenizer = load_model(
        base_model_name=model_name,
        adapter_path=args.adapter_path,
        use_4bit=not args.no_4bit,
    )

    if args.resume and args.job_desc:
        with open(args.resume, "r", encoding="utf-8") as f:
            resume_text = f.read()
        with open(args.job_desc, "r", encoding="utf-8") as f:
            job_description = f.read()
    else:
        resume_text = _SAMPLE_RESUME
        job_description = _SAMPLE_JD

    result = generate(
        model=model,
        tokenizer=tokenizer,
        resume_text=resume_text,
        job_description=job_description,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print(json.dumps(result, indent=2))
    if result.get("valid_json"):
        print(f"\nATS Score: {result['ats_score']}/100")
        print(f"Matched Skills: {', '.join(result['matched_skills'])}")
        print(f"Missing Skills: {', '.join(result['missing_skills'])}")
