"""
Deterministic keyword-based skill extraction and overlap scoring.

Used to:
- Ground-truth validate matched_skills / missing_skills in training data
- Post-validate LLM-generated ATS reports
"""

import re
from typing import Set, Dict, Any

# ---------------------------------------------------------------------------
# Skill vocabulary — covers cloud, devops, languages, frameworks, data science,
# security, databases, and soft-skill-adjacent tools (~300 terms).
# Multi-word skills come first so they are matched before sub-tokens.
# ---------------------------------------------------------------------------
_MULTI_WORD_SKILLS = {
    # Cloud & infra
    "amazon web services", "google cloud platform", "azure devops",
    "infrastructure as code", "continuous integration", "continuous deployment",
    "ci/cd", "ci cd", "site reliability engineering", "machine learning",
    "deep learning", "natural language processing", "computer vision",
    "data science", "data engineering", "data analysis", "data analytics",
    "big data", "business intelligence", "a/b testing", "product analytics",
    "version control", "agile methodology", "scrum master",
    "object oriented programming", "test driven development",
    "microservices architecture", "event driven architecture",
    "rest api", "graphql api", "sql server", "google bigquery",
    "apache kafka", "apache spark", "apache airflow", "apache flink",
    "elastic search", "power bi", "tableau desktop",
    "iso 27001", "soc 2", "zero trust", "devsecops", "cloud security",
    "penetration testing", "vulnerability assessment",
    "incident response", "threat modeling",
    "financial modeling", "risk management", "project management",
    "product management", "product roadmap",
}

_SINGLE_WORD_SKILLS = {
    # Programming languages
    "python", "java", "javascript", "typescript", "golang", "go", "rust",
    "c++", "c#", "ruby", "scala", "kotlin", "swift", "r", "matlab",
    "bash", "shell", "powershell", "perl", "php",
    # Web frameworks & libraries
    "react", "angular", "vue", "nextjs", "django", "flask", "fastapi",
    "spring", "express", "rails", "laravel", "graphql", "grpc",
    # Cloud platforms & services
    "aws", "azure", "gcp", "heroku", "digitalocean", "cloudflare",
    "ec2", "s3", "lambda", "eks", "ecs", "rds", "dynamodb", "sqs",
    "sns", "cloudwatch", "cloudformation", "route53", "vpc", "iam",
    # DevOps / infra tools
    "docker", "kubernetes", "helm", "terraform", "ansible", "puppet",
    "chef", "vagrant", "packer", "vault", "consul",
    "jenkins", "github", "gitlab", "bitbucket", "argocd", "flux",
    "prometheus", "grafana", "datadog", "splunk", "newrelic", "pagerduty",
    "nginx", "apache", "istio", "linkerd",
    # Databases
    "sql", "postgresql", "postgres", "mysql", "sqlite", "mongodb",
    "redis", "cassandra", "elasticsearch", "neo4j", "influxdb",
    "snowflake", "redshift", "bigquery", "hive", "presto", "dbt",
    # Data / ML
    "pandas", "numpy", "scipy", "sklearn", "scikit-learn", "tensorflow",
    "pytorch", "keras", "xgboost", "lightgbm", "spark", "hadoop",
    "airflow", "kafka", "flink", "dask", "ray", "mlflow", "kubeflow",
    "tableau", "looker", "metabase", "superset", "dbt",
    # Security
    "siem", "soar", "soc", "ids", "ips", "waf", "vpn", "pki", "ssl",
    "tls", "oauth", "jwt", "ldap", "saml", "qradar", "splunk",
    "nessus", "burpsuite", "metasploit", "wireshark", "nmap",
    # Mobile
    "android", "ios", "flutter", "reactnative", "xcode",
    # General tools & practices
    "git", "jira", "confluence", "slack", "figma", "postman",
    "linux", "unix", "macos", "windows",
    "agile", "scrum", "kanban", "devops", "sre", "finops",
    "microservices", "serverless", "monolith",
    # Soft / process skills that appear as hard keywords in JDs
    "leadership", "mentoring", "communication",
}

# Merge into one master set
TECH_SKILLS: Set[str] = _MULTI_WORD_SKILLS | _SINGLE_WORD_SKILLS


def extract_skills(text: str) -> Set[str]:
    """
    Return the subset of TECH_SKILLS that appear in *text*.

    Multi-word skills are matched first (longest-match priority).
    Single-word matching uses word-boundary regex to avoid false positives
    (e.g. "go" inside "going").
    """
    lower = text.lower()
    found: Set[str] = set()

    # Multi-word first
    for skill in _MULTI_WORD_SKILLS:
        if skill in lower:
            found.add(skill)

    # Single-word — word-boundary match
    for skill in _SINGLE_WORD_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, lower):
            found.add(skill)

    return found


def compute_overlap(resume_text: str, jd_text: str) -> Dict[str, Any]:
    """
    Deterministically compute skill overlap between a resume and a JD.

    Returns:
        {
            "matched": sorted list of skills present in BOTH texts,
            "missing": sorted list of JD skills absent from the resume,
            "keyword_coverage": int 0-100  (matched / jd_skills * 100)
        }
    """
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    matched = resume_skills & jd_skills
    missing = jd_skills - resume_skills

    if jd_skills:
        coverage = int(round(len(matched) / len(jd_skills) * 100))
    else:
        coverage = 0

    return {
        "matched": sorted(matched),
        "missing": sorted(missing),
        "keyword_coverage": min(coverage, 100),
    }
