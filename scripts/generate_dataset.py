"""Generate synthetic ATS evaluation dataset for fine-tuning."""
import json
import random
import os

random.seed(42)

resumes = [
    {
        "name": "Software Engineer",
        "text": "John Smith\nSoftware Engineer | john.smith@email.com | (555) 123-4567\n\nEXPERIENCE\nSoftware Engineer, TechCorp Inc. - Jan 2021 - Present\n- Developed RESTful APIs using Python and Flask serving 10K daily users\n- Managed PostgreSQL databases with 50+ tables and optimized query performance\n- Collaborated with cross-functional teams to deliver 3 major product releases\n- Responsible for maintaining CI/CD pipelines using Jenkins and Docker\n\nJunior Developer, StartupXYZ - Jun 2019 - Dec 2020\n- Built frontend components using React and TypeScript\n- Helped with bug fixes and code reviews\n- Participated in daily standup meetings and sprint planning\n- Worked on improving test coverage from 40% to 75%\n\nEDUCATION\nB.S. Computer Science, State University - 2019\nGPA: 3.5/4.0\n\nSKILLS\nPython, JavaScript, React, Flask, PostgreSQL, Docker, Git, AWS"
    },
    {
        "name": "Data Scientist",
        "text": "Sarah Chen\nData Scientist | sarah.chen@email.com | LinkedIn: /in/sarachen\n\nSUMMARY\nData scientist with 4 years of experience in machine learning and statistical analysis.\n\nEXPERIENCE\nSenior Data Scientist, DataDriven Corp - Mar 2022 - Present\n- Built machine learning models achieving 92% accuracy on customer churn prediction\n- Designed A/B testing framework that increased conversion rates by 15%\n- Processed and analyzed datasets containing 10M+ records using PySpark\n- Presented insights to C-suite executives resulting in $2M cost savings\n\nData Analyst, Analytics Co - Aug 2020 - Feb 2022\n- Created dashboards using Tableau for real-time business monitoring\n- Responsible for data cleaning and preprocessing tasks\n- Assisted in building predictive models using scikit-learn\n- Generated weekly reports for stakeholders\n\nEDUCATION\nM.S. Statistics, Top University - 2020\nB.S. Mathematics, State College - 2018\n\nSKILLS\nPython, R, SQL, TensorFlow, PyTorch, Tableau, PySpark, scikit-learn, pandas"
    },
    {
        "name": "DevOps Engineer",
        "text": "Michael Rodriguez\nDevOps Engineer | m.rodriguez@email.com | GitHub: github.com/mrodriguez\n\nEXPERIENCE\nSenior DevOps Engineer, CloudFirst Solutions - May 2021 - Present\n- Architected multi-region AWS infrastructure supporting 99.99% uptime for 5M users\n- Reduced deployment time by 70% through implementing GitOps with ArgoCD\n- Managed Kubernetes clusters handling 500+ microservices across 3 environments\n- Implemented infrastructure as code using Terraform reducing provisioning time by 80%\n\nDevOps Engineer, WebScale Inc - Jan 2019 - Apr 2021\n- Maintained Linux servers and managed deployments\n- Set up monitoring using Prometheus and Grafana\n- Worked on containerization of legacy applications\n- Helped with incident response and troubleshooting\n\nEDUCATION\nB.S. Information Technology, Tech University - 2018\n\nSKILLS\nAWS, Kubernetes, Docker, Terraform, Jenkins, Python, Bash, Prometheus, ArgoCD, Linux"
    },
    {
        "name": "Product Manager",
        "text": "Emily Johnson\nProduct Manager | emily.j@email.com | (555) 987-6543\n\nEXPERIENCE\nSenior Product Manager, InnovateTech - Jun 2020 - Present\n- Led product roadmap for SaaS platform generating $5M ARR\n- Increased user retention by 25% through data-driven feature prioritization\n- Managed a team of 8 engineers and 2 designers across 3 product lines\n- Conducted 50+ user interviews to validate product-market fit\n\nProduct Manager, GrowthStartup - Feb 2018 - May 2020\n- Defined product requirements and wrote detailed user stories\n- Coordinated with engineering team on sprint planning\n- Tracked KPIs and metrics for product performance\n- Involved in customer support escalations\n\nEDUCATION\nMBA, Business School - 2018\nB.A. Economics, Liberal Arts College - 2015\n\nSKILLS\nAgile/Scrum, JIRA, SQL, Product Analytics, A/B Testing, Figma, Roadmapping, Stakeholder Management"
    },
    {
        "name": "Frontend Developer",
        "text": "Alex Kim\nFrontend Developer | alex.kim@email.com | Portfolio: alexkim.dev\n\nEXPERIENCE\nSenior Frontend Developer, PixelPerfect Inc - Apr 2021 - Present\n- Engineered responsive web application serving 2M monthly active users\n- Reduced page load time by 45% through code splitting and lazy loading\n- Built reusable component library used by 15 developers across 4 teams\n- Implemented end-to-end testing with Cypress achieving 90% coverage\n\nFrontend Developer, DigitalAgency - Jul 2019 - Mar 2021\n- Developed websites for various clients using React\n- Responsible for converting Figma designs to pixel-perfect code\n- Worked on fixing cross-browser compatibility issues\n- Participated in code reviews and team meetings\n\nEDUCATION\nB.S. Computer Science, University of Technology - 2019\n\nSKILLS\nJavaScript, TypeScript, React, Next.js, Vue.js, CSS/SASS, Tailwind, Webpack, Cypress, Jest"
    },
    {
        "name": "Backend Developer",
        "text": "David Park\nBackend Engineer | david.park@email.com | (555) 456-7890\n\nEXPERIENCE\nSenior Backend Engineer, ScaleUp Systems - Aug 2020 - Present\n- Designed microservices architecture handling 100K requests per second\n- Optimized database queries reducing average response time from 500ms to 50ms\n- Implemented event-driven system using Kafka processing 1M events daily\n- Mentored 4 junior developers through code reviews and pair programming\n\nBackend Developer, CodeFactory - Mar 2018 - Jul 2020\n- Built REST APIs using Node.js and Express\n- Managed MongoDB databases\n- Helped with debugging production issues\n- Wrote unit tests for critical business logic\n\nEDUCATION\nB.S. Software Engineering, Engineering University - 2018\n\nSKILLS\nJava, Python, Node.js, PostgreSQL, MongoDB, Redis, Kafka, Docker, Kubernetes, GraphQL"
    },
    {
        "name": "ML Engineer",
        "text": "Lisa Wang\nMachine Learning Engineer | lisa.wang@email.com\n\nEXPERIENCE\nML Engineer, AIFirst Labs - Jan 2022 - Present\n- Deployed NLP model achieving 95% F1 score on sentiment analysis reducing manual review by 60%\n- Built end-to-end ML pipeline processing 5TB of data daily using Apache Airflow\n- Reduced model inference latency by 40% through ONNX optimization\n- Published internal research paper on transformer fine-tuning best practices\n\nData Scientist, TechData Inc - May 2020 - Dec 2021\n- Developed classification models using Random Forest and XGBoost\n- Responsible for feature engineering and data preprocessing\n- Created data visualization dashboards\n- Assisted senior engineers with model deployment\n\nEDUCATION\nM.S. Computer Science (ML focus), Research University - 2020\nB.S. Computer Science, State University - 2018\n\nSKILLS\nPython, PyTorch, TensorFlow, Hugging Face, MLflow, Docker, AWS SageMaker, SQL, Apache Airflow"
    },
    {
        "name": "Full Stack Developer",
        "text": "Ryan Martinez\nFull Stack Developer | ryan.m@email.com | GitHub: github.com/ryanm\n\nEXPERIENCE\nFull Stack Developer, WebSolutions Co - Mar 2021 - Present\n- Built e-commerce platform generating $500K monthly revenue with 99.9% uptime\n- Reduced customer onboarding time by 35% through streamlined UX workflows\n- Integrated payment gateway processing 10K+ transactions monthly\n- Led migration from monolithic to microservices architecture\n\nJunior Developer, SmallTech LLC - Jun 2019 - Feb 2021\n- Developed features for internal tools using Angular\n- Worked on database queries and API endpoints\n- Fixed bugs reported by QA team\n- Participated in agile ceremonies\n\nEDUCATION\nB.S. Computer Science, City University - 2019\n\nSKILLS\nJavaScript, TypeScript, React, Angular, Node.js, Python, Django, PostgreSQL, MongoDB, AWS"
    },
    {
        "name": "Cybersecurity Analyst",
        "text": "Jennifer Lee\nCybersecurity Analyst | j.lee@email.com | CISSP Certified\n\nEXPERIENCE\nSenior Security Analyst, SecureNet Corp - Feb 2021 - Present\n- Conducted 200+ vulnerability assessments identifying and remediating critical threats\n- Implemented SIEM solution reducing incident response time by 65%\n- Led security awareness training program reaching 500+ employees\n- Designed zero-trust architecture framework adopted company-wide\n\nSecurity Analyst, CyberShield Inc - Aug 2019 - Jan 2021\n- Monitored security alerts and investigated incidents\n- Responsible for managing firewall rules\n- Helped with compliance audits\n- Performed penetration testing on web applications\n\nEDUCATION\nM.S. Cybersecurity, Security Institute - 2019\nB.S. Information Systems, Tech College - 2017\n\nCERTIFICATIONS\nCISSP, CEH, CompTIA Security+\n\nSKILLS\nSIEM, Penetration Testing, Vulnerability Assessment, Firewall Management, Python, Splunk, Nessus"
    },
    {
        "name": "Cloud Architect",
        "text": "Thomas Brown\nCloud Solutions Architect | t.brown@email.com | AWS Solutions Architect Pro\n\nEXPERIENCE\nCloud Architect, EnterpriseCloud Solutions - Apr 2020 - Present\n- Architected cloud migration strategy saving organization $1.2M annually in infrastructure costs\n- Designed auto-scaling system handling traffic spikes of 300% during peak hours\n- Implemented multi-cloud disaster recovery with RPO of 15 minutes and RTO of 1 hour\n- Led team of 6 engineers in migrating 50+ legacy applications to AWS\n\nCloud Engineer, CloudFirst Inc - Jan 2018 - Mar 2020\n- Managed AWS EC2 instances and S3 buckets\n- Set up networking and VPC configurations\n- Assisted with cloud cost optimization\n- Worked on automating infrastructure provisioning\n\nEDUCATION\nB.S. Computer Engineering, Tech University - 2017\n\nCERTIFICATIONS\nAWS Solutions Architect Professional, AWS DevOps Engineer Professional\n\nSKILLS\nAWS, Azure, GCP, Terraform, CloudFormation, Docker, Kubernetes, Python, Networking, Security"
    }
]

job_descriptions = [
    {
        "title": "Senior Software Engineer",
        "text": "Senior Software Engineer\nTechGlobal Inc.\n\nRequirements:\n- 5+ years of experience in software development\n- Strong proficiency in Python and Java\n- Experience with RESTful API design and microservices architecture\n- Proficiency with SQL databases (PostgreSQL preferred)\n- Experience with cloud services (AWS or GCP)\n- Familiarity with Docker and Kubernetes\n- Strong understanding of CI/CD pipelines\n- Experience with agile development methodologies\n\nNice to have:\n- Experience with message queues (Kafka, RabbitMQ)\n- Knowledge of GraphQL\n- Contributions to open-source projects\n\nResponsibilities:\n- Design and implement scalable backend services\n- Write clean, maintainable, and well-tested code\n- Participate in code reviews and technical design discussions\n- Mentor junior engineers"
    },
    {
        "title": "Data Scientist",
        "text": "Data Scientist\nAnalyticsFirst Corp.\n\nRequirements:\n- 3+ years of experience in data science or machine learning\n- Strong proficiency in Python and SQL\n- Experience with ML frameworks (scikit-learn, TensorFlow, or PyTorch)\n- Strong statistical analysis skills\n- Experience with data visualization tools (Tableau, matplotlib)\n- Knowledge of A/B testing and experimentation\n- Experience with big data technologies (Spark, Hadoop)\n- Strong communication skills\n\nNice to have:\n- Experience with NLP or computer vision\n- Knowledge of cloud ML platforms (SageMaker, Vertex AI)\n\nResponsibilities:\n- Build and deploy machine learning models\n- Analyze large datasets to extract insights\n- Design and analyze experiments\n- Collaborate with product and engineering teams"
    },
    {
        "title": "DevOps Engineer",
        "text": "DevOps Engineer\nCloudScale Technologies\n\nRequirements:\n- 4+ years of DevOps or SRE experience\n- Strong experience with AWS or Azure\n- Proficiency with containerization (Docker, Kubernetes)\n- Experience with Infrastructure as Code (Terraform, CloudFormation)\n- Strong scripting skills (Python, Bash)\n- Experience with CI/CD tools (Jenkins, GitLab CI, GitHub Actions)\n- Knowledge of monitoring and observability tools\n- Understanding of networking and security\n\nNice to have:\n- Experience with service mesh (Istio)\n- GitOps experience (ArgoCD, Flux)\n\nResponsibilities:\n- Build and maintain cloud infrastructure\n- Implement and manage CI/CD pipelines\n- Monitor system health and optimize performance\n- Respond to incidents"
    },
    {
        "title": "Frontend Developer",
        "text": "Senior Frontend Developer\nPixelCraft Studios\n\nRequirements:\n- 4+ years of frontend development experience\n- Expert JavaScript/TypeScript skills\n- Strong experience with React or Vue.js\n- Experience with state management (Redux, Vuex)\n- Proficiency in CSS/SASS and responsive design\n- Experience with testing frameworks (Jest, Cypress)\n- Web performance optimization\n- Experience with build tools (Webpack, Vite)\n\nNice to have:\n- Experience with Next.js or Nuxt.js\n- Knowledge of accessibility standards\n\nResponsibilities:\n- Build responsive and performant web applications\n- Create reusable component libraries\n- Optimize application performance\n- Collaborate with designers and backend engineers"
    },
    {
        "title": "ML Engineer",
        "text": "Machine Learning Engineer\nAIVentures Inc.\n\nRequirements:\n- 3+ years of ML engineering experience\n- Strong Python skills\n- Experience with PyTorch or TensorFlow\n- Experience deploying ML models to production\n- Knowledge of MLOps (MLflow, Kubeflow)\n- Experience with NLP or Computer Vision\n- Cloud platforms (AWS, GCP)\n- Strong software engineering fundamentals\n\nNice to have:\n- Experience with LLMs and transformers\n- Model optimization techniques\n- Distributed training experience\n\nResponsibilities:\n- Design and implement ML pipelines\n- Deploy and monitor models in production\n- Optimize model performance\n- Collaborate with data scientists"
    },
    {
        "title": "Product Manager",
        "text": "Senior Product Manager\nInnovateSoft\n\nRequirements:\n- 5+ years of product management experience\n- SaaS product experience\n- Strong analytical and data-driven approach\n- Agile methodologies\n- Excellent communication and stakeholder management\n- Product analytics tools experience\n- Track record of successful launches\n- User research experience\n\nNice to have:\n- Technical background\n- AI/ML product experience\n- MBA preferred\n\nResponsibilities:\n- Define and execute product roadmap\n- Conduct user research\n- Work with engineering and design\n- Track product KPIs"
    },
    {
        "title": "Full Stack Developer",
        "text": "Full Stack Developer\nWebDynamics Inc.\n\nRequirements:\n- 3+ years of full stack experience\n- JavaScript/TypeScript proficiency\n- React or Angular experience\n- Backend with Node.js or Python (Django/Flask)\n- Database experience (PostgreSQL, MongoDB)\n- RESTful API design\n- Git proficiency\n- Cloud deployment experience\n\nNice to have:\n- GraphQL experience\n- Microservices architecture\n- Payment integrations\n\nResponsibilities:\n- Develop end-to-end features\n- Design and implement APIs\n- Write comprehensive tests\n- Participate in agile processes"
    },
    {
        "title": "Cybersecurity Analyst",
        "text": "Senior Cybersecurity Analyst\nCyberGuard Solutions\n\nRequirements:\n- 4+ years of cybersecurity experience\n- CISSP or equivalent certification\n- SIEM tools experience (Splunk, QRadar)\n- Vulnerability assessment knowledge\n- Penetration testing experience\n- Compliance frameworks (SOC 2, ISO 27001)\n- Incident response experience\n- Cloud security (AWS, Azure)\n\nNice to have:\n- Zero-trust architecture\n- DevSecOps practices\n- Malware analysis\n\nResponsibilities:\n- Conduct vulnerability assessments\n- Monitor and respond to incidents\n- Implement security controls\n- Lead security training"
    },
    {
        "title": "Cloud Architect",
        "text": "Senior Cloud Architect\nEnterpriseTech Solutions\n\nRequirements:\n- 6+ years of cloud engineering\n- AWS Solutions Architect Professional\n- Multi-cloud architecture experience\n- Networking and security knowledge\n- Infrastructure as Code\n- Kubernetes expertise\n- Cost optimization strategies\n- Strong leadership skills\n\nNice to have:\n- Hybrid cloud experience\n- Serverless architectures\n- FinOps certification\n\nResponsibilities:\n- Design cloud architecture strategies\n- Lead migration projects\n- Optimize costs and performance\n- Mentor engineering teams"
    },
    {
        "title": "Backend Engineer",
        "text": "Backend Engineer\nDataFlow Systems\n\nRequirements:\n- 3+ years of backend experience\n- Java, Python, or Go proficiency\n- Microservices architecture\n- SQL and NoSQL databases\n- Message queues (Kafka, RabbitMQ)\n- RESTful API design\n- Docker containerization\n- System design fundamentals\n\nNice to have:\n- gRPC experience\n- Event sourcing/CQRS\n- Distributed systems\n\nResponsibilities:\n- Design and implement backend services\n- Optimize performance\n- Build scalable APIs\n- Collaborate with teams"
    }
]

def generate_score():
    base = random.randint(35, 92)
    keyword = random.randint(15, 45)
    bullet = random.randint(8, 25)
    formatting = random.randint(5, 15)
    structure = random.randint(3, 12)
    total = keyword + bullet + formatting + structure
    factor = base / total if total > 0 else 1
    return {
        "ats_score": base,
        "score_breakdown": {
            "keyword_coverage": round(keyword * factor),
            "bullet_quality": round(bullet * factor),
            "formatting": round(formatting * factor),
            "structure": round(structure * factor)
        }
    }

skill_pools = {
    "Software Engineer": {
        "matched": ["Python", "REST APIs", "PostgreSQL", "Docker", "Git", "CI/CD", "Flask"],
        "missing": ["Java", "Kubernetes", "GraphQL", "Kafka", "AWS", "microservices"]
    },
    "Data Scientist": {
        "matched": ["Python", "machine learning", "scikit-learn", "A/B testing", "SQL", "PySpark", "Tableau"],
        "missing": ["TensorFlow", "NLP", "Hadoop", "SageMaker", "deep learning"]
    },
    "DevOps Engineer": {
        "matched": ["AWS", "Kubernetes", "Docker", "Terraform", "Python", "Prometheus", "ArgoCD"],
        "missing": ["Azure", "GitLab CI", "Istio", "service mesh"]
    },
    "Product Manager": {
        "matched": ["product roadmap", "agile", "user research", "data-driven", "stakeholder management", "KPIs"],
        "missing": ["SQL", "AI/ML products", "product analytics tools", "technical background"]
    },
    "Frontend Developer": {
        "matched": ["JavaScript", "TypeScript", "React", "CSS", "Cypress", "responsive design"],
        "missing": ["Vue.js", "Redux", "accessibility", "Webpack", "Next.js"]
    },
    "Backend Developer": {
        "matched": ["Java", "Python", "microservices", "PostgreSQL", "MongoDB", "Kafka", "Docker"],
        "missing": ["Go", "gRPC", "event sourcing", "distributed systems"]
    },
    "ML Engineer": {
        "matched": ["Python", "PyTorch", "NLP", "ML pipeline", "Docker", "model deployment"],
        "missing": ["MLflow", "Kubeflow", "distributed training", "GCP", "TensorFlow"]
    },
    "Full Stack Developer": {
        "matched": ["JavaScript", "React", "Node.js", "PostgreSQL", "MongoDB", "REST APIs", "Git"],
        "missing": ["TypeScript", "GraphQL", "microservices", "Django"]
    },
    "Cybersecurity Analyst": {
        "matched": ["CISSP", "SIEM", "vulnerability assessment", "penetration testing", "zero-trust"],
        "missing": ["QRadar", "SOC 2", "ISO 27001", "DevSecOps", "cloud security"]
    },
    "Cloud Architect": {
        "matched": ["AWS", "multi-cloud", "Terraform", "Kubernetes", "cost optimization", "networking"],
        "missing": ["hybrid cloud", "serverless", "FinOps", "Azure"]
    }
}

weak_bullets_pool = [
    {"original": "Responsible for maintaining CI/CD pipelines using Jenkins and Docker", "issue": "Starts with passive phrase; lacks quantifiable impact", "improved": "Streamlined CI/CD pipelines using Jenkins and Docker, reducing deployment failures by 40% and cutting release cycles from 2 weeks to 3 days"},
    {"original": "Helped with bug fixes and code reviews", "issue": "Vague action verb; no metrics", "improved": "Resolved 150+ bugs across 3 modules and conducted 200+ code reviews, improving code quality score by 25%"},
    {"original": "Participated in daily standup meetings and sprint planning", "issue": "Describes attendance, not accomplishments", "improved": "Drove sprint planning for 6-person team, consistently delivering 95% of committed story points across 12 sprints"},
    {"original": "Worked on improving test coverage from 40% to 75%", "issue": "Weak verb; metric could be stronger", "improved": "Increased test coverage from 40% to 75% by implementing 200+ unit and integration tests, reducing production bugs by 30%"},
    {"original": "Responsible for data cleaning and preprocessing tasks", "issue": "Passive voice; no scale or impact", "improved": "Engineered automated data preprocessing pipeline handling 5M+ records daily, reducing manual effort by 80%"},
    {"original": "Assisted in building predictive models using scikit-learn", "issue": "Weak verb suggesting minimal contribution", "improved": "Co-developed 3 predictive models achieving 88% accuracy, informing $500K in marketing budget allocation"},
    {"original": "Created dashboards using Tableau for real-time monitoring", "issue": "Missing quantifiable impact and scope", "improved": "Designed 12 interactive Tableau dashboards tracking 50+ KPIs, adopted by 30+ stakeholders"},
    {"original": "Maintained Linux servers and managed deployments", "issue": "Generic; no scale or achievements", "improved": "Administered 50+ Linux servers achieving 99.95% uptime and automated deployments reducing release time from 4 hours to 15 minutes"},
    {"original": "Set up monitoring using Prometheus and Grafana", "issue": "Lacks scope and business impact", "improved": "Implemented monitoring with Prometheus and Grafana covering 200+ metrics, reducing mean time to detection by 70%"},
    {"original": "Managed MongoDB databases", "issue": "Extremely vague; no scale details", "improved": "Managed 15 MongoDB clusters totaling 2TB, optimizing indexing that reduced query latency by 60%"},
    {"original": "Wrote unit tests for critical business logic", "issue": "Missing quantifiable details", "improved": "Authored 300+ unit tests covering payment and order logic, achieving 95% coverage and preventing 3 production incidents"},
    {"original": "Developed websites for various clients using React", "issue": "Vague scope; no metrics", "improved": "Delivered 8 React applications on schedule, generating $200K in revenue with 95+ Lighthouse scores"},
    {"original": "Fixed bugs reported by QA team", "issue": "Passive response; no scale", "improved": "Resolved 200+ QA issues with 98% first-fix rate, reducing regressions by 45% through root cause analysis"},
    {"original": "Tracked KPIs and metrics for product performance", "issue": "Lacks specificity and impact", "improved": "Tracked 25+ KPIs using Amplitude, identifying optimizations increasing engagement by 20% and reducing churn by 15%"},
    {"original": "Monitored security alerts and investigated incidents", "issue": "Routine task; lacks scale", "improved": "Triaged 1,000+ security alerts monthly, investigating 50+ incidents with average 2-hour resolution"},
    {"original": "Managed AWS EC2 instances and S3 buckets", "issue": "Generic without scale", "improved": "Managed 100+ EC2 instances and 50+ S3 buckets totaling 10TB, reducing storage costs by 35%"},
    {"original": "Assisted senior engineers with model deployment", "issue": "Suggests minimal ownership", "improved": "Deployed 5 ML models using Docker and SageMaker, reducing model shipping time by 50%"},
    {"original": "Created data visualization dashboards", "issue": "No specifics on tools or impact", "improved": "Built 8 dashboards using Plotly and Streamlit, enabling real-time monitoring for 20+ stakeholders"},
    {"original": "Worked on automating infrastructure provisioning", "issue": "Weak verb; no results", "improved": "Automated provisioning using Terraform, reducing server setup from 3 days to 30 minutes"}
]

formatting_issues_pool = [
    "Resume exceeds recommended 1-page length for experience level",
    "Inconsistent date formatting across entries",
    "Missing quantifiable metrics in 60% of bullet points",
    "Skills section uses paragraph format instead of categorized list",
    "Contact information missing LinkedIn URL",
    "Bullet points use inconsistent punctuation",
    "Summary section is too generic",
    "Work experience gaps not addressed",
    "Section headings use inconsistent capitalization",
    "Too many bullet points per role (recommend 3-5)",
    "No clear hierarchy between senior and junior roles",
    "Skills section mixes technical and soft skills",
    "Resume uses formatting that may not parse in ATS"
]

feedback_templates = [
    "The resume shows strong skills in {matched} but lacks keywords for {missing}. Bullet points need stronger verbs and metrics. Restructure to highlight relevant experience first.",
    "Good alignment with requirements in {matched}. Falls short on {missing} from the job description. Several bullets use passive language needing stronger impact statements.",
    "Relevant experience present but formatting needs ATS optimization. Strengths: {matched}. Gaps: {missing}. Add metrics to at least 70% of bullets.",
    "Strong keyword coverage for {matched}. Main improvements: add {missing} keywords, strengthen bullets with quantifiable results, improve formatting consistency.",
    "Needs significant optimization. {matched} well-represented but {missing} should be highlighted. Multiple bullets lack measurable impact.",
    "Moderately aligned resume. Strengths in {matched} evident. Gaps in {missing} need addressing. Some bullets too verbose and lack specificity."
]

dataset = []
sample_id = 0

for resume in resumes:
    for jd in job_descriptions:
        for _ in range(2):
            sample_id += 1
            scores = generate_score()
            resume_name = resume["name"]
            skills_data = skill_pools.get(resume_name, skill_pools["Software Engineer"])

            n_matched = random.randint(2, len(skills_data["matched"]))
            n_missing = random.randint(1, len(skills_data["missing"]))
            matched = random.sample(skills_data["matched"], n_matched)
            missing = random.sample(skills_data["missing"], n_missing)

            n_weak = random.randint(1, 3)
            weak_bullets = random.sample(weak_bullets_pool, n_weak)

            n_fmt = random.randint(1, 4)
            fmt_issues = random.sample(formatting_issues_pool, n_fmt)

            tpl = random.choice(feedback_templates)
            feedback = tpl.format(
                matched=", ".join(matched[:3]),
                missing=", ".join(missing[:2])
            )

            output = {
                "ats_score": scores["ats_score"],
                "score_breakdown": scores["score_breakdown"],
                "matched_skills": matched,
                "missing_skills": missing,
                "weak_bullets": weak_bullets,
                "formatting_issues": fmt_issues,
                "overall_feedback": feedback
            }

            sample = {
                "instruction": "Evaluate the following resume against the job description and provide a detailed ATS (Applicant Tracking System) compliance analysis. Return a structured JSON evaluation including ATS score, score breakdown, matched skills, missing skills, weak bullet analysis with improvements, formatting issues, and overall feedback.",
                "input": f"RESUME:\n{resume['text']}\n\nJOB DESCRIPTION:\n{jd['text']}",
                "output": json.dumps(output, indent=2)
            }
            dataset.append(sample)

random.shuffle(dataset)

os.makedirs("data", exist_ok=True)
with open("data/raw_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Generated {len(dataset)} samples")
print(f"Saved to data/raw_dataset.json")
