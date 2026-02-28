# Training Data for Resume Bullet Point Optimization

This directory contains training data for fine-tuning the LLM to generate better resume bullet point rewrites.

## Data Structure

### Format: JSONL (JSON Lines)
Each line is a JSON object representing a training example.

### Schema

```json
{
  "id": "unique_identifier",
  "weak_bullet": "Original weak resume bullet point",
  "strong_bullet": "Improved version with specific improvements",
  "job_context": "Relevant job description snippet (optional)",
  "improvements": [
    "Added action verb",
    "Included quantifiable metric",
    "Made more specific"
  ],
  "category": "experience|project|achievement"
}
```

## Sample Training Pairs

### Example 1: Adding Metrics
**Weak:** "Responsible for managing team projects"
**Strong:** "Led 5-person engineering team to deliver 3 major features, reducing deployment time by 40%"

### Example 2: Action Verbs
**Weak:** "Worked on improving the customer experience"
**Strong:** "Redesigned customer onboarding flow, increasing user retention by 25% across 10K+ monthly users"

### Example 3: Specificity
**Weak:** "Helped with data analysis tasks"
**Strong:** "Analyzed 2M+ transaction records using Python/SQL, identifying $500K in cost-saving opportunities"

## Data Collection Strategy

1. **Synthetic Generation:** Use GPT-4/Claude to generate weak-strong pairs
2. **Real Resume Mining:** Collect anonymized before/after examples from career services
3. **Expert Annotation:** Have HR professionals review and annotate bullet points
4. **Diversity:** Cover multiple industries, roles, and seniority levels

## Fine-tuning Approach (Future Enhancement)

For Phase 2, we're using **few-shot prompting** instead of full fine-tuning for faster MVP delivery.

Future fine-tuning process:
1. Collect 1,000-5,000 high-quality examples
2. Use LoRA/QLoRA for parameter-efficient fine-tuning
3. Fine-tune on task: "Given a weak bullet and job context, generate improved bullet"
4. Evaluate using BLEU, ROUGE, and human evaluation

## Current Status

- ✅ Data structure defined
- ✅ Sample examples created
- ⏳ Using few-shot prompting (no fine-tuning required for MVP)
- 🔜 Future: Collect production data for fine-tuning
