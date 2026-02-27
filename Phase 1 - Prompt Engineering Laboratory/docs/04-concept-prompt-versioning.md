---
date: 2025-02-27
type: concept
topic: "Prompt Versioning & A/B Testing"
project: "Phase 1 - Prompt Engineering Laboratory"
status: complete
---

# Learning: Prompt Versioning & A/B Testing

## In My Own Words

Prompts are code. Just as you version your Python modules, you should version your prompts. A prompt version captures not just the template text, but also the model, parameters, and expected output schema. **Semantic versioning** (MAJOR.MINOR.PATCH) provides a clear language for communicating the nature of changes. A/B testing then lets you compare versions with data, not gut feel.

## Why This Matters

- **Reproducibility**: Know exactly which prompt produced which output
- **Regression detection**: Catch when a change makes things worse
- **Team collaboration**: Multiple people can work on prompts with clear history
- **Rollback capability**: Go back to a working version if something breaks
- **Data-driven optimization**: Compare versions quantitatively

---

## Semantic Versioning for Prompts

Adapted from [semver.org](https://semver.org):

### Version Format: `MAJOR.MINOR.PATCH`

| Component | When to Increment | Examples |
|-----------|------------------|---------|
| **MAJOR** | Breaking changes to output schema or behavior | Changed JSON schema, different output format, switched task type |
| **MINOR** | New capabilities, backward-compatible | Added a field, improved instructions, added examples |
| **PATCH** | Bug fixes, minor wording tweaks | Fixed typo, adjusted phrasing, minor clarification |

### Examples

```
v1.0.0 → v1.0.1  # Fixed typo in instructions (PATCH)
v1.0.1 → v1.1.0  # Added few-shot examples (MINOR)
v1.1.0 → v1.2.0  # Added confidence score to output (MINOR - additive)
v1.2.0 → v2.0.0  # Changed output from text to JSON (MAJOR - breaking)
v2.0.0 → v2.0.1  # Adjusted temperature from 0.3 to 0.2 (PATCH)
```

---

## Prompt as a Versioned Artifact

### What to Version

A prompt version should capture **everything** needed to reproduce results:

```python
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class PromptVersion(BaseModel):
    """A versioned prompt artifact."""
    # Identity
    name: str               # e.g., "document_summarizer"
    version: str            # e.g., "1.2.0"
    
    # Template
    system_prompt: str
    user_prompt_template: str
    variables: list[str]    # Required template variables
    
    # Model configuration
    model: str              # e.g., "gpt-4o"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    
    # Output specification
    output_schema: Optional[dict] = None  # JSON Schema
    output_format: str = "text"  # "text", "json", "structured"
    
    # Few-shot examples
    examples: list[dict] = []  # [{"input": ..., "output": ...}]
    
    # Metadata
    description: str = ""
    author: str = ""
    created_at: datetime = datetime.now()
    changelog: str = ""
    tags: list[str] = []
    
    # Evaluation baseline
    baseline_scores: Optional[dict] = None  # From last evaluation
```

### Storage Options

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **YAML/JSON files + Git** | Simple, version controlled, diffable | No query capability | Small-medium projects |
| **Python objects** | Type-safe, IDE support, testable | Harder to edit by non-devs | Framework/library |
| **Database** | Queryable, dynamic, multi-user | Complex, need migration strategy | Enterprise platforms |
| **Hybrid (files + registry)** | Best of both worlds | More complexity | Production systems |

### File-Based Storage Example

```
prompts/
├── summarization/
│   ├── v1.0.0.yaml
│   ├── v1.1.0.yaml
│   ├── v2.0.0.yaml
│   └── latest.yaml → v2.0.0.yaml (symlink)
├── classification/
│   ├── v1.0.0.yaml
│   └── latest.yaml → v1.0.0.yaml
└── extraction/
    ├── v1.0.0.yaml
    └── latest.yaml → v1.0.0.yaml
```

### YAML Format

```yaml
# prompts/summarization/v1.2.0.yaml
name: document_summarizer
version: "1.2.0"
description: "Summarizes documents into structured bullet points"
author: "bas"
created_at: "2025-02-27"
changelog: "Added confidence score to output"

model:
  name: "gpt-4o"
  temperature: 0.2
  max_tokens: 500

system_prompt: |
  You are an expert document analyst. You create concise, 
  accurate summaries that capture the essential information.

user_prompt_template: |
  Summarize the following {document_type} document.
  
  Requirements:
  - Provide exactly {num_points} key points
  - Each point should be one sentence
  - Include a confidence score (0.0-1.0)
  
  Document:
  ---
  {document_text}
  ---

variables:
  - document_type
  - num_points
  - document_text

output_format: json
output_schema:
  type: object
  properties:
    key_points:
      type: array
      items:
        type: string
    confidence:
      type: number
      minimum: 0
      maximum: 1

examples:
  - input:
      document_type: "financial report"
      num_points: 3
      document_text: "Q3 revenue grew 15%..."
    output:
      key_points:
        - "Q3 revenue increased 15% year-over-year"
        - "Operating margins improved to 23%"
        - "Company raised full-year guidance"
      confidence: 0.92

tags:
  - summarization
  - production
  - financial
```

---

## A/B Testing Framework

### Concepts

A/B testing for prompts compares two (or more) prompt versions against the same test set:

```
                    ┌──────────────────┐
                    │   Test Dataset   │
                    │   (N test cases) │
                    └────────┬─────────┘
                             │
                    ┌────────┴─────────┐
                    │                  │
              ┌─────▼─────┐     ┌─────▼─────┐
              │ Prompt v1  │     │ Prompt v2  │
              │ (Control)  │     │ (Variant)  │
              └─────┬─────┘     └─────┬─────┘
                    │                  │
              ┌─────▼─────┐     ┌─────▼─────┐
              │ Outputs A  │     │ Outputs B  │
              └─────┬─────┘     └─────┬─────┘
                    │                  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   Evaluation     │
                    │   (Same metrics) │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   Comparison     │
                    │   Report         │
                    └──────────────────┘
```

### A/B Test Design

```python
from dataclasses import dataclass

@dataclass
class ABTestConfig:
    """Configuration for a prompt A/B test."""
    test_name: str
    prompt_a: PromptVersion  # Control
    prompt_b: PromptVersion  # Variant
    test_dataset: list[TestCase]
    metrics: list[str]  # ["accuracy", "relevance", "latency", "cost"]
    num_runs_per_case: int = 1  # >1 for consistency measurement
    significance_threshold: float = 0.05  # For statistical testing
```

### Statistical Significance

For categorical outcomes (pass/fail), use **McNemar's test** or **chi-squared**.
For continuous scores, use **paired t-test** or **Wilcoxon signed-rank test**:

```python
from scipy import stats

def compare_scores(scores_a: list[float], scores_b: list[float]) -> dict:
    """Compare evaluation scores between two prompt versions."""
    # Paired t-test (assumes normal distribution)
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    
    # Wilcoxon signed-rank (non-parametric alternative)
    w_stat, w_p_value = stats.wilcoxon(scores_a, scores_b)
    
    mean_a = sum(scores_a) / len(scores_a)
    mean_b = sum(scores_b) / len(scores_b)
    
    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "difference": mean_b - mean_a,
        "percent_change": ((mean_b - mean_a) / mean_a) * 100 if mean_a > 0 else 0,
        "t_test_p_value": p_value,
        "wilcoxon_p_value": w_p_value,
        "is_significant": p_value < 0.05,
        "winner": "B" if mean_b > mean_a and p_value < 0.05 
                  else "A" if mean_a > mean_b and p_value < 0.05 
                  else "tie"
    }
```

### A/B Test Report

```python
@dataclass
class ABTestReport:
    test_name: str
    prompt_a_version: str
    prompt_b_version: str
    
    # Per-metric results
    metrics: dict[str, dict]  # metric_name -> comparison stats
    
    # Overall
    total_test_cases: int
    total_api_calls: int
    total_cost: float
    
    # Recommendation
    winner: str  # "A", "B", or "tie"
    confidence: float
    summary: str
```

---

## Prompt Regression Testing

Run the full test suite whenever a prompt changes:

```python
class PromptRegressionTest:
    """Detect when prompt changes cause quality degradation."""
    
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold  # Max acceptable quality drop
    
    async def run(
        self, 
        old_version: PromptVersion, 
        new_version: PromptVersion,
        test_cases: list[TestCase]
    ) -> dict:
        """Run regression test between two prompt versions."""
        old_scores = await self.evaluate(old_version, test_cases)
        new_scores = await self.evaluate(new_version, test_cases)
        
        regressions = []
        improvements = []
        
        for metric in old_scores:
            diff = new_scores[metric]["mean"] - old_scores[metric]["mean"]
            if diff < -self.threshold:
                regressions.append({
                    "metric": metric,
                    "old_score": old_scores[metric]["mean"],
                    "new_score": new_scores[metric]["mean"],
                    "change": diff
                })
            elif diff > self.threshold:
                improvements.append({
                    "metric": metric,
                    "old_score": old_scores[metric]["mean"],
                    "new_score": new_scores[metric]["mean"],
                    "change": diff
                })
        
        return {
            "passed": len(regressions) == 0,
            "regressions": regressions,
            "improvements": improvements,
            "summary": f"{'PASS' if not regressions else 'FAIL'}: "
                      f"{len(improvements)} improvements, "
                      f"{len(regressions)} regressions"
        }
```

---

## Best Practices

- ✅ **Version everything**: template, model, parameters, examples
- ✅ **Never edit a released version** — create a new one
- ✅ **Keep a changelog** for each version
- ✅ **Run regression tests** before promoting a new version
- ✅ **Track evaluation scores** as part of the version metadata
- ✅ **Use "latest" aliases** for production references
- ❌ Don't compare versions without the **same test set**
- ❌ Don't declare a winner without **statistical significance**
- ❌ Don't ignore **cost** — a 2% quality improvement isn't worth 5× cost

---

## Application to My Project

### How I'll Use This

1. **YAML-based prompt storage** with semantic versioning
2. **PromptVersion model** (Pydantic) as the core data structure
3. **Built-in A/B testing** that runs both versions and compares
4. **Regression tests** that run automatically on version changes
5. **Version registry** that tracks all versions with their baseline scores

### Decisions to Make

- [ ] YAML vs JSON for prompt storage?
- [ ] How to handle version promotion (staging → production)?
- [ ] Minimum test set size for statistical significance?
- [ ] How to version the test sets themselves?

---

## Resources for Deeper Learning

- [Semantic Versioning](https://semver.org/) — The versioning standard
- [Promptfoo](https://promptfoo.dev/) — Mature CLI tool for prompt testing and comparison
- [Statistical Hypothesis Testing](https://docs.scipy.org/doc/scipy/reference/stats.html) — SciPy statistics module
- [MLflow](https://mlflow.org/) — ML experiment tracking (applicable to prompt experiments)
